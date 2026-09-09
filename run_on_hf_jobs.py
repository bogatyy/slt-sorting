"""
Execute a notebook on a rented Hugging Face GPU ("HF Jobs") and bring the executed notebook back.

    python run_on_hf_jobs.py launch --notebook slt_sorting.ipynb --flavor t4-small [--files a.py b.py]
    python run_on_hf_jobs.py status <job_id>
    python run_on_hf_jobs.py logs   <job_id>
    python run_on_hf_jobs.py fetch  <run_id> --out results/

Mechanics: the inputs are uploaded to a private HF dataset repo (inputs/<run_id>/), the job
downloads them, runs `papermill`, and uploads outputs/<run_id>/ (executed notebook + figures)
back to the same repo.  Requires HF_TOKEN in the environment (a PRO account for GPU flavors).
"""
import argparse, os, sys, time, json, uuid
from huggingface_hub import HfApi

REPO = os.environ.get("HF_RUNS_REPO", "ivanbogatyy/slt-sorting-runs")
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"

JOB_SCRIPT = r"""
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
pip install -q --no-cache-dir "huggingface_hub>=0.30" papermill nbformat nbconvert ipykernel matplotlib numpy pandas 2>&1 | tail -n 2
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=os.environ["RUNS_REPO"], repo_type="dataset",
                      allow_patterns=[f"inputs/{os.environ['RUN_ID']}/*"], local_dir="/work")
print("downloaded to", p)
EOF
cd /work/inputs/$RUN_ID
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
mkdir -p /work/out
set +e
papermill --kernel python3 --log-output --progress-bar --cwd . "$NOTEBOOK" /work/out/"$NOTEBOOK"
STATUS=$?
set -e
cp -r figures /work/out/ 2>/dev/null || true
cp -r results /work/out/ 2>/dev/null || true
echo "papermill exit status: $STATUS" | tee /work/out/STATUS.txt
python - <<'EOF'
import os
from huggingface_hub import upload_folder
upload_folder(repo_id=os.environ["RUNS_REPO"], repo_type="dataset", folder_path="/work/out",
              path_in_repo=f"outputs/{os.environ['RUN_ID']}")
print("uploaded outputs")
EOF
exit $STATUS
"""


def launch(args):
    api = HfApi()
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    for f in [args.notebook] + args.files:
        api.upload_file(path_or_fileobj=f, path_in_repo=f"inputs/{run_id}/{os.path.basename(f)}",
                        repo_id=REPO, repo_type="dataset")
    job = api.run_job(image=IMAGE, command=["bash", "-c", JOB_SCRIPT],
                      env={"RUNS_REPO": REPO, "RUN_ID": run_id, "NOTEBOOK": os.path.basename(args.notebook),
                           **dict(kv.split("=", 1) for kv in args.env)},
                      secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
                      flavor=args.flavor, timeout=args.timeout, name=f"slt-sorting-{run_id}")
    print(json.dumps({"job_id": job.id, "run_id": run_id, "flavor": args.flavor, "url": getattr(job, "url", None)}))


def status(args):
    job = HfApi().inspect_job(job_id=args.job_id)
    st = job.status
    print(json.dumps({"stage": st.stage, "message": st.message, "created": str(job.created_at)}))


def logs(args):
    for line in HfApi().fetch_job_logs(job_id=args.job_id):
        print(line)


def fetch(args):
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=[f"outputs/{args.run_id}/*"],
                          local_dir=args.out)
    print("downloaded to", os.path.join(p, "outputs", args.run_id))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("launch"); l.add_argument("--notebook", required=True); l.add_argument("--files", nargs="*", default=[])
    l.add_argument("--flavor", default="t4-small"); l.add_argument("--timeout", default="3h"); l.add_argument("--run-id", default=None)
    l.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE environment variables for the job (e.g. SLT_FAST=1)")
    s = sub.add_parser("status"); s.add_argument("job_id")
    g = sub.add_parser("logs"); g.add_argument("job_id")
    f = sub.add_parser("fetch"); f.add_argument("run_id"); f.add_argument("--out", default="hf_runs")
    a = ap.parse_args()
    {"launch": launch, "status": status, "logs": logs, "fetch": fetch}[a.cmd](a)
