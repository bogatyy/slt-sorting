"""Copy an executed run into the repo: python finalize_run.py <run_dir>
- verifies that the executed notebook's code cells equal the freshly built notebook's code cells,
- transplants the current markdown cells (text may have been edited after launch),
- copies figures/ and results/ (checkpoints included) into the repo, and writes results/RUN_INFO.json."""
import sys, os, shutil, json, nbformat
run_dir = sys.argv[1]
fresh = nbformat.read("slt_sorting.ipynb", as_version=4)
ex = nbformat.read(os.path.join(run_dir, "slt_sorting.ipynb"), as_version=4)
assert len(fresh.cells) == len(ex.cells), (len(fresh.cells), len(ex.cells))
for i, (a, b) in enumerate(zip(fresh.cells, ex.cells)):
    assert a.cell_type == b.cell_type, i
    if a.cell_type == "code":
        assert a.source.strip() == b.source.strip(), f"code cell {i} differs"
    else:
        b.source = a.source
errs = [(i, o["ename"]) for i, c in enumerate(ex.cells) if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
assert not errs, errs
# strip papermill's per-cell metadata noise but keep execution timing in RUN_INFO
durations = {i: c.get("metadata", {}).get("papermill", {}).get("duration") for i, c in enumerate(ex.cells) if c.cell_type == "code"}
for c in ex.cells:
    c.metadata.pop("papermill", None)
    if c.cell_type == "code":
        c.execution_count = None
ex.metadata.pop("papermill", None)
nbformat.write(ex, "slt_sorting.ipynb")
for sub in ["figures", "results"]:
    if os.path.isdir(sub):
        shutil.rmtree(sub)
    shutil.copytree(os.path.join(run_dir, sub), sub)
info = {"run_dir": os.path.basename(run_dir.rstrip("/")), "total_cell_seconds": sum(v or 0 for v in durations.values()),
        "cell_seconds": {str(k): v for k, v in durations.items()}}
json.dump(info, open("results/RUN_INFO.json", "w"), indent=1)
print("finalized:", info["run_dir"], f"{info['total_cell_seconds']/60:.1f} min of cell execution")
