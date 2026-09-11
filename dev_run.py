"""Dev driver: train one model and print the key diagnostics.  Usage:
python dev_run.py --mode min --steps 2000 --pos learned --out runs/min.json"""
import argparse, json, time, torch
import slt_sorting_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="pntr"); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--pos", default="learned"); ap.add_argument("--d", type=int, default=128)
ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--probe_layer", type=int, default=1); ap.add_argument("--state_frac", type=float, default=0.7)
ap.add_argument("--lr_joint", type=float, default=3e-4); ap.add_argument("--state_noise", type=float, default=0.02)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nmin", type=int, default=4); ap.add_argument("--nmax", type=int, default=10)
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--out", default=None)
ap.add_argument("--save", default=None)
a = ap.parse_args()
torch.set_num_threads(a.threads)
cfg = L.Config(d_model=a.d, n_heads=a.heads, n_layers=a.layers, d_mlp=4 * a.d, pos=a.pos, probe_layer=a.probe_layer)
t0 = time.time()
model, log = L.train_model(a.mode, cfg, steps=a.steps, batch=a.batch, lr=a.lr, lr_joint=a.lr_joint, n_range=(a.nmin, a.nmax),
                           state_frac=a.state_frac, state_noise=a.state_noise, seed=a.seed, log_every=max(1, a.steps // 12))
print(f"trained in {time.time()-t0:.0f}s, params={L.n_params(model)}")
res = {"args": vars(a), "train": L.train_loss_estimate(model, n_range=(a.nmin, a.nmax))}
print("train:", res["train"])
res["state_fidelity"] = L.state_fidelity(model, n_range=(a.nmin, a.nmax))
print("state fidelity:", res["state_fidelity"])
res["lengths"] = L.evaluate_lengths(model, [4, 6, 8, 10, 11, 12, 14, 16, 20, 24, 32], B=512)
for r in res["lengths"]:
    print(f"  n={r['n']:2d} exact={r['exact']:.3f} token={r['token']:.3f} tf_token={r['tf_token']:.3f} ce={r['ce']:.4f}")
res["dups"] = {f"alphabet{k}": L.evaluate(model, 10, B=512, alphabet=k) for k in [4, 2, 1]}
for k, r in res["dups"].items():
    print(f"  n=10 {k}: exact={r['exact']:.3f} token={r['token']:.3f}")
res["prefix"] = L.prefix_sensitivity(model, n=10, B=1024)
print("prefix sensitivity:", res["prefix"])
res["probes_resid"] = L.probe_report(model, which="resid")
res["probes_final"] = L.probe_report(model, which="final")
print("probes@resid:", res["probes_resid"]); print("probes@final:", res["probes_final"])
if a.pos == "learned":
    pe = model.pos_emb.detach()
    res["pos_norms"] = pe.norm(dim=1).tolist()
    print("pos-emb norms (first 24):", [round(v, 2) for v in res["pos_norms"][:24]])
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
if a.save:
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, a.save)
