"""Dev: LLC sweep on saved models.  python dev_llc.py model1.pt model2.pt ... --eps 1e-4 --gamma 100"""
import argparse, json, math, time, torch, numpy as np
import slt_sorting_lib as L
ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+"); ap.add_argument("--eps", type=float, nargs="+", default=[1e-4])
ap.add_argument("--gamma", type=float, nargs="+", default=[100.0]); ap.add_argument("--nbeta", type=float, nargs="+", default=[None])
ap.add_argument("--chains", type=int, default=2); ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--burnin", type=int, default=200); ap.add_argument("--N", type=int, default=8192)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--batch", type=int, default=256)
a = ap.parse_args()
torch.set_num_threads(a.threads)
data = L.FixedDataset(N=a.N)
nbetas = [nb if nb is not None else data.N / math.log(data.N) for nb in a.nbeta]
print(f"N={data.N} nbetas={nbetas}")
for path in a.models:
    ck = torch.load(path, map_location="cpu")
    cfg = L.Config(**ck["cfg"]); m = L.SortingTransformer(cfg); m.load_state_dict(ck["state"]); m.eval()
    for nbeta in nbetas:
      for eps in a.eps:
        for gamma in a.gamma:
            t0 = time.time()
            r = L.estimate_llc(m, data, eps=eps, gamma=gamma, nbeta=nbeta, n_chains=a.chains, n_steps=a.steps, burnin=a.burnin, batch=a.batch, seed=1)
            tr = r["traces"]
            q = a.steps // 4
            print(f"{path.split('/')[-1]:16s} nb={nbeta:6.0f} eps={eps:.0e} gamma={gamma:6.0f} L0={r['L0']:.5f} llc={r['llc']:9.3f} +- {r['llc_std']:.3f} | trace quarters: {tr[:, :q].mean():.4f} {tr[:, q:2*q].mean():.4f} {tr[:, 2*q:3*q].mean():.4f} {tr[:, 3*q:].mean():.4f} max={tr.max():.3f} ({time.time()-t0:.0f}s)", flush=True)
