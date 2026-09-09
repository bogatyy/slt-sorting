"""Extended length generalisation on the committed checkpoints (runs on CPU in a few minutes).
Writes results/ood_lengths_extended.csv and figures/ood_length_extended.png."""
import torch, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import slt_sorting_lib as L
torch.set_num_threads(4)
device = "cpu"
models = {}
for name in ["MIN", "HIST", "NONE"]:
    ck = torch.load(f"results/model_{name}.pt", map_location=device)
    m = L.SortingTransformer(L.Config(**ck["cfg"])); m.load_state_dict(ck["state"]); m.eval(); models[name] = m
LENGTHS = [10, 16, 20, 24, 32, 40, 48, 64, 80, 100, 128]
rows = []
for name, m in models.items():
    for n in LENGTHS:
        r = L.evaluate(m, n, B=512 if n <= 64 else 256, seed=1234, device=device)
        rows.append({"model": name, **r}); print(f"{name:5s} n={n:3d} exact={r['exact']:.3f} token={r['token']:.3f} teacher-forced token={r['tf_token']:.3f}", flush=True)
df = pd.DataFrame(rows); df.to_csv("results/ood_lengths_extended.csv", index=False)

# Where does MIN go wrong on long inputs?  Split teacher-forced errors by whether the correct next digit
# repeats the previous one (a duplicate: the model must decide "is another copy left?") or advances.
diag = []
with torch.no_grad():
    for name in ["MIN", "NONE"]:
        for n in [32, 64, 128]:
            g = torch.Generator(device=device).manual_seed(99)
            x = L.sample_inputs(256, n, device=device, generator=g)
            out, y = L.forward_sort(models[name], x)
            pred = out["logits"][:, n:, :].argmax(-1)
            err = (pred != y)
            repeat = torch.cat([torch.zeros(256, 1, dtype=torch.bool), y[:, 1:] == y[:, :-1]], 1)
            diag.append({"model": name, "n": n, "error rate at 'repeat' steps": err[repeat].float().mean().item(),
                         "error rate at 'advance' steps": err[~repeat].float().mean().item(),
                         "share of errors at 'repeat' steps": (err & repeat).sum().item() / max(1, err.sum().item())})
df_diag = pd.DataFrame(diag); df_diag.to_csv("results/ood_lengths_extended_diagnostic.csv", index=False); print(df_diag.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for name in models:
    sub = df[df.model == name]
    axes[0].plot(sub.n, sub.exact, "o-", label=name); axes[1].plot(sub.n, sub.token, "o-", label=name)
for ax, t in zip(axes, ["exact-match accuracy (free-running)", "per-token accuracy (free-running)"]):
    ax.axvspan(3.5, 10.5, color="green", alpha=0.08, label="training lengths"); ax.set_xscale("log"); ax.set_xticks(LENGTHS); ax.set_xticklabels(LENGTHS)
    ax.set_xlabel("input length n (log scale)"); ax.set_title(t); ax.set_ylim(-0.02, 1.02); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("figures/ood_length_extended.png", dpi=110); print("saved")
