"""Assemble slt_sorting.ipynb from the library file (inlined, split into sections) and the
experiment cells below.  Run: python build_notebook.py   (writes slt_sorting.ipynb)"""
import re, nbformat as nbf

LIB = open("slt_sorting_lib.py").read()
# split the library at its section banners so that each section becomes one code cell
parts = re.split(r"\n(?=# -{40,}\n# )", LIB)
lib_cells = []
for p in parts:
    p = p.strip("\n")
    if p.startswith('"""'):           # module docstring + imports
        p = p.split('"""', 2)[2].strip("\n")
    lib_cells.append(p)

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

cells = []
cells.append(md(open("nb_text/00_title.md").read()))
cells.append(md(open("nb_text/01_background.md").read()))
cells.append(md(open("nb_text/02_design.md").read()))
cells.append(md("## Code\n\nEverything below is self-contained (also available as `slt_sorting_lib.py` in the repo)."))
for c in lib_cells:
    cells.append(code(c))

cells.append(md(open("nb_text/03_setup.md").read()))
cells.append(code(r'''
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)

FAST = os.environ.get("SLT_FAST", "0") == "1"      # SLT_FAST=1 gives a ~5-minute CPU run with fewer steps/chains
print("device:", DEVICE, "| torch", torch.__version__, "| fast mode:", FAST)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

CFG = Config(d_model=128, n_heads=4, n_layers=2, d_mlp=512, max_len=96, pos="none", probe_layer=1)
TRAIN = dict(steps=1500 if FAST else 6000, state_frac=0.8, batch=256, lr=1e-3, lr_joint=1e-4, wd=0.01,
             n_range=(4, 10), state_noise=0.1, seed=0, log_every=250)
OOD_LENGTHS = [4, 6, 8, 10, 11, 12, 13, 14, 16, 18, 20, 24, 28, 32]
MODES = {"PNTR": "pntr", "HIST": "hist", "NONE": "none"}   # NONE = no auxiliary state (what does SGD find on its own?)
print("model parameters:", n_params(SortingTransformer(CFG)))
'''))

cells.append(md(open("nb_text/04_train.md").read()))
cells.append(code(r'''
models, logs = {}, {}
for name, mode in MODES.items():
    print(f"\n===== training {name} (aux state = {mode}) =====")
    models[name], logs[name] = train_model(mode, CFG, device=DEVICE, **TRAIN)
'''))
cells.append(code(r'''
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for name, lg in logs.items():
    steps = [r["step"] for r in lg]
    axes[0].plot(steps, [max(r["ce"], 1e-7) for r in lg], label=name)
    st = [(r["step"], r["state_mse"]) for r in lg if "state_mse" in r]
    if st:
        axes[1].plot(*zip(*st), label=name)
    ro = [(r["step"], r["readout_ce"]) for r in lg if "readout_ce" in r]
    if ro:
        axes[2].plot(*zip(*ro), label=name)
S1 = int(TRAIN["steps"] * TRAIN["state_frac"])
for ax in axes:
    ax.axvline(S1, color="k", ls=":", lw=0.8)
axes[0].set_yscale("log"); axes[0].set_title("sorting loss of the composed model (body -> state -> read-out)"); axes[0].set_xlabel("step"); axes[0].legend()
axes[1].set_yscale("log"); axes[1].set_title("state MSE (natural units); dotted line = end of state phase"); axes[1].set_xlabel("step"); axes[1].legend()
axes[2].set_yscale("log"); axes[2].set_title("read-out loss on the *true* (noised) state, state phase"); axes[2].set_xlabel("step"); axes[2].legend()
plt.tight_layout(); plt.savefig("figures/training_curves.png"); plt.show()
'''))

cells.append(code(r'''
# Which state does each trained model actually hold in its bottleneck?  (MSE against each algorithm's
# state in natural units; bit accuracy of the availability slots; accuracy of the rounded count slots.)
fid = {name: state_fidelity(m, n_range=TRAIN["n_range"], device=DEVICE) for name, m in models.items()}
pd.DataFrame(fid).T
'''))
cells.append(md(open("nb_text/05_same_loss.md").read()))
cells.append(code(r'''
train_stats = {name: train_loss_estimate(m, n_range=TRAIN["n_range"], device=DEVICE) for name, m in models.items()}
df_train = pd.DataFrame(train_stats).T
df_train.columns = ["CE per token", "NLL per sequence", "exact-match acc (teacher forced)"]
df_train.to_csv("results/train_loss.csv")
df_train
'''))

cells.append(md(open("nb_text/06_ood_length.md").read()))
cells.append(code(r'''
ood = {name: evaluate_lengths(m, OOD_LENGTHS, B=1024, device=DEVICE) for name, m in models.items()}
rows = []
for name, rs in ood.items():
    for r in rs:
        rows.append({"model": name, **r})
df_ood = pd.DataFrame(rows); df_ood.to_csv("results/ood_lengths.csv", index=False)
piv = df_ood.pivot(index="n", columns="model", values="exact")
piv.columns = [f"{c}: exact-match acc" for c in piv.columns]
piv
'''))
cells.append(code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for name, rs in ood.items():
    ns = [r["n"] for r in rs]
    axes[0].plot(ns, [r["exact"] for r in rs], "o-", label=name)
    axes[1].plot(ns, [r["token"] for r in rs], "o-", label=name)
for ax, t in zip(axes, ["exact-match accuracy (free-running)", "per-token accuracy (free-running)"]):
    ax.axvspan(TRAIN["n_range"][0] - 0.5, TRAIN["n_range"][1] + 0.5, color="green", alpha=0.08, label="training lengths")
    ax.set_xlabel("input length n"); ax.set_title(t); ax.set_ylim(-0.02, 1.02); ax.legend()
plt.tight_layout(); plt.savefig("figures/ood_length.png"); plt.show()
'''))
cells.append(code(r'''
# What does the failure look like?  Show a few OOD examples for each model.
g = torch.Generator(device=DEVICE).manual_seed(3)
x = sample_inputs(3, 16, device=DEVICE, generator=g)
for i in range(3):
    print("input :", " ".join(map(str, x[i].tolist())))
    print("sorted:", " ".join(map(str, x[i].sort().values.tolist())))
    for name, m in models.items():
        yhat = generate(m, x[i:i+1])[0].tolist()
        ok = yhat == x[i].sort().values.tolist()
        print(f"{name:5s}:", " ".join(map(str, yhat)), "" if ok else "  <-- wrong")
    print()
'''))

cells.append(md(open("nb_text/07_ood_dups.md").read()))
cells.append(code(r'''
dup_rows = []
for name, m in models.items():
    for alpha in [10, 5, 3, 2, 1]:
        r = evaluate(m, 10, B=1024, alphabet=alpha, device=DEVICE)
        dup_rows.append({"model": name, "alphabet": f"{{0..{alpha-1}}}", "max count (typical)": "", **{k: r[k] for k in ["exact", "token"]}})
df_dup = pd.DataFrame(dup_rows).drop(columns=["max count (typical)"]); df_dup.to_csv("results/ood_duplicates.csv", index=False)
df_dup.pivot(index="alphabet", columns="model", values="exact")
'''))

cells.append(md(open("nb_text/08_mechanistic.md").read()))
cells.append(code(r'''
ps = {name: prefix_sensitivity(m, n=10, B=2048, device=DEVICE) for name, m in models.items()}
df_ps = pd.DataFrame(ps).T; df_ps.to_csv("results/prefix_sensitivity.csv")
df_ps
'''))
cells.append(code(r'''
probe_rows = {}
for name, m in models.items():
    for which in ["resid", "final"]:
        probe_rows[(name, "after layer 1" if which == "resid" else "after layer 2")] = probe_report(m, which=which, device=DEVICE)
df_probe = pd.DataFrame(probe_rows).T; df_probe.to_csv("results/probes.csv")
df_probe
'''))
cells.append(code(r'''
# Attention patterns of every head, at the n decision points, for one example of length 10.
g = torch.Generator(device=DEVICE).manual_seed(5)
x = sample_inputs(1, 10, device=DEVICE, generator=g)
labels = token_labels(x[0])
fig, axes = plt.subplots(len(models), CFG.n_layers * CFG.n_heads, figsize=(2.6 * CFG.n_layers * CFG.n_heads, 2.9 * len(models)))
for r, (name, m) in enumerate(models.items()):
    attn, y = attention_patterns(m, x)
    for l in range(CFG.n_layers):
        for h in range(CFG.n_heads):
            ax = axes[r, l * CFG.n_heads + h]
            A = attn[l][0, h, 10:, :].cpu().numpy()            # rows: decision points (predict y_0..y_9)
            ax.imshow(A, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6)
            ax.set_yticks(range(10)); ax.set_yticklabels([f"y{t}" for t in range(10)], fontsize=6)
            ax.set_title(f"{name} L{l}H{h}", fontsize=8); ax.grid(False)
plt.suptitle(f"attention at the decision points; input = {' '.join(map(str, x[0].tolist()))}  (columns: input digits | separator | emitted digits')", fontsize=9)
plt.tight_layout(); plt.savefig("figures/attention.png"); plt.show()
'''))

cells.append(md(open("nb_text/09_llc.md").read()))
cells.append(code(r'''
data = FixedDataset(N=4096 if FAST else 16384, n_range=TRAIN["n_range"], seed=2024, device=DEVICE)
print(f"LLC dataset: N={data.N} sequences  (theoretical n*beta* = N/log N = {data.N / math.log(data.N):.0f}, too hot for these minima -- see text)")
for name, m in models.items():
    print(f"{name}: L_n(w*) = {data.full_loss(m, sequence_nll):.6f}   (mean per-sequence NLL on the LLC dataset)")
'''))
cells.append(code(r'''
# Sweep on the PNTR model.  Good traces rise above L(w*), level off, agree across chains, and have no spikes.
SWEEP_STEPS, SWEEP_BURN = (300, 150) if FAST else (1500, 750)
GRID = [(10.0, 1e-5, 1e4), (10.0, 1e-5, 1e3), (100.0, 1e-6, 1e4), (100.0, 1e-6, 1e3), (100.0, 1e-5, 1e4), (10.0, 1e-4, 1e4)] if not FAST else [(10.0, 1e-5, 1e4), (100.0, 1e-6, 1e4)]
sweep = {}
for nbeta, eps, gamma in GRID:
    r = estimate_llc(models["PNTR"], data, eps=eps, gamma=gamma, nbeta=nbeta, n_chains=2, n_steps=SWEEP_STEPS, burnin=SWEEP_BURN, batch=256, seed=1)
    sweep[(nbeta, eps, gamma)] = r
    tr = r["traces"]; q = SWEEP_STEPS // 4
    print(f"nbeta={nbeta:5.0f} eps={eps:.0e} gamma={gamma:6.0f}: llc={r['llc']:9.2f} +- {r['llc_std']:.2f} | trace quarters: {np.nanmean(tr[:, :q]):.4f} {np.nanmean(tr[:, q:2*q]):.4f} {np.nanmean(tr[:, 2*q:3*q]):.4f} {np.nanmean(tr[:, 3*q:]):.4f}  max={np.nanmax(tr):.3f}  nan={np.isnan(tr).any()}")
fig, axes = plt.subplots(1, len(sweep), figsize=(3.0 * len(sweep), 3.1))
for ax, ((nbeta, eps, gamma), r) in zip(np.atleast_1d(axes), sweep.items()):
    for c in range(r["traces"].shape[0]):
        ax.plot(r["traces"][c], lw=0.6)
    ax.axhline(r["L0"], color="k", ls="--", lw=0.8); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title(f"nb={nbeta:.0f} eps={eps:.0e} g={gamma:.0f}\nllc={r['llc']:.2f}", fontsize=8); ax.set_xlabel("SGLD step")
np.atleast_1d(axes)[0].set_ylabel("minibatch NLL / sequence")
plt.tight_layout(); plt.savefig("figures/llc_sweep.png"); plt.show()
'''))
cells.append(code(r'''
# Final estimates: identical sampler settings for every model (as recommended when comparing LLCs).
LLC_HP = dict(eps=1e-5, gamma=1e4, nbeta=10.0, batch=256)
LLC_RUN = dict(n_chains=2, n_steps=400, burnin=200) if FAST else dict(n_chains=6, n_steps=2000, burnin=1000)
llc = {}
for name, m in models.items():
    llc[name] = estimate_llc(m, data, seed=7, verbose=True, **LLC_HP, **LLC_RUN)
    print(f"==> {name}: LLC = {llc[name]['llc']:.3f} +- {llc[name]['llc_std']:.3f} (chain std), L_n(w*) = {llc[name]['L0']:.6f}")
# control: nbeta = 0 removes the loss-gradient term (chains only feel noise + localisation)
llc0 = {name: estimate_llc(m, data, seed=7, **{**LLC_HP, "nbeta": 0.0}, **{**LLC_RUN, "n_chains": 2}) for name, m in models.items()}
df_llc = pd.DataFrame({name: {"LLC": r["llc"], "chain std": r["llc_std"], "L_n(w*)": r["L0"], "E[L] - L(w*)": r["llc"] / r["nbeta"],
                              "control nbeta=0: (E[L]-L(w*)) x nbeta": (llc0[name]["traces"][:, LLC_RUN["burnin"]:].mean() - llc0[name]["L0"]) * LLC_HP["nbeta"]}
                       for name, r in llc.items()}).T
df_llc.to_csv("results/llc.csv")
df_llc
'''))
cells.append(code(r'''
# Robustness: the same comparison at a second sampler setting (colder chains, smaller steps).
LLC_HP2 = dict(eps=1e-6, gamma=1e4, nbeta=100.0, batch=256)
LLC_RUN2 = dict(n_chains=2, n_steps=400, burnin=200) if FAST else dict(n_chains=4, n_steps=2500, burnin=1250)
llc2 = {name: estimate_llc(m, data, seed=9, **LLC_HP2, **LLC_RUN2) for name, m in models.items()}
df_llc2 = pd.DataFrame({name: {"LLC (setting 2)": r["llc"], "chain std": r["llc_std"], "trace mean after burn-in": r["traces"][:, LLC_RUN2["burnin"]:].mean(), "L_n(w*)": r["L0"]}
                        for name, r in llc2.items()}).T
df_llc2.to_csv("results/llc_setting2.csv")
df_llc2
'''))
cells.append(code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for name, r in llc.items():
    for c in range(r["traces"].shape[0]):
        axes[0].plot(r["traces"][c], lw=0.6, color={"PNTR": "C0", "HIST": "C1", "NONE": "C2"}[name], alpha=0.8, label=name if c == 0 else None)
axes[0].axvline(LLC_RUN["burnin"], color="k", ls=":", lw=0.8); axes[0].set_xlabel("SGLD step"); axes[0].set_ylabel("minibatch NLL / sequence"); axes[0].set_title("SGLD loss traces (all chains)"); axes[0].legend()
names = list(llc); vals = [llc[k]["llc"] for k in names]; errs = [llc[k]["llc_std"] for k in names]
axes[1].bar(names, vals, yerr=errs, capsize=4, color=["C0", "C1", "C2"][:len(names)]); axes[1].set_title("estimated local learning coefficient"); axes[1].set_ylabel("LLC")
plt.tight_layout(); plt.savefig("figures/llc.png"); plt.show()
'''))

cells.append(code(r'''
# Weight-refined LLC (Wang et al. 2024): sample only one group of parameters at a time, everything else
# frozen at w*.  Same sampler settings as above.  This asks *where* each solution's complexity sits.
GROUPS = {"embeddings": lambda n: "emb" in n,
          "attention": lambda n: ".attn." in n,
          "MLPs": lambda n: ".mlp." in n,
          "state + read-out": lambda n: n.startswith("state") or n.startswith("readout")}
RLLC_RUN = dict(n_chains=2, n_steps=300, burnin=150) if FAST else dict(n_chains=4, n_steps=1200, burnin=600)
rllc = {}
for name, m in models.items():
    for gname, filt in GROUPS.items():
        r = estimate_llc(m, data, seed=11, param_filter=filt, **LLC_HP, **RLLC_RUN)
        rllc[(name, gname)] = {"rLLC": r["llc"], "chain std": r["llc_std"], "#params": r["n_params"]}
        print(f"{name:5s} {gname:18s} ({r['n_params']:7d} params): rLLC = {r['llc']:8.2f} +- {r['llc_std']:.2f}")
df_rllc = pd.DataFrame(rllc).T; df_rllc.to_csv("results/refined_llc.csv")
fig, ax = plt.subplots(figsize=(7, 3.4))
piv = df_rllc["rLLC"].unstack(0)
piv.plot.bar(ax=ax, color=[{"PNTR": "C0", "HIST": "C1", "NONE": "C2"}[c] for c in piv.columns], rot=0)
ax.set_ylabel("weight-refined LLC"); ax.set_title("where does the complexity sit?")
plt.tight_layout(); plt.savefig("figures/refined_llc.png"); plt.show()
'''))
cells.append(md(open("nb_text/09b_seeds.md").read()))
cells.append(code(r'''
# Seeds: train two more models of each kind (different initialisation and data stream, everything else
# identical) and repeat the length-generalisation, prefix and LLC measurements.
SEEDS = [1] if FAST else [1, 2]
seed_models = {}
for seed in SEEDS:
    for name, mode in MODES.items():
        print(f"===== training {name} seed {seed} =====")
        seed_models[(name, seed)] = train_model(mode, CFG, device=DEVICE, verbose=False, **{**TRAIN, "seed": seed})[0]
all_models = {**{(name, TRAIN["seed"]): m for name, m in models.items()}, **seed_models}
seed_rows, seed_ood = [], {}
for (name, seed), m in all_models.items():
    r_llc = llc[name] if seed == TRAIN["seed"] else estimate_llc(m, data, seed=7, **LLC_HP, **{**LLC_RUN, "n_chains": 4})
    oo = ood[name] if seed == TRAIN["seed"] else evaluate_lengths(m, OOD_LENGTHS, B=1024, device=DEVICE)
    pp = ps[name] if seed == TRAIN["seed"] else prefix_sensitivity(m, n=10, B=2048, device=DEVICE)
    seed_ood[(name, seed)] = oo
    seed_rows.append({"model": name, "seed": seed, "L_n(w*)": r_llc["L0"], "LLC": r_llc["llc"], "LLC chain std": r_llc["llc_std"],
                      **{f"exact@n={n}": next(x["exact"] for x in oo if x["n"] == n) for n in [10, 12, 14, 16, 20, 24, 32]},
                      "follows prefix": pp["follows_prefix(PNTR-like)"], "ignores prefix": pp["ignores_prefix(HIST-like)"]})
    print(f"{name:5s} seed {seed}: LLC = {r_llc['llc']:.3f} +- {r_llc['llc_std']:.3f}  exact@20 = {seed_rows[-1]['exact@n=20']:.3f}")
df_seeds = pd.DataFrame(seed_rows).set_index(["model", "seed"]).sort_index(); df_seeds.to_csv("results/seeds.csv")
df_seeds
'''))
cells.append(code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
colors = {"PNTR": "C0", "HIST": "C1", "NONE": "C2"}
for (name, seed), oo in seed_ood.items():
    axes[0].plot([r["n"] for r in oo], [r["exact"] for r in oo], "o-", color=colors[name], alpha=0.6, lw=1, label=name if seed == TRAIN["seed"] else None)
axes[0].axvspan(TRAIN["n_range"][0] - 0.5, TRAIN["n_range"][1] + 0.5, color="green", alpha=0.08)
axes[0].set_xlabel("input length n"); axes[0].set_ylabel("exact-match accuracy"); axes[0].set_title(f"length generalisation, {1 + len(SEEDS)} seeds per model"); axes[0].legend()
for i, name in enumerate(MODES):
    sub = df_seeds.loc[name]
    axes[1].errorbar([i + 0.15 * (k - 1) for k in range(len(sub))], sub["LLC"], yerr=sub["LLC chain std"], fmt="o", color=colors[name], capsize=3)
axes[1].set_xticks(range(len(MODES))); axes[1].set_xticklabels(list(MODES)); axes[1].set_ylabel("LLC (setting 1)"); axes[1].set_title("LLC per model and seed"); axes[1].set_yscale("log")
plt.tight_layout(); plt.savefig("figures/seeds.png"); plt.show()
'''))
cells.append(md(open("nb_text/10_ablations.md").read()))
cells.append(code(r'''
# Ablation A: learned absolute position embeddings (zero-initialised) instead of no position embeddings.
CFG_PE = Config(**{**CFG.__dict__, "pos": "learned"})
models_pe = {}
for name, mode in [("PNTR", "pntr"), ("HIST", "hist")]:
    print(f"===== training {name} with learned position embeddings =====")
    models_pe[name], _ = train_model(mode, CFG_PE, device=DEVICE, verbose=False, **TRAIN)
ood_pe = {name: evaluate_lengths(m, OOD_LENGTHS, B=1024, device=DEVICE) for name, m in models_pe.items()}
ps_pe = {name: prefix_sensitivity(m, n=10, B=2048, device=DEVICE) for name, m in models_pe.items()}
fid_pe = {name: state_fidelity(m, n_range=TRAIN["n_range"], device=DEVICE) for name, m in models_pe.items()}
print(pd.DataFrame(ps_pe).T); print(pd.DataFrame(fid_pe).T)
'''))
cells.append(code(r'''
# Ablation B: no bottleneck -- the naive version of supervised-state training.  The sorting head reads the
# whole residual stream and the algorithm's state is only *encouraged* by a jointly trained linear probe.
CFG_NB = Config(**{**CFG.__dict__, "bottleneck": False})
models_nb = {}
for name, mode in [("PNTR", "pntr"), ("HIST", "hist")]:
    print(f"===== training {name} without the bottleneck (joint probe loss) =====")
    models_nb[name], _ = train_model_joint(mode, CFG_NB, steps=TRAIN["steps"], batch=TRAIN["batch"], lr=TRAIN["lr"],
                                           n_range=TRAIN["n_range"], seed=TRAIN["seed"], device=DEVICE, verbose=False)
ood_nb = {name: evaluate_lengths(m, OOD_LENGTHS, B=1024, device=DEVICE) for name, m in models_nb.items()}
ps_nb = {name: prefix_sensitivity(m, n=10, B=2048, device=DEVICE) for name, m in models_nb.items()}
probes_nb = {name: probe_report(m, which="final", device=DEVICE) for name, m in models_nb.items()}
print(pd.DataFrame(ps_nb).T); print(pd.DataFrame(probes_nb).T)
'''))
cells.append(code(r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for name, rs in ood.items():
    axes[0].plot([r["n"] for r in rs], [r["exact"] for r in rs], "o-", color={"PNTR": "C0", "HIST": "C1", "NONE": "C2"}[name], label=f"{name} (main run, no position embeddings)")
for name, rs in ood_pe.items():
    axes[0].plot([r["n"] for r in rs], [r["exact"] for r in rs], "s--", color={"PNTR": "C0", "HIST": "C1"}[name], label=f"{name} (learned position embeddings)")
for name, rs in ood_nb.items():
    axes[1].plot([r["n"] for r in rs], [r["exact"] for r in rs], "^:", color={"PNTR": "C0", "HIST": "C1"}[name], label=f"{name} (no bottleneck, joint probe loss)")
for name, rs in ood.items():
    axes[1].plot([r["n"] for r in rs], [r["exact"] for r in rs], "o-", color={"PNTR": "C0", "HIST": "C1", "NONE": "C2"}[name], alpha=0.35, label=f"{name} (main run)")
for ax, t in zip(axes, ["A: position embeddings", "B: bottleneck vs. no bottleneck"]):
    ax.axvspan(TRAIN["n_range"][0] - 0.5, TRAIN["n_range"][1] + 0.5, color="green", alpha=0.08)
    ax.set_xlabel("input length n"); ax.set_title(t); ax.legend(fontsize=7)
axes[0].set_ylabel("exact-match accuracy")
plt.tight_layout(); plt.savefig("figures/ablations.png"); plt.show()
rows = []
for tag, oo, pp in [("no PE (main)", ood, ps), ("learned PE", ood_pe, ps_pe), ("no bottleneck", ood_nb, ps_nb)]:
    for name in pp:
        r = {"variant": tag, "model": name, **{f"exact@n={n}": next(x["exact"] for x in oo[name] if x["n"] == n) for n in [10, 12, 16, 20, 24, 32]}, **pp[name]}
        rows.append(r)
df_abl = pd.DataFrame(rows).set_index(["variant", "model"]); df_abl.to_csv("results/ablations.csv")
df_abl
'''))
cells.append(md(open("nb_text/10_discussion.md").read()))
cells.append(code(r'''
summary = {"train": train_stats, "ood_length": {k: v for k, v in ood.items()}, "prefix_sensitivity": ps,
           "probes": {f"{k[0]}/{k[1]}": v for k, v in probe_rows.items()},
           "llc": {k: {kk: vv for kk, vv in v.items() if kk != "traces"} for k, v in llc.items()},
           "llc_setting2": {k: {kk: vv for kk, vv in v.items() if kk != "traces"} for k, v in llc2.items()},
           "seeds": seed_rows,
           "state_fidelity": fid, "ood_duplicates": dup_rows, "refined_llc": {f"{k[0]}/{k[1]}": v for k, v in rllc.items()}, "ablation_pe": {"ood": ood_pe, "prefix": ps_pe, "fidelity": fid_pe},
           "ablation_no_bottleneck": {"ood": ood_nb, "prefix": ps_nb, "probes": probes_nb},
           "config": CFG.__dict__, "train_hp": TRAIN, "llc_hp": LLC_HP, "llc_run": LLC_RUN, "device": DEVICE}
json.dump(summary, open("results/summary.json", "w"), indent=1, default=float)
for name, m in models.items():
    torch.save({"cfg": CFG.__dict__, "state": m.state_dict()}, f"results/model_{name}.pt")
print("saved results/summary.json and model checkpoints")
'''))
cells.append(md(open("nb_text/11_repro.md").read()))

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}}
nb.cells = cells
nbf.write(nb, "slt_sorting.ipynb")
print("wrote slt_sorting.ipynb with", len(cells), "cells")
