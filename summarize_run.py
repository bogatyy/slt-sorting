"""Print a markdown summary of an executed run: python summarize_run.py <dir with results/ and the .ipynb>"""
import json, sys, os, nbformat
d = sys.argv[1]
s = json.load(open(os.path.join(d, "results", "summary.json")))
nbs = [f for f in os.listdir(d) if f.endswith(".ipynb")]
if nbs:
    nb = nbformat.read(os.path.join(d, nbs[0]), as_version=4)
    errs = [(i, o["ename"], o["evalue"]) for i, c in enumerate(nb.cells) if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    tot = sum(c.get("metadata", {}).get("papermill", {}).get("duration") or 0 for c in nb.cells)
    print(f"notebook: {nbs[0]}  errors: {errs}  total cell time: {tot/60:.1f} min  device: {s.get('device')}")
names = list(s["train"])
print("\n| model | CE / token | NLL / sequence | exact (teacher-forced) |\n|---|---|---|---|")
for n in names:
    t = s["train"][n]; print(f"| {n} | {t['ce_per_token']:.2e} | {t['nll_per_sequence']:.2e} | {t['seq_acc_teacher_forced']:.4f} |")
ns = [r["n"] for r in s["ood_length"][names[0]]]
print("\n| model | " + " | ".join(f"n={n}" for n in ns) + " |\n|---|" + "---|" * len(ns))
for n in names:
    print(f"| {n} | " + " | ".join(f"{r['exact']:.2f}" for r in s["ood_length"][n]) + " |")
print("\nprefix sensitivity:")
for n in names:
    p = s["prefix_sensitivity"][n]; print(f"  {n}: follows {p['follows_prefix(PNTR-like)']:.2f}  ignores {p['ignores_prefix(HIST-like)']:.2f}  other {p['other']:.2f}")
print("\nstate fidelity:")
for n in names:
    f = s["state_fidelity"][n]; print(f"  {n}: mse vs PNTR {f['mse_vs_MIN_state']:.3f} (bits {f['avail_bit_acc']:.3f}) | mse vs HIST {f['mse_vs_HIST_state']:.3f} (counts {f['count_acc']:.3f})")
print("\nprobes (post hoc):")
for k, v in s["probes"].items():
    print(f"  {k}: hist R2 {v['hist_R2']:.3f} idx R2 {v['idx_R2']:.3f} avail bits {v['avail_bit_acc']:.3f}")
print("\nLLC:", {n: (round(v["llc"], 3), round(v["llc_std"], 3)) for n, v in s["llc"].items()}, "hp:", s["llc_hp"], s["llc_run"])
if "refined_llc" in s:
    print("refined LLC:")
    for k, v in s["refined_llc"].items():
        print(f"  {k}: {v['rLLC']:.3f} +- {v['chain std']:.3f} ({v['#params']} params)")
print("\nablation PE:")
for n, rs in s["ablation_pe"]["ood"].items():
    print(f"  {n}: " + " ".join(f"{r['n']}:{r['exact']:.2f}" for r in rs), "| prefix", {k: round(v, 2) for k, v in s["ablation_pe"]["prefix"][n].items() if k != "n_cases"})
print("ablation no-bottleneck:")
for n, rs in s["ablation_no_bottleneck"]["ood"].items():
    print(f"  {n}: " + " ".join(f"{r['n']}:{r['exact']:.2f}" for r in rs), "| prefix", {k: round(v, 2) for k, v in s["ablation_no_bottleneck"]["prefix"][n].items() if k != "n_cases"}, "| probes", {k: round(v, 3) for k, v in s["ablation_no_bottleneck"]["probes"][n].items()})
print("\nduplicates (n=10):")
for r in s["ood_duplicates"]:
    print(f"  {r['model']} {r['alphabet']}: exact {r['exact']:.3f} token {r['token']:.3f}")
