# Same loss, different algorithm: digit-sorting transformers as a toy for SLT / underspecification

A self-contained toy experiment for the argument in
[Singular Learning Theory for AI Safety](https://www.jessehoogland.com/article/lesswrong/2025-07-01-slt-for-ai-safety),
the [SLT wiki](https://www.lesswrong.com/w/singular-learning-theory) and
[You Are What You Eat](https://arxiv.org/abs/2502.05475): two networks with the **same architecture
and the same training loss** can implement **different algorithms** that only come apart out of
distribution — and the geometry of the loss landscape (the local learning coefficient) is one
handle on telling them apart *before* the distribution shift.

**Task.** Sort strings of `n ∈ {4..10}` digits with a 2-layer GPT-style decoder.

**Two algorithms, elicited by supervising an explicit 11-number "state" in the same architecture:**

| | `MIN` — track the current minimum | `HIST` — counting sort |
|---|---|---|
| state | which digits are still available (10 bits) + the current value `v` | cumulative counts `C(d)` (10 numbers) + the output index `t` |
| decision | smallest available digit ≥ `v` | the digit `d` with `C(d-1) ≤ t < C(d)` |
| depends on | the previous outputs (comparisons only) | absolute counts and the absolute position |

`NONE` is a third model trained with no state supervision at all (what SGD finds on its own).

## Results (from the committed run: `results/summary.json`, figures in `figures/`)

**Same training loss, 100% in-distribution accuracy** (population loss on fresh samples, teacher forced):

| model | CE / token | NLL / sequence | exact-match (teacher forced) |
|---|---|---|---|
| MIN | 1.3e-09 | 5.4e-09 | 1.000 |
| HIST | 3.2e-07 | 2.3e-06 | 1.000 |
| NONE | 1.9e-10 | 1.6e-09 | 1.000 |

**Different algorithms, visible only out of distribution** — free-running exact-match accuracy by input
length (training lengths 4–10; `figures/ood_length.png`):

| model | n=10 | n=12 | n=13 | n=14 | n=16 | n=18 | n=20 | n=24 | n=28 | n=32 |
|---|---|---|---|---|---|---|---|---|---|---|
| MIN | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.98 | 0.87 | 0.59 | 0.27 |
| HIST | 1.00 | 1.00 | 0.95 | 0.70 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| NONE | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.83 | 0.31 | 0.06 | 0.00 |

**Mechanistic evidence that they differ** (`figures/attention.png`):

| test | MIN | HIST | NONE |
|---|---|---|---|
| corrupted prefix: follows it (min-tracker) / ignores it (counting sort) | 0.99 / 0.00 | 0.01 / 0.83 | 0.72 / 0.21 |
| bottleneck holds availability bits (bit acc) / cumulative counts (count acc) | 1.00 / 0.10 | 0.40 / 0.59 | 0.45 / 0.07 |
| attention | sharp content look-ups of particular digits | uniform over inputs + reads the separator | like MIN |

**Local learning coefficient** (SGLD, nβ = 10, γ = 10⁴, ε = 10⁻⁵, 6 chains × 2000 steps, identical for
all models; `figures/llc.png`, `figures/refined_llc.png`):

| | MIN | HIST | NONE |
|---|---|---|---|
| LLC (± chain std) | 1.18 ± 0.10 | 0.62 ± 0.06 | 0.015 ± 0.004 |
| weight-refined LLC: attention / MLPs / embeddings / state+read-out | 0.79 / 0.00 / 0.00 / 0.18 | 0.03 / 0.00 / 0.00 / 0.16 | 0.01 / 0.00 / 0.00 / 0.00 |

So: geometry does separate the two equal-loss solutions (factor 2, far outside chain noise), and says
*where* they differ (the min-tracker's stiffness is in its attention look-ups, the counting sorter's in
its read-out thresholds).  But the brittle counting sorter is the *more* degenerate of the two, and the
solution SGD finds on its own (a min-tracker) is ~50× more degenerate than either — "lower LLC" is not
by itself a generalisation guarantee.  A second sampler setting and three seeds per model (notebook
Sections 9–9b) check that these orderings are not sampler or seed artefacts.

**Ablations** (`figures/ablations.png`): with learned position embeddings the counting sorter fails
immediately past the training range (4% at n = 11) while the min-tracker still generalises (96% at
n = 20); without the state bottleneck, "supervising" the histogram with a jointly trained probe leaves the
model a min-tracker (the histogram becomes linearly decodable, R² ≈ 0.94, but is not used).

## Files

* `slt_sorting.ipynb` — the executed notebook (literature background, design, code, results, discussion).
* `slt_sorting_lib.py` — the same code as an importable module (data, model, two-phase supervised-state
  training, evaluation, mechanistic tests, SGLD-based LLC estimator with weight restriction).
* `build_notebook.py`, `nb_text/` — the notebook is assembled from the library and these markdown cells.
* `run_on_hf_jobs.py` — runs the notebook on a rented GPU via Hugging Face Jobs and brings back the
  executed notebook, `figures/` and `results/`; `finalize_run.py` copies a run into the repo;
  `summarize_run.py` prints the tables above from `results/summary.json`.
* `dev_run.py`, `dev_llc.py` — CPU pilot drivers used while developing (train one model / sweep SGLD settings).
* `figures/`, `results/` — outputs of the committed run (plots, CSV tables, `summary.json`, model checkpoints).

## Reproduce

```bash
pip install torch numpy matplotlib pandas jupyter papermill
# quick CPU sanity run (~10 min, fewer steps / SGLD chains):
SLT_FAST=1 papermill slt_sorting.ipynb out.ipynb
# full run on a rented GPU (needs HF_TOKEN of a PRO account; ~30 min on a T4):
python run_on_hf_jobs.py launch --notebook slt_sorting.ipynb --flavor t4-small
python run_on_hf_jobs.py status <job_id>      # poll
python run_on_hf_jobs.py fetch <run_id>       # download outputs/<run_id>/
python finalize_run.py hf_runs/outputs/<run_id>
```
