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

| | `PNTR` — a pointer to the current minimum | `HIST` — counting sort |
|---|---|---|
| state | which digits are still available (10 bits) + the current value `v` | cumulative counts `C(d)` (10 numbers) + the output index `t` |
| decision | smallest available digit ≥ `v` | the digit `d` with `C(d-1) ≤ t < C(d)` |
| depends on | the previous outputs (comparisons only) | absolute counts and the absolute position |

`NONE` is a third model trained with no state supervision at all (what SGD finds on its own).

## Results (committed run `full4`: Hugging Face Jobs, one T4, 37 min of cell time; `results/summary.json`, figures in `figures/`)

**Same training loss, 100% in-distribution accuracy** (population loss on fresh samples, teacher forced):

| model | CE / token | NLL / sequence | exact-match (teacher forced) |
|---|---|---|---|
| PNTR | 1.3e-09 | 5.4e-09 | 1.000 |
| HIST | 3.2e-07 | 2.3e-06 | 1.000 |
| NONE | 1.9e-10 | 1.6e-09 | 1.000 |

**Different algorithms, visible only out of distribution** — free-running exact-match accuracy by input
length (training lengths 4–10; `figures/ood_length.png`):

| model | n=10 | n=12 | n=13 | n=14 | n=16 | n=18 | n=20 | n=24 | n=28 | n=32 |
|---|---|---|---|---|---|---|---|---|---|---|
| PNTR | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.98 | 0.87 | 0.59 | 0.27 |
| HIST | 1.00 | 1.00 | 0.95 | 0.70 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| NONE | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.83 | 0.31 | 0.06 | 0.00 |

**Mechanistic evidence that they differ** (`figures/attention.png`):

| test | PNTR | HIST | NONE |
|---|---|---|---|
| corrupted prefix: follows it (pointer-tracker) / ignores it (counting sort) | 0.99 / 0.00 | 0.01 / 0.83 | 0.72 / 0.21 |
| bottleneck holds availability bits (bit acc) / cumulative counts (count acc) | 1.00 / 0.10 | 0.40 / 0.59 | 0.45 / 0.07 |
| attention | sharp content look-ups of particular digits | uniform over inputs + reads the separator | like PNTR |

**Local learning coefficient** (SGLD with identical settings for all models; setting 1: nβ = 10, γ = 10⁴,
ε = 10⁻⁵, 6 chains × 2000 steps; setting 2: nβ = 100, γ = 10⁴, ε = 10⁻⁶, 4 chains × 2500 steps;
`figures/llc.png`, `figures/refined_llc.png`, `figures/seeds.png`):

| | PNTR | HIST | NONE |
|---|---|---|---|
| LLC, setting 1 (± chain std) | 1.18 ± 0.10 | 0.62 ± 0.06 | 0.015 ± 0.004 |
| LLC, setting 1, two more seeds | 1.22 ± 0.04, 0.91 ± 0.06 | 0.62 ± 0.07, 0.61 ± 0.02 | 0.037 ± 0.005, 0.046 ± 0.026 |
| LLC, setting 2 | 1.11 ± 0.03 | 1.18 ± 0.16 | 0.051 ± 0.017 |
| weight-refined LLC (setting 1): attention / MLPs / embeddings / state+read-out | 0.84 / 0.00 / 0.00 / 0.18 | 0.03 / 0.00 / 0.00 / 0.16 | 0.02 / 0.00 / 0.00 / 0.00 |

Across seeds, the two extra seeds of each model reproduce the length-generalisation and prefix results
(PNTR: 88–98% exact at n = 20, follows the prefix 97–100%; HIST: 0% at n = 20, ignores the prefix 77–88%;
NONE: 83–90% at n = 20, follows 72–74%).

So: the solution SGD finds on its own (a pointer-tracker) is 20–70× more degenerate than either scaffolded
solution, at both settings and all seeds.  Between the two equal-loss solutions, geometry separates them
by a factor of two at the setting that probes only the sharpest directions — consistently across seeds,
and the refined LLC says *where*: the pointer-tracker's stiffness is in its attention look-ups, the counting
sorter's in its read-out thresholds — but not at the colder setting, where they are indistinguishable.
Where it does separate them, the brittle counting sorter is the *more* degenerate one: "lower LLC" is
not by itself a generalisation guarantee, and the ranking of two implementations can depend on the
scale at which the geometry is probed.

**Extended lengths** (added after the GPU run; `extended_lengths.py`, run on CPU from the committed
checkpoints; `figures/ood_length_extended.png`, `results/ood_lengths_extended*.csv`):

| | n=20 | n=32 | n=64 | n=128 |
|---|---|---|---|---|
| PNTR, exact-match | 1.00 | 0.30 | 0.00 | 0.00 |
| PNTR, per-token (teacher forced) | 1.00 | 0.97 | 0.86 | 0.83 |
| HIST, per-token (teacher forced) | 0.71 | 0.58 | 0.50 | 0.47 |
| NONE, per-token (teacher forced) | 0.99 | 0.91 | 0.83 | 0.80 |

The pointer-tracker generalises far better than the counting sorter but not to arbitrary length.  All of its
errors on long inputs occur at steps where the correct digit *repeats* the previous one (the model must
decide "is another copy left?"), none where the sorted output advances: the availability comparison is
done by softmax attention, whose precision shrinks as the counts grow with n.

**Ablations** (`figures/ablations.png`): with learned position embeddings the counting sorter fails
immediately past the training range (4% at n = 11) while the pointer-tracker still generalises (96% at
n = 20); without the state bottleneck, "supervising" the histogram with a jointly trained probe leaves the
model a pointer-tracker (the histogram becomes linearly decodable, R² ≈ 0.94, but is not used).

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
