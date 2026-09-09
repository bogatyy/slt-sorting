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

_Results summary: see the bottom of this file (filled in from the executed notebook)._

## Files

* `slt_sorting.ipynb` — the executed notebook (literature background, design, code, results, discussion).
* `slt_sorting_lib.py` — the same code as an importable module (data, model, staged training,
  evaluation, mechanistic tests, SGLD-based LLC estimator).
* `build_notebook.py`, `nb_text/` — the notebook is assembled from the library and these markdown cells.
* `run_on_hf_jobs.py` — runs the notebook on a rented GPU via Hugging Face Jobs and brings back the
  executed notebook, `figures/` and `results/`.
* `figures/`, `results/` — outputs of the committed run (plots, CSV tables, `summary.json`, checkpoints).

## Reproduce

```bash
pip install torch numpy matplotlib pandas jupyter papermill
# quick CPU sanity run (~5-10 min, fewer steps / SGLD chains):
SLT_FAST=1 papermill slt_sorting.ipynb out.ipynb
# full run on a rented GPU (needs HF_TOKEN of a PRO account):
python run_on_hf_jobs.py launch --notebook slt_sorting.ipynb --files slt_sorting_lib.py --flavor t4-small
python run_on_hf_jobs.py status <job_id>      # poll
python run_on_hf_jobs.py fetch <run_id>       # download outputs/<run_id>/
```
