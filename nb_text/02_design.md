## 2. The toy: two sorting algorithms in one transformer

**Task.**  Sort a string of `n` digits.  The model sees `x_1 … x_n  |  y_1 … y_n` where `y = sorted(x)`,
and is trained (teacher-forced, cross-entropy) to predict `y_t` from everything before it.  Input
digits and output digits use *different* token ids, so the model can tell them apart by content
alone.  Training lengths are `n ∈ {4, …, 10}`; digits are i.i.d. uniform, so duplicates are common.
There is no train/test split in the usual sense — every batch is freshly sampled — so "training loss"
is population loss on the training distribution.

**Two algorithms that both solve the task.**

| | MIN — "track the current minimum" | HIST — "counting sort" |
|---|---|---|
| state at output step *t* | which digits are *still available* (10 bits) and the current value `v = y_{t-1}` | the histogram of absolute counts of the 10 digits, and the output index `t` |
| decision | emit the smallest available digit ≥ `v` | emit the digit `d` with `C(d-1) ≤ t < C(d)`, `C` = cumulative counts |
| what it depends on | the *previous outputs* (sequential) | the *position index* and *absolute counts* (parallel, ignores previous outputs) |
| scale | only comparisons ("is d still available?", "is d ≥ v?") — nothing depends on `n` | absolute magnitudes (counts, index) whose range is bounded by the training lengths |

Both are 100% correct on the training distribution, so the sorting loss cannot distinguish them.
They differ in *what else they do*: the min-tracker keeps working for longer inputs, the counting
sorter has never seen a count above ~5 or an index above 9 and has no reason to handle them.

**Architecture (identical for every model).**  A 2-layer, 4-head, d=128 GPT-style decoder with
zero-initialised learned absolute position embeddings (so that positions never seen in training
contribute exactly nothing rather than random noise).  The transformer body ends in an explicit
**11-dimensional state vector** `s = W·resid`, and the sorting decision is a small MLP read-out of
`s` alone.  This bottleneck is what makes the "supervised state" *binding*: the model can only
sort through those 11 numbers.  (Without it — see the ablation at the end — the auxiliary supervision
is satisfied but ignored: SGD represents the histogram and still sorts by tracking the minimum.)

**Supervised-state training.**  The three models differ *only* in an auxiliary MSE loss on `s`,
which is switched off for the last 20% of training so that the final weights are a minimum of the
pure sorting objective:

* `MIN`  : `s = [ availability bits (10) | v/10 ]`,
* `HIST` : `s = [ counts/10 (10) | t/10 ]`,
* `NONE` : no auxiliary loss — whatever SGD finds on its own.

**Measurements.**  (1) training loss / in-distribution accuracy; (2) free-running accuracy for lengths
`n = 4 … 32`; (3) accuracy at `n = 10` with many duplicates (small alphabets: counts far above the
training range while positions stay in range); (4) mechanistic tests — does the next prediction follow a
*corrupted* previous output (min-tracker) or ignore it (counting sort)?  which algorithm's state is
linearly decodable?  what do the attention heads attend to?; (5) the local learning coefficient of each
solution, estimated with SGLD on a fixed sample from the training distribution with identical sampler
settings for every model.
