## 11. Discussion

**What the toy shows.**

1. *Underspecification is real even for a 400k-parameter sorter.*  `MIN` and `HIST` have the same
   architecture, saw the same data stream, and end at a sorting loss that is zero to within noise
   (about 1e-9 and 3e-7 nats per token) with 100% exact-match accuracy on the training distribution
   (Section 5).  The moment the inputs get longer they diverge: `MIN` still sorts every 16-digit string and
   87% of 24-digit strings, `HIST` is at 3% by n = 16 and 0% from n = 18 on (Section 6).  No evaluation
   on the training distribution would have told them apart — the situation *You Are What You Eat*
   calls the underspecification problem and Hoogland's essay the generalisation problem.
2. *The difference is algorithmic.*  With a plausible-but-wrong prefix, `MIN` follows the prefix 99%
   of the time and never ignores it; `HIST` follows it 1% of the time and ignores it 83% of the time
   (Section 8).  Each bottleneck holds its own algorithm's state and not the other's (Section 4).  The
   attention maps show it directly: `MIN`'s heads do sharp content look-ups of particular input digits,
   `HIST`'s layer-0 heads attend uniformly over the inputs (that *is* the histogram) and its layer-1 heads
   read the separator (under no position embeddings, the fraction of attention that lands on the
   separator is how the model knows n and t).
3. *Which algorithm fails, and why, follows from what its state has to represent.*  The min-tracker's
   state is made of comparisons ("is digit d still available?"), the same computation at any length.  The
   counting sorter's state is made of absolute magnitudes — counts and an output index — whose training
   range is bounded, so every implementation has to extrapolate them.  Position embeddings only move the
   cliff (ablation A): with none, the counter fails a few positions past the training range (it has to
   extrapolate a fraction→count map); with learned ones it fails immediately (4% at n = 11).  The
   duplicates control (Section 7) makes the same point from the other side: at n = 10 the counter is fine
   until the histogram itself becomes unlike anything seen in training (all ten digits equal: 0% exact).
4. *Supervising a state does not by itself change the algorithm.*  Without the bottleneck (ablation B)
   the auxiliary probe is satisfied — the histogram is linearly decodable from the `HIST`-probed model
   with R² ≈ 0.94 — while the model goes on sorting by tracking the minimum (59% follows-prefix, full
   length generalisation).  Representation is not use; a probe finding a variable is not evidence that the
   variable is on the computational path.
5. *SGD's default is the min-tracker.*  The unsupervised `NONE` model behaves like `MIN` in every test
   (72% follows-prefix, generalises to n = 20 at 83%), and its attention maps look like `MIN`'s.  Counting
   sort is the algorithm that had to be scaffolded in.

**What the LLC says.**  With one sampler setting for all models (nβ = 10, γ = 10⁴, ε = 10⁻⁵; chains agree
to within ~10%, the nβ = 0 control collapses to ~0, and a second setting gives the same ordering):

* `NONE` is *far* more degenerate than either scaffolded solution — λ̂ ≈ 0.015 against ≈ 1.2 (`MIN`) and
  ≈ 0.6 (`HIST`).  Whatever SGD finds on its own sits in a much flatter region than what supervised-state
  training produces, even when the algorithm is the same as `MIN`'s.  (Part of this is procedure: `NONE`
  spent all of training being annealed by SGD on the sorting loss, `MIN`/`HIST` only the last 20% at a
  small learning rate; SGD's implicit drift towards flat regions is itself one of the phenomena the LLC
  was introduced to measure — Lau et al.'s Figure 1.)
* Between the two solutions with equal loss the LLC *does* differ, by a factor of two, well outside the
  chain-to-chain spread — so geometry separates two solutions that the loss cannot — and the
  weight-refined LLC says where: `MIN`'s stiffness is in its **attention** weights (rLLC ≈ 0.8 of the total
  1.2 — sharp content look-ups are fragile), `HIST`'s attention is almost free (≈ 0.03 — uniform averaging
  is robust to weight perturbations) and its stiffness sits in the **state/read-out** thresholds.
* But the *direction* of the difference is the opposite of "simpler generalises better": at this scale the
  brittle counting sorter is the more degenerate of the two scaffolded solutions.  Read through Eq. (3)
  of *You Are What You Eat*, nΔℓ ≈ 16380 × 2.3·10⁻⁶ ≈ 0.04 nats is negligible next to Δλ·log n ≈ 0.6 × 9.7 ≈ 5
  nats, so a Bayesian posterior at this temperature would prefer `HIST` to `MIN` by roughly e⁵ — and `NONE`
  to both by far more.  In our pilot checkpoints after a quarter of the training the ordering of `MIN` and
  `HIST` was reversed, and Section 9b checks it across seeds; the honest summary is that the LLC ranks
  *implementations*, and the ranking of the two scaffolded implementations is not something one should
  read a generalisation guarantee from.

**Relation to the safety argument.**  The alignment version of this story replaces "sort longer strings"
by "behave well off the training distribution" and "which solution SGD finds" by "which solution the
posterior prefers".  The toy reproduces the premise (equal loss, different algorithms, divergence under
shift) and shows the geometric tool doing what it is supposed to do (distinguishing the solutions in
distribution, and localising where they differ), while also showing the caveat that matters for safety:
the lower-λ solution is not automatically the one that generalises the way you want.  The "S4"
programme — learn which *data* produces which *geometry* produces which *algorithm* — is exactly the map
one would need to predict, before the shift, which of these two a given training set will yield.

**Limitations and next steps.**  (i) The scaffold is strong (an explicit bottleneck plus a teacher-forced
read-out); weaker elicitation — attention supervision, or data engineering alone — would be closer to
how real training shapes algorithms.  (ii) Three seeds per model (Section 9b) is enough to see whether
the orderings are stable, not to estimate their distribution.  (iii) The LLC is estimated at one
localisation scale chosen by the standard diagnostics; because the minima are extremely sharp, the
estimate is a scale-dependent count of stiff directions rather than the asymptotic learning coefficient.
Data-refined LLCs (e.g. restricted to long sequences) and susceptibilities would be the natural
next tools.  (iv) The natural developmental question — does the counting sorter's body→read-out hand-off
show up as a phase transition in λ̂(t) over training? — is a small extension of this notebook.
