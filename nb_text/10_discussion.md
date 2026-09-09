## 11. Discussion

**What the toy shows.**

1. *Underspecification is real even for a 400k-parameter sorter.*  Two networks with the same
   architecture, the same data and a training loss that differs only in the noise reach the same
   in-distribution behaviour (Section 5) and then diverge sharply the moment the inputs get longer
   (Section 6).  No amount of evaluation on the training distribution would have distinguished them —
   exactly the situation that *You Are What You Eat* calls the underspecification problem and that
   Hoogland's essay calls the generalisation problem.
2. *The difference is algorithmic, not cosmetic.*  The min-tracker follows a corrupted prefix and never
   ignores it; the counting sorter largely ignores the prefix or is derailed by it; each model's
   bottleneck holds its own algorithm's state and not the other's; and the attention heads of the two
   models attend to different things (Section 8).
3. *Which algorithm fails, and why, is predictable from what the state has to represent.*  The
   min-tracker's state consists of comparisons ("is digit d still available?"), which are the same
   computation at any length.  The counting sorter's state consists of absolute magnitudes — counts and
   an output index — whose range is bounded by the training lengths, so any implementation has to
   *extrapolate* them.  Positional embeddings only change where the cliff is (ablation A): with no
   positional embeddings the counter breaks a couple of positions past the training range, with
   learned positional embeddings it breaks immediately.
4. *Supervising a state does not, by itself, change the algorithm.*  Without the bottleneck (ablation B)
   the auxiliary probe is satisfied — the histogram becomes linearly decodable with R² ≈ 0.98 — while
   the model goes on sorting by tracking the minimum.  Representation is not use.  This is worth
   remembering when linear probes are used as evidence about mechanisms.
5. *SGD's default.*  The unsupervised `NONE` model behaves like the min-tracker in every test.  For this
   architecture and data, min-tracking is the algorithm that is found without help; counting sort is the
   one that had to be scaffolded in.

**What the LLC says.**  See the numbers in Sections 9–10.  Two caveats that apply to any LLC comparison:
the absolute value depends on the sampler settings (nβ, γ, ε), so only comparisons at identical settings
are meaningful, and these minima are very sharp (a loss of 1e-5 per token), which forces small nβ and
strong localisation — the estimate is a *scale-dependent* effective dimension rather than the asymptotic
learning coefficient.  Within those caveats the comparison is a direct test of the SLT claim that the
geometry of the loss landscape around a solution reflects the algorithm it implements: if it does, two
solutions with the same loss but different algorithms should have different LLCs, and the weight-refined
LLC should show the complexity sitting in different components.

**Relation to the safety argument.**  The alignment version of this story replaces "sort longer strings"
by "behave well off the training distribution", and the question of which solution SGD finds by
"which solution the posterior prefers".  Eq. (3) of *You Are What You Eat* says that, at equal loss, the
posterior odds are decided by Δλ: the more degenerate solution wins.  Whether the generalising solution
is the more degenerate one is an empirical question — here it can be read off the LLC table — and a
scalable way of measuring λ (or refined versions of it) for candidate solutions is precisely the kind of
tool the SLT-for-safety agenda is after.

**Limitations and next steps.**  (i) The scaffold is strong: an explicit bottleneck plus a teacher-forced
read-out.  A weaker elicitation (e.g. attention supervision, or data engineering alone) would be closer to
how real training shapes algorithms.  (ii) Only one seed per model; the OOD curves of the min-tracker
vary by ±10–20 percentage points at the longest lengths between seeds in our pilots, though the
qualitative gap to the counting sorter never closed.  (iii) The LLC estimates are at one sampler setting
chosen by the usual diagnostics; a proper study would show the comparison is stable across settings and
would add data-refined LLCs (e.g. restricted to long sequences) and susceptibilities.  (iv) A natural
extension is to watch the *development* of the two solutions (LLC over training time) and to check
whether the transition from the histogram-computing body to the counting-sort read-out shows up as a
phase transition, as in the developmental-interpretability programme.
