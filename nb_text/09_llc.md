## 9. Geometry: the local learning coefficient

We estimate the LLC of each trained model on a fixed i.i.d. sample of `N` sequences from the training
distribution, with the per-sequence negative log-likelihood as `L_n`, using SGLD as in Lau et al. (2023)
and the diagnostics of the `devinterp` sampling guide (the loss trace must rise above `L_n(w*)`, level off,
agree across chains, and show no spikes or NaNs).

Two practical points, both consequences of how *sharp* these minima are (a loss of ~1e-5 per token means
the logit margins are enormous, and a random perturbation of 0.003 per weight already costs several nats):

* The theoretical inverse temperature `nβ* = N / log N` (≈ 1700 here) makes SGLD diverge for every step
  size we tried, so `nβ` is treated as a sampler knob and swept over `{10, 100}` as the sampling guide
  recommends.  The estimate then scales with `nβ`; only comparisons at *identical* settings are meaningful.
* A strong localisation (`γ = 10⁴`, i.e. a Gaussian ball of radius ~0.01 per weight) and a small step size
  are needed to keep the chains in the region where the loss is still small.  With these settings the
  estimate is a *scale-dependent* count of stiff directions,  λ̂ ≈ ½ Σᵢ nβhᵢ / (nβhᵢ + γ)  in the quadratic
  picture, rather than the asymptotic learning coefficient — but it is the same functional of the local
  geometry for every model, which is what the comparison needs.

The cells below run a small sweep on the `MIN` model, then use one setting for all models, then an
`nβ = 0` control (no loss-gradient term: the chains only feel the noise and the localisation; if the
estimate does not collapse, the "signal" was never coming from the loss landscape), and finally the
*weight-refined* LLC of Wang et al. (2024) — the same estimate with only one group of parameters free —
to see where each solution's stiffness sits.
