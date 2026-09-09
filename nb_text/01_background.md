## 1. Background: why "same loss, different algorithm" matters for safety

This notebook is a toy instantiation of an argument that runs through three pieces of recent
writing on **singular learning theory (SLT)** and alignment:

* Jesse Hoogland, *[Singular Learning Theory for AI Safety](https://www.jessehoogland.com/article/lesswrong/2025-07-01-slt-for-ai-safety)* (2025).
  Alignment today is mostly *data engineering*: the same optimiser and architecture are pointed at a
  curated distribution and we hope the right algorithm falls out.  Two things can go wrong.  The
  **learning problem**: training can converge to a solution that matches the training specification
  but is not the one we intended.  The **generalisation problem**: the intended and unintended
  solutions agree on the training distribution and only come apart under distribution shift, when
  "capabilities generalise further than alignment".  SLT's proposal is that the *geometry of the loss
  landscape around a solution reflects the algorithm it implements* (the "S4" chain: statistical
  structure in data → geometric structure of the loss → developmental structure → algorithmic
  structure in the weights), so that measuring geometry can tell solutions apart even when their
  loss cannot.
* The LessWrong [Singular Learning Theory wiki page](https://www.lesswrong.com/w/singular-learning-theory)
  and the "Distilling SLT" sequence: neural networks are *singular* models — many weight
  configurations implement the same function, and the set of low-loss weights is not a nice
  manifold but a variety with degenerate directions.  Watanabe's theory replaces the parameter count
  by the **learning coefficient** λ (the real log canonical threshold): the Bayesian free energy of a
  region around a minimum w* behaves as  F_n ≈ n L_n(w*) + λ(w*) log n,  so among solutions with equal
  loss the posterior overwhelmingly prefers the *more degenerate* one (smaller λ = "simpler").
* Lehalleur, Hoogland, Farrugia-Roberts, Wei, Gietelink Oldenziel, Wang, Carroll & Murfet,
  *[You Are What You Eat — AI alignment requires understanding how data shapes structure and generalisation](https://arxiv.org/abs/2502.05475)* (2025).
  Its Section 3 is the **underspecification problem**: many parameter configurations reach the same
  training loss but generalise differently outside the training distribution, so evaluations on the
  training distribution cannot certify safety.  Their Eq. (3) writes the posterior log-odds between two
  such solutions 𝒰 and 𝒱 as

  $$\log \frac{p_n(\mathcal U)}{p_n(\mathcal V)} = n\,\Delta\ell_n + \Delta\lambda\,\log n + O_p(\log\log n),$$

  where Δℓ is the difference in training loss and Δλ the difference in **local learning coefficient
  (LLC)**.  When Δℓ ≈ 0 the geometry term decides, and the paper's worry is that the simpler (lower-λ)
  solution may be the *misaligned* shortcut.

The LLC of a trained network can be estimated (Lau, Furman, Wang, Murfet & Wei, 2023,
*[The Local Learning Coefficient: a singularity-aware complexity measure](https://arxiv.org/abs/2308.12108)*)
by sampling the tempered, localised posterior with stochastic-gradient Langevin dynamics:

$$\hat\lambda(w^*) = n\beta^*\Big(\mathbb E_{w\,\sim\,p(w\mid w^*,\beta^*,\gamma)}[L_n(w)] - L_n(w^*)\Big),\qquad
\Delta w = \tfrac{\epsilon}{2}\Big(-n\beta^*\nabla L_{\text{batch}}(w) + \gamma\,(w^*-w)\Big) + \mathcal N(0,\epsilon),$$

with β* = 1/log n, localisation γ and step size ε (this is what the `devinterp` library implements;
we re-implement the ~30 lines below so that the notebook has no dependencies beyond PyTorch).

**What the toy is meant to show.**  Take one architecture and one training objective, and produce two
weight vectors with the *same* training loss that implement *different algorithms*.  Then check that
(i) the difference is invisible in-distribution, (ii) it becomes visible under a distribution shift
(longer inputs) — the "alignment/misalignment boundary" — and (iii) whether the LLC, a purely
geometric quantity computed on the *training* distribution, can tell the two apart.
