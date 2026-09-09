## 10. Ablations

**A. Position embeddings.**  The main run uses *no* position embeddings, so the only thing that
changes out of distribution is the number of tokens.  Here the same two models are trained with learned
absolute position embeddings (zero-initialised, so unseen positions contribute nothing).

**B. Is the bottleneck necessary?**  The naive way to "supervise a state" is to add a linear probe on the
residual stream and train it jointly with the sorting loss.  Without the bottleneck the read-out can use
the whole residual stream, and the probe only has to be *satisfiable*, not *used*.
