# Same loss, different algorithm: two sorting transformers and the geometry that separates them

*A self-contained toy experiment on the "underspecification" argument from singular learning theory.
Everything runs in one notebook; the committed outputs were produced on a rented Hugging Face T4 GPU
(about 30 minutes).*

**Summary.**  One tiny transformer architecture, one training objective (sort strings of 4–10 digits),
two weight vectors that both reach 100% accuracy at a loss of ~1e-7 nats per token or less.  One
implements a *pointer-tracking* algorithm (keep the last emitted value, emit the smallest input digit that is
still available), the other a *counting sort* (compute cumulative counts once, emit digit `d` at output
indices `C(d-1) … C(d)-1`).  They were produced by supervising an explicit 11-number state bottleneck in
the same architecture.  Nothing about the training distribution distinguishes them; on longer inputs the
pointer-tracker keeps sorting (100% at n = 16, 87% at n = 24) while the counting sorter collapses within a few
positions of the training range (3% at n = 16).  Three independent tests — corrupted-prefix behaviour,
what each bottleneck holds, attention patterns — confirm that the two networks compute different things,
and a third model trained with no state supervision turns out to be a pointer-tracker: SGD's default.
Finally we estimate the local learning coefficient (LLC) of each solution with SGLD, for three seeds per
model and two sampler settings.  SGD's own solution is 20–70× more degenerate than either scaffolded one.
Between the two equal-loss solutions, geometry separates them by a factor of two at the setting that
probes the sharpest directions (consistently across seeds; the weight-refined LLC puts the pointer-tracker's
stiffness in its attention look-ups and the counting sorter's in its read-out thresholds) — with the
brittle counting sorter as the *more* degenerate one — and does not separate them at a colder setting.
