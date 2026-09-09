# Same loss, different algorithm: two sorting transformers and the geometry that separates them

*A self-contained toy experiment on the "underspecification" argument from singular learning theory.
Everything runs in one notebook; the committed outputs were produced on a rented Hugging Face T4 GPU
(about 30 minutes).*

**Summary.**  One tiny transformer architecture, one training objective (sort strings of 4–10 digits),
two weight vectors that both reach 100% accuracy at a loss of ~1e-7 nats per token or less.  One
implements a *min-tracking* algorithm (keep the last emitted value, emit the smallest input digit that is
still available), the other a *counting sort* (compute cumulative counts once, emit digit `d` at output
indices `C(d-1) … C(d)-1`).  They were produced by supervising an explicit 11-number state bottleneck in
the same architecture.  Nothing about the training distribution distinguishes them; on longer inputs the
min-tracker keeps sorting (100% at n = 16, 87% at n = 24) while the counting sorter collapses within a few
positions of the training range (3% at n = 16).  Three independent tests — corrupted-prefix behaviour,
what each bottleneck holds, attention patterns — confirm that the two networks compute different things,
and a third model trained with no state supervision turns out to be a min-tracker: SGD's default.
Finally we estimate the local learning coefficient (LLC) of each solution with SGLD.  Geometry does
separate the two equal-loss solutions (by a factor of two, and the weight-refined LLC says the
min-tracker's stiffness lives in its attention while the counting sorter's lives in its read-out) — but the
brittle counting sorter is the *more* degenerate of the two, and SGD's own solution is far more
degenerate than either.
