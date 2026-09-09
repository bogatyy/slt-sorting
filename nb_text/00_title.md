# Same loss, different algorithm: two sorting transformers and the geometry that separates them

*A self-contained toy experiment on the "underspecification" argument from singular learning theory.
Everything runs in one notebook; the committed outputs were produced on a rented Hugging Face T4 GPU.*

**Summary.**  One tiny transformer architecture, one training objective (sort strings of 4–10 digits),
two weight vectors that both reach ~100% accuracy and a sorting loss of about 1e-5 per token.  One
implements a *min-tracking* algorithm (keep the last emitted value, emit the smallest input digit that is
still available), the other a *counting sort* (compute the histogram of absolute counts, emit digit `d`
at output indices `C(d-1) … C(d)-1`).  They were produced by supervising an explicit 11-number "state"
in the same architecture.  Nothing about the training distribution distinguishes them; on longer
inputs the min-tracker keeps working while the counting sorter collapses within a few positions of the
training range.  Three independent tests (corrupted-prefix behaviour, linear decodability of each
algorithm's state, attention patterns) confirm that the two networks really do compute different things,
and a third model trained with no state supervision at all turns out to be a min-tracker — SGD's default.
Finally we estimate the local learning coefficient (LLC) of each solution with SGLD and ask whether
this purely in-distribution, geometric quantity separates them.
