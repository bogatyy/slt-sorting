## 8. Are they really different algorithms?

**Prefix corruption.**  Teacher-force a prefix in which the *last emitted digit* `y_{t-1}` has been
replaced by a different digit `v'` — chosen so that the prefix is still sorted and `v'` occurs in the
input, i.e. the prefix looks perfectly plausible but is inconsistent with the input multiset — and look at
the model's next prediction.  A pointer-tracker follows the corrupted value (it emits the smallest
still-available digit ≥ `v'`); a counting sorter never looks at the prefix and emits the true `y_t`.
Only cases where the two answers differ are counted; "other" means the model emitted neither.

**Post-hoc linear probes.**  Fresh linear probes, fit on held-out data on the frozen residual stream
(after layer 1 and after layer 2), for both algorithms' states: R² for the absolute counts, R² for the
output index `t`, and bit accuracy for the availability bits.

**Attention patterns** at the ten decision points of one length-10 example.
