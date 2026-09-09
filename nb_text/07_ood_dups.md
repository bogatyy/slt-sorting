## 7. Control: many duplicates at an in-distribution length

At `n = 10` nothing about the sequence length changes.  Drawing the digits from a smaller alphabet
makes the histogram unusual (with alphabet `{0}` one digit occurs 10 times, which never happens in
training), but the cumulative counts stay in their training range `0..10`.  This separates "the input
is longer" from "the counts are unusual".
