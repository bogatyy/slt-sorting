## 7. Distribution shift without new positions: many duplicates

At `n = 10` every position has been seen in training.  Drawing the digits from a smaller alphabet
pushes the *counts* far above anything seen in training (with the full alphabet a digit rarely
appears more than 4 times; with alphabet `{0}` it appears 10 times).  A counting sorter has to
represent these counts; a min-tracker only needs "is there still one left?".
