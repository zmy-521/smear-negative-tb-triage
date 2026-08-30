# Fixed nested-CV splits

These files contain **row indices only**; they contain no patient measurements or identifiers.

- `nested_cv_splits_aim1_aim2.npz`: n=4,177
- `nested_cv_splits_aim3.npz`: n=625

Keys are `rep{0-4}_fold{0-4}_train` and `rep{0-4}_fold{0-4}_test`. The repeated outer-loop seeds were 42–46. The indices refer to the row order of the private modelling datasets and are therefore useful only with the corresponding original row order.
