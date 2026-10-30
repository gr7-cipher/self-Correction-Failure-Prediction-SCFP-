# Combining text and generator-side signals (5-fold grouped CV, nested stacking)

Accuracy uses a 0.5 threshold on class-balanced models. Hidden state: last_R0.

## natural set, target: final answer wrong (paper label), generator: qwen2.5-1.5b

Majority class: 66.7% (n = 1236)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.703 [0.674, 0.734] | 68.3% [65.7%, 70.9%] | 0.665 | 0.437 | -- |
| signals_scalar | 0.749 [0.721, 0.775] | 69.3% [66.7%, 72.0%] | 0.677 | 0.186 | 0.187 |
| signals_gbm | 0.752 [0.725, 0.780] | 69.7% [67.3%, 72.2%] | 0.683 | 0.141 | 0.286 |
| hidden_last_R0 | 0.720 [0.690, 0.750] | 66.9% [64.3%, 69.5%] | 0.653 | 0.966 | 0.036 |
| concat_text_scalar | 0.741 [0.715, 0.769] | 69.8% [67.5%, 72.4%] | 0.670 | 0.086 | 0.228 |
| stack_text_scalar | 0.749 [0.720, 0.775] | 69.0% [66.4%, 71.7%] | 0.674 | 0.251 | 0.368 |
| stack_text_scalar_hidden | 0.737 [0.709, 0.766] | 68.3% [65.7%, 70.8%] | 0.667 | 0.445 | 1.000 |
| stack_all | 0.762 [0.734, 0.789] | 69.5% [67.1%, 72.0%] | 0.684 | 0.177 | 0.232 |

## natural set, target: final answer wrong (paper label), generator: smollm2-1.7b

Majority class: 52.9% (n = 1950)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.606 [0.580, 0.632] | 58.9% [56.7%, 61.0%] | 0.588 | 0.000 | -- |
| signals_scalar | 0.708 [0.686, 0.733] | 67.3% [65.2%, 69.4%] | 0.671 | 0.000 | 0.000 |
| signals_gbm | 0.696 [0.672, 0.721] | 66.5% [64.5%, 68.6%] | 0.662 | 0.000 | 0.000 |
| hidden_last_R0 | 0.624 [0.600, 0.651] | 59.2% [57.1%, 61.3%] | 0.592 | 0.000 | 0.865 |
| concat_text_scalar | 0.709 [0.686, 0.734] | 67.1% [65.1%, 69.2%] | 0.668 | 0.000 | 0.000 |
| stack_text_scalar | 0.712 [0.689, 0.737] | 67.1% [65.0%, 69.2%] | 0.668 | 0.000 | 0.000 |
| stack_text_scalar_hidden | 0.709 [0.686, 0.734] | 67.4% [65.4%, 69.5%] | 0.672 | 0.000 | 0.000 |
| stack_all | 0.709 [0.686, 0.733] | 67.1% [65.1%, 69.2%] | 0.669 | 0.000 | 0.000 |

## natural set, target: final answer wrong (paper label), generator: pooled (one model over all generators)

Majority class: 54.7% (n = 3186)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.646 [0.626, 0.664] | 58.6% [56.8%, 60.2%] | 0.586 | 0.002 | -- |
| signals_scalar | 0.719 [0.701, 0.736] | 65.1% [63.5%, 66.7%] | 0.650 | 0.000 | 0.000 |
| signals_gbm | 0.738 [0.721, 0.754] | 67.4% [65.8%, 69.0%] | 0.674 | 0.000 | 0.000 |
| concat_text_scalar | 0.729 [0.713, 0.748] | 66.9% [65.3%, 68.7%] | 0.669 | 0.000 | 0.000 |
| stack_text_scalar | 0.722 [0.705, 0.740] | 65.6% [63.9%, 67.3%] | 0.656 | 0.000 | 0.000 |

## natural set, target: initial answer wrong (diagnostic), generator: qwen2.5-1.5b

Majority class: 70.2% (n = 1236)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.699 [0.664, 0.732] | 67.7% [65.0%, 70.3%] | 0.651 | 0.192 | -- |
| signals_scalar | 0.787 [0.758, 0.811] | 68.7% [66.0%, 71.2%] | 0.675 | 0.475 | 0.488 |
| signals_gbm | 0.776 [0.747, 0.801] | 70.5% [67.9%, 73.0%] | 0.685 | 0.932 | 0.034 |
| hidden_last_R0 | 0.719 [0.689, 0.752] | 67.7% [64.8%, 70.4%] | 0.652 | 0.195 | 1.000 |
| concat_text_scalar | 0.781 [0.753, 0.807] | 69.7% [67.1%, 72.2%] | 0.680 | 0.804 | 0.166 |
| stack_text_scalar | 0.786 [0.757, 0.811] | 68.2% [65.5%, 70.7%] | 0.672 | 0.349 | 0.744 |
| stack_text_scalar_hidden | 0.790 [0.762, 0.814] | 69.6% [66.8%, 72.2%] | 0.684 | 0.780 | 0.094 |
| stack_all | 0.790 [0.762, 0.814] | 69.5% [66.7%, 72.2%] | 0.684 | 0.751 | 0.137 |

## natural set, target: initial answer wrong (diagnostic), generator: smollm2-1.7b

Majority class: 58.2% (n = 1950)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.680 [0.657, 0.702] | 62.9% [60.8%, 65.0%] | 0.617 | 0.001 | -- |
| signals_scalar | 0.768 [0.746, 0.788] | 70.4% [68.2%, 72.4%] | 0.698 | 0.000 | 0.000 |
| signals_gbm | 0.766 [0.744, 0.787] | 69.5% [67.5%, 71.7%] | 0.690 | 0.000 | 0.000 |
| hidden_last_R0 | 0.731 [0.710, 0.753] | 67.1% [65.1%, 69.2%] | 0.669 | 0.000 | 0.000 |
| concat_text_scalar | 0.771 [0.749, 0.790] | 70.1% [68.1%, 72.3%] | 0.695 | 0.000 | 0.000 |
| stack_text_scalar | 0.777 [0.756, 0.797] | 70.6% [68.5%, 72.6%] | 0.701 | 0.000 | 0.000 |
| stack_text_scalar_hidden | 0.782 [0.761, 0.802] | 70.6% [68.5%, 72.7%] | 0.700 | 0.000 | 0.000 |
| stack_all | 0.783 [0.762, 0.802] | 70.2% [68.2%, 72.3%] | 0.697 | 0.000 | 0.000 |

## natural set, target: initial answer wrong (diagnostic), generator: pooled (one model over all generators)

Majority class: 52.9% (n = 3186)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.730 [0.713, 0.748] | 66.3% [64.5%, 67.9%] | 0.662 | 0.000 | -- |
| signals_scalar | 0.785 [0.770, 0.800] | 69.7% [68.1%, 71.3%] | 0.697 | 0.000 | 0.001 |
| signals_gbm | 0.804 [0.789, 0.818] | 72.2% [70.6%, 73.8%] | 0.722 | 0.000 | 0.000 |
| concat_text_scalar | 0.803 [0.788, 0.819] | 71.7% [70.0%, 73.2%] | 0.716 | 0.000 | 0.000 |
| stack_text_scalar | 0.802 [0.787, 0.817] | 71.9% [70.3%, 73.5%] | 0.719 | 0.000 | 0.000 |

## v1 set, target: final answer wrong (paper label), generator: qwen2.5-1.5b

Majority class: 65.0% (n = 177)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.777 [0.704, 0.840] | 75.1% [68.9%, 81.4%] | 0.741 | 0.057 | -- |
| signals_scalar | 0.757 [0.680, 0.829] | 73.4% [67.2%, 79.1%] | 0.722 | 0.110 | 0.250 |
| signals_gbm | 0.777 [0.696, 0.847] | 73.4% [66.1%, 79.1%] | 0.715 | 0.091 | 0.711 |
| hidden_last_R0 | 0.793 [0.722, 0.857] | 75.7% [69.5%, 81.9%] | 0.743 | 0.034 | 1.000 |
| concat_text_scalar | 0.777 [0.702, 0.843] | 74.0% [67.8%, 80.2%] | 0.727 | 0.085 | 0.688 |
| stack_text_scalar | 0.724 [0.644, 0.797] | 67.8% [61.0%, 74.6%] | 0.671 | 0.672 | 0.001 |
| stack_text_scalar_hidden | 0.791 [0.720, 0.855] | 76.3% [70.1%, 81.9%] | 0.752 | 0.031 | 0.625 |
| stack_all | 0.797 [0.719, 0.861] | 77.4% [71.2%, 83.1%] | 0.764 | 0.017 | 0.344 |

## v1 set, target: final answer wrong (paper label), generator: smollm2-1.7b

Majority class: 53.6% (n = 295)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.572 [0.503, 0.637] | 57.3% [51.5%, 62.7%] | 0.571 | 0.393 | -- |
| signals_scalar | 0.665 [0.604, 0.727] | 62.7% [56.9%, 68.1%] | 0.620 | 0.017 | 0.101 |
| signals_gbm | 0.625 [0.564, 0.686] | 57.3% [52.2%, 63.1%] | 0.569 | 0.379 | 1.000 |
| hidden_last_R0 | 0.604 [0.540, 0.669] | 54.6% [49.2%, 61.0%] | 0.546 | 0.876 | 0.466 |
| concat_text_scalar | 0.669 [0.612, 0.730] | 62.0% [56.3%, 67.1%] | 0.610 | 0.022 | 0.141 |
| stack_text_scalar | 0.634 [0.569, 0.699] | 60.7% [54.9%, 66.1%] | 0.600 | 0.066 | 0.337 |
| stack_text_scalar_hidden | 0.662 [0.599, 0.724] | 61.4% [55.6%, 66.8%] | 0.609 | 0.049 | 0.235 |
| stack_all | 0.631 [0.566, 0.692] | 59.7% [54.2%, 65.1%] | 0.593 | 0.136 | 0.510 |

## v1 set, target: final answer wrong (paper label), generator: pooled (one model over all generators)

Majority class: 53.4% (n = 472)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.683 [0.637, 0.730] | 66.1% [61.9%, 70.6%] | 0.661 | 0.000 | -- |
| signals_scalar | 0.720 [0.670, 0.764] | 67.6% [63.3%, 71.8%] | 0.674 | 0.000 | 0.349 |
| signals_gbm | 0.741 [0.696, 0.786] | 66.7% [62.5%, 71.0%] | 0.667 | 0.000 | 0.855 |
| concat_text_scalar | 0.747 [0.704, 0.790] | 67.6% [63.8%, 71.8%] | 0.674 | 0.000 | 0.470 |
| stack_text_scalar | 0.722 [0.674, 0.764] | 66.7% [62.3%, 71.0%] | 0.666 | 0.000 | 0.749 |

## v1 set, target: initial answer wrong (diagnostic), generator: qwen2.5-1.5b

Majority class: 69.5% (n = 177)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.768 [0.690, 0.843] | 72.9% [66.1%, 79.7%] | 0.710 | 0.572 | -- |
| signals_scalar | 0.826 [0.762, 0.886] | 74.6% [68.4%, 80.8%] | 0.727 | 0.362 | 0.581 |
| signals_gbm | 0.805 [0.738, 0.862] | 71.8% [65.0%, 78.5%] | 0.690 | 0.720 | 0.878 |
| hidden_last_R0 | 0.735 [0.653, 0.821] | 71.8% [65.0%, 78.0%] | 0.692 | 0.724 | 0.774 |
| concat_text_scalar | 0.864 [0.805, 0.913] | 74.6% [68.4%, 80.8%] | 0.727 | 0.362 | 0.711 |
| stack_text_scalar | 0.818 [0.753, 0.876] | 68.4% [61.6%, 75.1%] | 0.676 | 0.919 | 0.185 |
| stack_text_scalar_hidden | 0.799 [0.731, 0.865] | 74.0% [67.2%, 80.2%] | 0.724 | 0.434 | 0.791 |
| stack_all | 0.819 [0.754, 0.874] | 71.2% [64.4%, 78.0%] | 0.701 | 0.832 | 0.701 |

## v1 set, target: initial answer wrong (diagnostic), generator: smollm2-1.7b

Majority class: 59.3% (n = 295)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.603 [0.538, 0.671] | 59.3% [53.9%, 65.1%] | 0.570 | 1.000 | -- |
| signals_scalar | 0.716 [0.658, 0.770] | 65.4% [60.3%, 70.5%] | 0.650 | 0.153 | 0.095 |
| signals_gbm | 0.735 [0.676, 0.790] | 65.8% [60.3%, 71.2%] | 0.646 | 0.101 | 0.067 |
| hidden_last_R0 | 0.679 [0.613, 0.738] | 64.1% [58.6%, 69.8%] | 0.639 | 0.295 | 0.130 |
| concat_text_scalar | 0.749 [0.694, 0.802] | 70.2% [65.4%, 75.6%] | 0.695 | 0.006 | 0.001 |
| stack_text_scalar | 0.729 [0.670, 0.783] | 66.4% [61.4%, 71.5%] | 0.664 | 0.117 | 0.048 |
| stack_text_scalar_hidden | 0.737 [0.678, 0.791] | 68.5% [63.4%, 73.9%] | 0.684 | 0.041 | 0.006 |
| stack_all | 0.752 [0.695, 0.807] | 69.5% [64.1%, 74.6%] | 0.692 | 0.015 | 0.003 |

## v1 set, target: initial answer wrong (diagnostic), generator: pooled (one model over all generators)

Majority class: 51.5% (n = 472)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.680 [0.632, 0.724] | 62.7% [58.5%, 66.7%] | 0.626 | 0.001 | -- |
| signals_scalar | 0.754 [0.711, 0.796] | 68.2% [64.0%, 72.5%] | 0.678 | 0.000 | 0.011 |
| signals_gbm | 0.801 [0.761, 0.838] | 70.3% [66.3%, 74.8%] | 0.703 | 0.000 | 0.004 |
| concat_text_scalar | 0.784 [0.745, 0.822] | 67.8% [63.8%, 72.0%] | 0.677 | 0.000 | 0.043 |
| stack_text_scalar | 0.737 [0.692, 0.782] | 68.0% [63.8%, 72.2%] | 0.674 | 0.000 | 0.010 |

## Learning curve (natural set, paper label): out-of-fold AUC vs. training traces per fold

| Predictor | 25% (~637 traces) | 50% (~1274 traces) | 75% (~1912 traces) | 100% (~2549 traces) |
|---|---|---|---|---|
| text_tfidf | 0.640 ± 0.010 | 0.643 ± 0.011 | 0.651 ± 0.008 | 0.646 ± 0.000 |
| stack_text_scalar | 0.698 ± 0.007 | 0.708 ± 0.007 | 0.712 ± 0.006 | 0.722 ± 0.000 |
