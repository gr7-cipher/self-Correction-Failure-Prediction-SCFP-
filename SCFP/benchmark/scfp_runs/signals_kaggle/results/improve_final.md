# Combining text and generator-side signals (5-fold grouped CV, nested stacking)

Accuracy uses a 0.5 threshold on class-balanced models. Hidden state: last_R0.

## natural set, target: final answer wrong (paper label), generator: qwen2.5-1.5b

Majority class: 66.7% (n = 1236)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.703 [0.674, 0.734] | 68.3% [65.7%, 70.9%] | 0.665 | 0.437 | -- |
| signals_scalar | 0.750 [0.722, 0.776] | 69.3% [66.7%, 72.0%] | 0.678 | 0.187 | 0.193 |
| signals_gbm | 0.752 [0.724, 0.779] | 68.9% [66.4%, 71.4%] | 0.675 | 0.274 | 0.643 |
| concat_text_scalar | 0.741 [0.715, 0.769] | 69.9% [67.6%, 72.6%] | 0.673 | 0.081 | 0.210 |
| stack_text_scalar | 0.749 [0.721, 0.776] | 69.1% [66.5%, 71.7%] | 0.676 | 0.235 | 0.320 |

## natural set, target: final answer wrong (paper label), generator: smollm2-1.7b

Majority class: 52.9% (n = 1950)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.606 [0.580, 0.632] | 58.9% [56.7%, 61.0%] | 0.588 | 0.000 | -- |
| signals_scalar | 0.708 [0.686, 0.733] | 67.3% [65.2%, 69.4%] | 0.671 | 0.000 | 0.000 |
| signals_gbm | 0.696 [0.672, 0.721] | 66.5% [64.5%, 68.6%] | 0.662 | 0.000 | 0.000 |
| concat_text_scalar | 0.710 [0.686, 0.734] | 67.0% [65.0%, 69.1%] | 0.668 | 0.000 | 0.000 |
| stack_text_scalar | 0.712 [0.689, 0.737] | 67.1% [65.0%, 69.2%] | 0.668 | 0.000 | 0.000 |

## natural set, target: final answer wrong (paper label), generator: qwen2.5-3b

Majority class: 74.0% (n = 1920)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.744 [0.722, 0.765] | 71.5% [69.5%, 73.4%] | 0.645 | 0.044 | -- |
| signals_scalar | 0.787 [0.767, 0.807] | 67.9% [65.9%, 70.1%] | 0.655 | 0.000 | 0.002 |
| signals_gbm | 0.794 [0.773, 0.814] | 68.2% [66.2%, 70.4%] | 0.660 | 0.000 | 0.008 |
| concat_text_scalar | 0.793 [0.773, 0.812] | 73.6% [71.7%, 75.5%] | 0.673 | 0.802 | 0.019 |
| stack_text_scalar | 0.782 [0.762, 0.802] | 68.4% [66.2%, 70.5%] | 0.658 | 0.000 | 0.004 |

## natural set, target: final answer wrong (paper label), generator: pooled (one model over all generators)

Majority class: 62.0% (n = 5106)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.716 [0.702, 0.731] | 64.8% [63.6%, 66.1%] | 0.631 | 0.002 | -- |
| signals_scalar | 0.746 [0.734, 0.760] | 65.5% [64.2%, 66.9%] | 0.653 | 0.001 | 0.390 |
| signals_gbm | 0.776 [0.763, 0.788] | 68.1% [66.7%, 69.4%] | 0.679 | 0.000 | 0.000 |
| concat_text_scalar | 0.759 [0.746, 0.773] | 68.1% [66.8%, 69.3%] | 0.668 | 0.000 | 0.000 |
| stack_text_scalar | 0.760 [0.748, 0.774] | 67.3% [66.1%, 68.6%] | 0.669 | 0.000 | 0.000 |

## natural set, target: initial answer wrong (diagnostic), generator: qwen2.5-1.5b

Majority class: 70.2% (n = 1236)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.699 [0.664, 0.732] | 67.7% [65.0%, 70.3%] | 0.651 | 0.192 | -- |
| signals_scalar | 0.788 [0.760, 0.812] | 68.4% [65.8%, 71.0%] | 0.674 | 0.409 | 0.612 |
| signals_gbm | 0.776 [0.748, 0.801] | 70.5% [67.7%, 73.0%] | 0.685 | 0.932 | 0.033 |
| concat_text_scalar | 0.790 [0.762, 0.814] | 68.7% [65.9%, 71.4%] | 0.675 | 0.474 | 0.490 |
| stack_text_scalar | 0.787 [0.759, 0.811] | 68.0% [65.3%, 70.5%] | 0.671 | 0.295 | 0.898 |

## natural set, target: initial answer wrong (diagnostic), generator: smollm2-1.7b

Majority class: 58.2% (n = 1950)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.680 [0.657, 0.702] | 62.9% [60.8%, 65.0%] | 0.617 | 0.001 | -- |
| signals_scalar | 0.768 [0.746, 0.788] | 70.4% [68.2%, 72.4%] | 0.698 | 0.000 | 0.000 |
| signals_gbm | 0.766 [0.744, 0.787] | 69.5% [67.5%, 71.7%] | 0.690 | 0.000 | 0.000 |
| concat_text_scalar | 0.771 [0.749, 0.790] | 70.1% [68.1%, 72.3%] | 0.695 | 0.000 | 0.000 |
| stack_text_scalar | 0.777 [0.756, 0.797] | 70.6% [68.5%, 72.6%] | 0.701 | 0.000 | 0.000 |

## natural set, target: initial answer wrong (diagnostic), generator: qwen2.5-3b

Majority class: 75.9% (n = 1920)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.744 [0.721, 0.766] | 70.8% [68.7%, 72.8%] | 0.612 | 0.000 | -- |
| signals_scalar | 0.773 [0.753, 0.793] | 66.9% [64.8%, 69.0%] | 0.639 | 0.000 | 0.001 |
| signals_gbm | 0.757 [0.735, 0.778] | 65.5% [63.3%, 67.6%] | 0.623 | 0.000 | 0.000 |
| concat_text_scalar | 0.773 [0.752, 0.793] | 72.1% [70.1%, 74.1%] | 0.640 | 0.002 | 0.163 |
| stack_text_scalar | 0.763 [0.742, 0.784] | 66.6% [64.5%, 68.5%] | 0.638 | 0.000 | 0.000 |

## natural set, target: initial answer wrong (diagnostic), generator: pooled (one model over all generators)

Majority class: 61.5% (n = 5106)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.753 [0.740, 0.766] | 68.1% [66.8%, 69.4%] | 0.674 | 0.000 | -- |
| signals_scalar | 0.790 [0.778, 0.801] | 69.0% [67.7%, 70.2%] | 0.687 | 0.000 | 0.260 |
| signals_gbm | 0.811 [0.800, 0.822] | 70.0% [68.8%, 71.2%] | 0.696 | 0.000 | 0.008 |
| concat_text_scalar | 0.810 [0.799, 0.821] | 72.2% [71.0%, 73.4%] | 0.710 | 0.000 | 0.000 |
| stack_text_scalar | 0.806 [0.794, 0.816] | 69.3% [68.0%, 70.5%] | 0.691 | 0.000 | 0.066 |

## v1 set, target: final answer wrong (paper label), generator: qwen2.5-1.5b

Majority class: 65.0% (n = 177)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.777 [0.704, 0.840] | 75.1% [68.9%, 81.4%] | 0.741 | 0.057 | -- |
| signals_scalar | 0.757 [0.680, 0.829] | 73.4% [67.2%, 79.1%] | 0.722 | 0.110 | 0.250 |
| signals_gbm | 0.777 [0.696, 0.847] | 73.4% [66.1%, 79.1%] | 0.715 | 0.091 | 0.711 |
| concat_text_scalar | 0.777 [0.702, 0.843] | 74.0% [67.8%, 80.2%] | 0.727 | 0.085 | 0.688 |
| stack_text_scalar | 0.724 [0.644, 0.797] | 67.8% [61.0%, 74.6%] | 0.671 | 0.672 | 0.001 |

## v1 set, target: final answer wrong (paper label), generator: smollm2-1.7b

Majority class: 53.6% (n = 295)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.572 [0.503, 0.637] | 57.3% [51.5%, 62.7%] | 0.571 | 0.393 | -- |
| signals_scalar | 0.665 [0.604, 0.727] | 62.7% [56.9%, 68.1%] | 0.620 | 0.017 | 0.101 |
| signals_gbm | 0.625 [0.564, 0.686] | 57.3% [52.2%, 63.1%] | 0.569 | 0.379 | 1.000 |
| concat_text_scalar | 0.669 [0.612, 0.730] | 62.0% [56.3%, 67.1%] | 0.610 | 0.022 | 0.141 |
| stack_text_scalar | 0.634 [0.569, 0.699] | 60.7% [54.9%, 66.1%] | 0.600 | 0.066 | 0.337 |

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
| concat_text_scalar | 0.864 [0.805, 0.913] | 74.6% [68.4%, 80.8%] | 0.727 | 0.362 | 0.711 |
| stack_text_scalar | 0.818 [0.753, 0.876] | 68.4% [61.6%, 75.1%] | 0.676 | 0.919 | 0.185 |

## v1 set, target: initial answer wrong (diagnostic), generator: smollm2-1.7b

Majority class: 59.3% (n = 295)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.603 [0.538, 0.671] | 59.3% [53.9%, 65.1%] | 0.570 | 1.000 | -- |
| signals_scalar | 0.716 [0.658, 0.770] | 65.4% [60.3%, 70.5%] | 0.650 | 0.153 | 0.095 |
| signals_gbm | 0.735 [0.676, 0.790] | 65.8% [60.3%, 71.2%] | 0.646 | 0.101 | 0.067 |
| concat_text_scalar | 0.749 [0.694, 0.802] | 70.2% [65.4%, 75.6%] | 0.695 | 0.006 | 0.001 |
| stack_text_scalar | 0.729 [0.670, 0.783] | 66.4% [61.4%, 71.5%] | 0.664 | 0.117 | 0.048 |

## v1 set, target: initial answer wrong (diagnostic), generator: pooled (one model over all generators)

Majority class: 51.5% (n = 472)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.680 [0.632, 0.724] | 62.7% [58.5%, 66.7%] | 0.626 | 0.001 | -- |
| signals_scalar | 0.754 [0.711, 0.796] | 68.2% [64.0%, 72.5%] | 0.678 | 0.000 | 0.011 |
| signals_gbm | 0.801 [0.761, 0.838] | 70.3% [66.3%, 74.8%] | 0.703 | 0.000 | 0.004 |
| concat_text_scalar | 0.784 [0.745, 0.822] | 67.8% [63.8%, 72.0%] | 0.677 | 0.000 | 0.043 |
| stack_text_scalar | 0.737 [0.692, 0.782] | 68.0% [63.8%, 72.2%] | 0.674 | 0.000 | 0.010 |
