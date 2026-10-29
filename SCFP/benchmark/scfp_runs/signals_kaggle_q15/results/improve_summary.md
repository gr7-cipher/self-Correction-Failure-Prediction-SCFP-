# Combining text and generator-side signals (5-fold grouped CV, nested stacking)

Accuracy uses a 0.5 threshold on class-balanced models. Hidden state: last_R0.

## natural set, target: final answer wrong (paper label)

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

## natural set, target: initial answer wrong (diagnostic)

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

## v1 set, target: final answer wrong (paper label)

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

## v1 set, target: initial answer wrong (diagnostic)

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

## Learning curve (natural set, paper label): out-of-fold AUC vs. training traces per fold

| Predictor | 25% (~247 traces) | 50% (~494 traces) | 75% (~742 traces) | 100% (~989 traces) |
|---|---|---|---|---|
| text_tfidf | 0.695 ± 0.019 | 0.694 ± 0.007 | 0.706 ± 0.008 | 0.703 ± 0.000 |
| stack_text_scalar_hidden | 0.707 ± 0.024 | 0.731 ± 0.013 | 0.740 ± 0.008 | 0.737 ± 0.000 |
