# Combining text and generator-side signals (5-fold grouped CV, nested stacking)

Accuracy uses a 0.5 threshold on class-balanced models. Hidden state: last_R0.

## natural set, target: final answer wrong (paper label)

Majority class: 56.6% (n = 1718)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.611 [0.585, 0.637] | 59.7% [57.4%, 62.0%] | 0.589 | 0.055 | -- |
| signals_scalar | 0.729 [0.706, 0.751] | 68.6% [66.5%, 70.8%] | 0.680 | 0.000 | 0.000 |
| signals_gbm | 0.726 [0.702, 0.748] | 66.6% [64.5%, 68.7%] | 0.660 | 0.000 | 0.000 |
| hidden_last_R0 | 0.593 [0.566, 0.622] | 57.5% [55.1%, 60.0%] | 0.571 | 0.649 | 0.107 |
| concat_text_scalar | 0.732 [0.709, 0.754] | 68.4% [66.3%, 70.5%] | 0.678 | 0.000 | 0.000 |
| stack_text_scalar | 0.734 [0.712, 0.756] | 68.5% [66.4%, 70.7%] | 0.679 | 0.000 | 0.000 |
| stack_text_scalar_hidden | 0.734 [0.711, 0.756] | 68.5% [66.3%, 70.6%] | 0.678 | 0.000 | 0.000 |
| stack_all | 0.734 [0.711, 0.757] | 68.0% [65.8%, 70.0%] | 0.674 | 0.000 | 0.000 |

## natural set, target: initial answer wrong (diagnostic)

Majority class: 67.7% (n = 1718)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.679 [0.656, 0.705] | 65.8% [63.7%, 68.0%] | 0.603 | 0.175 | -- |
| signals_scalar | 0.757 [0.734, 0.779] | 66.1% [63.9%, 68.2%] | 0.652 | 0.366 | 0.897 |
| signals_gbm | 0.775 [0.754, 0.798] | 67.3% [65.4%, 69.6%] | 0.662 | 0.864 | 0.285 |
| hidden_last_R0 | 0.698 [0.675, 0.723] | 62.5% [60.4%, 64.8%] | 0.619 | 0.004 | 0.015 |
| concat_text_scalar | 0.748 [0.723, 0.771] | 67.2% [65.0%, 69.3%] | 0.650 | 0.796 | 0.250 |
| stack_text_scalar | 0.754 [0.731, 0.776] | 64.8% [62.6%, 66.9%] | 0.640 | 0.104 | 0.431 |
| stack_text_scalar_hidden | 0.748 [0.725, 0.769] | 65.3% [63.3%, 67.5%] | 0.646 | 0.188 | 0.720 |
| stack_all | 0.773 [0.751, 0.795] | 65.4% [63.2%, 67.5%] | 0.648 | 0.206 | 0.767 |

## v1 set, target: final answer wrong (paper label)

Majority class: 52.0% (n = 244)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.511 [0.438, 0.584] | 54.5% [48.4%, 60.7%] | 0.545 | 0.648 | -- |
| signals_scalar | 0.635 [0.565, 0.706] | 61.9% [55.7%, 67.6%] | 0.619 | 0.038 | 0.054 |
| signals_gbm | 0.553 [0.481, 0.624] | 54.1% [48.0%, 60.3%] | 0.538 | 0.699 | 1.000 |
| hidden_last_R0 | 0.502 [0.423, 0.573] | 49.2% [42.6%, 54.9%] | 0.492 | 0.597 | 0.193 |
| concat_text_scalar | 0.618 [0.550, 0.688] | 59.4% [53.3%, 65.2%] | 0.594 | 0.127 | 0.195 |
| stack_text_scalar | 0.625 [0.555, 0.694] | 61.9% [55.3%, 67.2%] | 0.619 | 0.037 | 0.050 |
| stack_text_scalar_hidden | 0.594 [0.521, 0.665] | 61.1% [54.9%, 66.8%] | 0.611 | 0.063 | 0.072 |
| stack_all | 0.558 [0.492, 0.633] | 54.9% [48.8%, 61.9%] | 0.549 | 0.589 | 1.000 |

## v1 set, target: initial answer wrong (diagnostic)

Majority class: 60.7% (n = 244)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.577 [0.504, 0.649] | 59.0% [53.3%, 64.8%] | 0.578 | 0.771 | -- |
| signals_scalar | 0.706 [0.642, 0.772] | 63.1% [56.6%, 68.9%] | 0.628 | 0.656 | 0.282 |
| signals_gbm | 0.662 [0.594, 0.726] | 60.2% [54.9%, 66.0%] | 0.592 | 1.000 | 0.822 |
| hidden_last_R0 | 0.683 [0.613, 0.748] | 64.8% [59.0%, 70.5%] | 0.643 | 0.411 | 0.065 |
| concat_text_scalar | 0.708 [0.646, 0.772] | 62.7% [56.1%, 68.4%] | 0.620 | 0.709 | 0.306 |
| stack_text_scalar | 0.694 [0.627, 0.757] | 62.3% [56.1%, 68.4%] | 0.622 | 0.799 | 0.358 |
| stack_text_scalar_hidden | 0.710 [0.649, 0.772] | 63.9% [58.6%, 70.1%] | 0.639 | 0.551 | 0.126 |
| stack_all | 0.676 [0.611, 0.739] | 63.5% [57.8%, 69.3%] | 0.633 | 0.600 | 0.177 |
