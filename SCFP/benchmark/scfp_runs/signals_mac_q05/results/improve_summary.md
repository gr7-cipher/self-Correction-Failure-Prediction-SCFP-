# Combining text and generator-side signals (5-fold grouped CV, nested stacking)

Accuracy uses a 0.5 threshold on class-balanced models. Hidden state: L24_R0.

## natural set, target: final answer wrong (paper label)

Majority class: 55.6% (n = 471)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.640 [0.592, 0.689] | 62.8% [58.6%, 67.5%] | 0.621 | 0.019 | -- |
| signals_scalar | 0.627 [0.578, 0.677] | 63.5% [59.2%, 68.2%] | 0.624 | 0.008 | 0.815 |
| signals_gbm | 0.646 [0.598, 0.696] | 61.1% [56.9%, 65.6%] | 0.608 | 0.089 | 0.546 |
| hidden_L24_R0 | 0.619 [0.568, 0.670] | 61.8% [57.5%, 66.0%] | 0.615 | 0.059 | 0.672 |
| concat_text_scalar | 0.636 [0.585, 0.688] | 61.6% [56.9%, 65.8%] | 0.608 | 0.054 | 0.586 |
| stack_text_scalar | 0.620 [0.570, 0.671] | 60.3% [55.8%, 64.5%] | 0.597 | 0.143 | 0.207 |
| stack_text_scalar_hidden | 0.623 [0.571, 0.672] | 63.3% [59.0%, 67.3%] | 0.629 | 0.017 | 0.913 |
| stack_all | 0.642 [0.589, 0.691] | 62.4% [58.0%, 66.9%] | 0.621 | 0.037 | 0.920 |

## natural set, target: initial answer wrong (diagnostic)

Majority class: 70.5% (n = 471)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.654 [0.600, 0.706] | 66.9% [62.6%, 71.1%] | 0.575 | 0.128 | -- |
| signals_scalar | 0.695 [0.640, 0.741] | 62.6% [58.2%, 66.9%] | 0.607 | 0.017 | 0.138 |
| signals_gbm | 0.691 [0.638, 0.742] | 67.3% [63.1%, 71.6%] | 0.630 | 0.284 | 0.937 |
| hidden_L24_R0 | 0.634 [0.580, 0.687] | 59.0% [54.8%, 63.3%] | 0.566 | 0.000 | 0.003 |
| concat_text_scalar | 0.674 [0.618, 0.724] | 67.5% [63.1%, 71.3%] | 0.614 | 0.279 | 0.847 |
| stack_text_scalar | 0.684 [0.628, 0.735] | 62.2% [57.5%, 66.5%] | 0.606 | 0.013 | 0.082 |
| stack_text_scalar_hidden | 0.682 [0.626, 0.731] | 60.7% [56.1%, 65.0%] | 0.593 | 0.004 | 0.022 |
| stack_all | 0.686 [0.632, 0.737] | 60.1% [55.4%, 64.3%] | 0.587 | 0.002 | 0.017 |

## v1 set, target: final answer wrong (paper label)

Majority class: 52.0% (n = 244)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.511 [0.438, 0.584] | 54.5% [48.4%, 60.7%] | 0.545 | 0.648 | -- |
| signals_scalar | 0.479 [0.410, 0.549] | 50.0% [44.3%, 56.1%] | 0.500 | 0.729 | 0.260 |
| signals_gbm | 0.460 [0.386, 0.534] | 47.1% [41.0%, 53.7%] | 0.468 | 0.290 | 0.092 |
| hidden_L24_R0 | 0.502 [0.423, 0.573] | 49.2% [42.6%, 54.9%] | 0.492 | 0.597 | 0.193 |
| concat_text_scalar | 0.502 [0.432, 0.575] | 51.2% [45.5%, 57.4%] | 0.512 | 0.929 | 0.389 |
| stack_text_scalar | 0.460 [0.385, 0.535] | 47.5% [41.4%, 54.1%] | 0.468 | 0.315 | 0.114 |
| stack_text_scalar_hidden | 0.477 [0.403, 0.551] | 48.4% [42.6%, 54.5%] | 0.483 | 0.467 | 0.163 |
| stack_all | 0.493 [0.424, 0.563] | 50.4% [44.3%, 57.0%] | 0.502 | 0.777 | 0.395 |

## v1 set, target: initial answer wrong (diagnostic)

Majority class: 60.7% (n = 244)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.577 [0.504, 0.649] | 59.0% [53.3%, 64.8%] | 0.578 | 0.771 | -- |
| signals_scalar | 0.666 [0.598, 0.729] | 61.5% [54.9%, 67.2%] | 0.612 | 0.930 | 0.539 |
| signals_gbm | 0.612 [0.540, 0.680] | 57.8% [51.6%, 63.9%] | 0.568 | 0.569 | 0.832 |
| hidden_L24_R0 | 0.683 [0.613, 0.748] | 64.8% [59.0%, 70.5%] | 0.643 | 0.411 | 0.065 |
| concat_text_scalar | 0.655 [0.585, 0.720] | 61.9% [55.3%, 67.6%] | 0.615 | 0.858 | 0.450 |
| stack_text_scalar | 0.660 [0.593, 0.724] | 60.7% [54.5%, 66.4%] | 0.606 | 1.000 | 0.694 |
| stack_text_scalar_hidden | 0.661 [0.591, 0.729] | 61.1% [54.5%, 66.8%] | 0.610 | 1.000 | 0.560 |
| stack_all | 0.650 [0.580, 0.713] | 59.8% [53.3%, 65.6%] | 0.596 | 0.930 | 0.890 |

## Learning curve (natural set, paper label): out-of-fold AUC vs. training traces per fold

| Predictor | 25% (~94 traces) | 50% (~188 traces) | 75% (~283 traces) | 100% (~377 traces) |
|---|---|---|---|---|
| text_tfidf | 0.580 ± 0.013 | 0.605 ± 0.016 | 0.620 ± 0.007 | 0.640 ± 0.000 |
| stack_text_scalar_hidden | 0.557 ± 0.030 | 0.580 ± 0.021 | 0.623 ± 0.022 | 0.623 ± 0.000 |
