# Combining text and generator-side signals (5-fold grouped CV, nested stacking)

Accuracy uses a 0.5 threshold on class-balanced models. Hidden state: L24_R0.

## natural set, target: final answer wrong (paper label)

Majority class: 56.6% (n = 1718)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.611 [0.585, 0.637] | 59.7% [57.4%, 62.0%] | 0.589 | 0.055 | -- |
| signals_scalar | 0.615 [0.589, 0.641] | 58.4% [56.2%, 60.6%] | 0.583 | 0.308 | 0.357 |
| signals_gbm | 0.619 [0.593, 0.645] | 58.9% [56.6%, 61.1%] | 0.585 | 0.179 | 0.590 |
| hidden_L24_R0 | 0.593 [0.566, 0.622] | 57.5% [55.1%, 60.0%] | 0.571 | 0.649 | 0.107 |
| concat_text_scalar | 0.616 [0.589, 0.641] | 58.6% [56.4%, 60.8%] | 0.582 | 0.262 | 0.299 |
| stack_text_scalar | 0.634 [0.608, 0.659] | 60.4% [58.0%, 62.6%] | 0.603 | 0.032 | 0.563 |
| stack_text_scalar_hidden | 0.633 [0.607, 0.658] | 60.4% [58.0%, 62.6%] | 0.603 | 0.030 | 0.546 |
| stack_all | 0.641 [0.614, 0.665] | 62.2% [60.0%, 64.4%] | 0.621 | 0.001 | 0.031 |

## natural set, target: initial answer wrong (diagnostic)

Majority class: 67.7% (n = 1718)

| Predictor | AUC [95% CI] | Accuracy [95% CI] | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|
| text_tfidf | 0.679 [0.656, 0.705] | 65.8% [63.7%, 68.0%] | 0.603 | 0.175 | -- |
| signals_scalar | 0.704 [0.681, 0.728] | 62.9% [60.8%, 65.2%] | 0.620 | 0.007 | 0.031 |
| signals_gbm | 0.746 [0.724, 0.770] | 65.2% [62.8%, 67.5%] | 0.639 | 0.147 | 0.667 |
| hidden_L24_R0 | 0.698 [0.675, 0.723] | 62.5% [60.4%, 64.8%] | 0.619 | 0.004 | 0.015 |
| concat_text_scalar | 0.696 [0.672, 0.723] | 65.5% [63.3%, 67.8%] | 0.622 | 0.149 | 0.776 |
| stack_text_scalar | 0.709 [0.685, 0.733] | 62.2% [60.0%, 64.4%] | 0.614 | 0.002 | 0.003 |
| stack_text_scalar_hidden | 0.707 [0.683, 0.731] | 62.7% [60.6%, 65.0%] | 0.622 | 0.006 | 0.019 |
| stack_all | 0.741 [0.719, 0.764] | 62.9% [60.6%, 65.1%] | 0.624 | 0.009 | 0.031 |

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

| Predictor | 25% (~344 traces) | 50% (~687 traces) | 75% (~1031 traces) | 100% (~1374 traces) |
|---|---|---|---|---|
| text_tfidf | 0.597 ± 0.019 | 0.611 ± 0.008 | 0.612 ± 0.004 | 0.611 ± 0.000 |
| stack_text_scalar_hidden | 0.608 ± 0.008 | 0.621 ± 0.002 | 0.629 ± 0.007 | 0.633 ± 0.000 |
