# Generator-side signals: out-of-fold results (5-fold, grouped by problem)

## natural set, target: final answer wrong (paper label)

| Generator | Predictor | n | AUC [95% CI] | Accuracy | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | majority | 1236 | 0.500 [0.500, 0.500] | 66.7% | 0.400 | -- | -- |
| qwen2.5-1.5b | text_tfidf | 1236 | 0.703 [0.674, 0.734] | 68.3% | 0.665 | 0.437 | -- |
| qwen2.5-1.5b | signals_scalar | 1236 | 0.722 [0.693, 0.751] | 67.9% | 0.659 | 0.572 | 0.762 |
| smollm2-1.7b | majority | 1950 | 0.500 [0.500, 0.500] | 52.9% | 0.346 | -- | -- |
| smollm2-1.7b | text_tfidf | 1950 | 0.606 [0.580, 0.632] | 58.9% | 0.588 | 0.000 | -- |
| smollm2-1.7b | signals_scalar | 1950 | 0.598 [0.572, 0.623] | 55.6% | 0.556 | 0.100 | 0.016 |
| qwen2.5-3b | majority | 1920 | 0.500 [0.500, 0.500] | 74.0% | 0.425 | -- | -- |
| qwen2.5-3b | text_tfidf | 1920 | 0.744 [0.722, 0.765] | 71.5% | 0.645 | 0.044 | -- |
| qwen2.5-3b | signals_scalar | 1920 | 0.767 [0.746, 0.789] | 71.1% | 0.671 | 0.048 | 0.813 |
| all | majority | 5106 | 0.500 [0.500, 0.500] | 62.0% | 0.383 | -- | -- |
| all | text_tfidf | 5106 | 0.685 [0.671, 0.700] | 65.9% | 0.644 | 0.000 | -- |
| all | signals_scalar | 5106 | 0.697 [0.683, 0.711] | 64.4% | 0.632 | 0.009 | 0.040 |

## natural set, target: initial answer wrong (diagnostic)

| Generator | Predictor | n | AUC [95% CI] | Accuracy | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | majority | 1236 | 0.500 [0.500, 0.500] | 70.2% | 0.413 | -- | -- |
| qwen2.5-1.5b | text_tfidf | 1236 | 0.699 [0.664, 0.732] | 67.7% | 0.651 | 0.192 | -- |
| qwen2.5-1.5b | signals_scalar | 1236 | 0.722 [0.690, 0.750] | 68.7% | 0.657 | 0.421 | 0.402 |
| smollm2-1.7b | majority | 1950 | 0.500 [0.500, 0.500] | 58.2% | 0.368 | -- | -- |
| smollm2-1.7b | text_tfidf | 1950 | 0.680 [0.657, 0.702] | 62.9% | 0.617 | 0.001 | -- |
| smollm2-1.7b | signals_scalar | 1950 | 0.669 [0.645, 0.691] | 61.9% | 0.619 | 0.032 | 0.487 |
| qwen2.5-3b | majority | 1920 | 0.500 [0.500, 0.500] | 75.9% | 0.432 | -- | -- |
| qwen2.5-3b | text_tfidf | 1920 | 0.744 [0.721, 0.766] | 70.8% | 0.612 | 0.000 | -- |
| qwen2.5-3b | signals_scalar | 1920 | 0.748 [0.726, 0.771] | 69.0% | 0.642 | 0.000 | 0.114 |
| all | majority | 5106 | 0.500 [0.500, 0.500] | 61.5% | 0.381 | -- | -- |
| all | text_tfidf | 5106 | 0.719 [0.705, 0.732] | 67.1% | 0.659 | 0.000 | -- |
| all | signals_scalar | 5106 | 0.685 [0.670, 0.700] | 66.2% | 0.646 | 0.000 | 0.257 |

## v1 set, target: final answer wrong (paper label)

| Generator | Predictor | n | AUC [95% CI] | Accuracy | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | majority | 177 | 0.500 [0.500, 0.500] | 65.0% | 0.394 | -- | -- |
| qwen2.5-1.5b | text_tfidf | 177 | 0.777 [0.704, 0.840] | 75.1% | 0.741 | 0.057 | -- |
| qwen2.5-1.5b | signals_scalar | 177 | 0.745 [0.667, 0.808] | 71.8% | 0.698 | 0.188 | 0.263 |
| smollm2-1.7b | majority | 295 | 0.500 [0.500, 0.500] | 53.6% | 0.349 | -- | -- |
| smollm2-1.7b | text_tfidf | 295 | 0.572 [0.503, 0.637] | 57.3% | 0.571 | 0.393 | -- |
| smollm2-1.7b | signals_scalar | 295 | 0.552 [0.487, 0.613] | 53.6% | 0.532 | 1.000 | 0.300 |
| all | majority | 472 | 0.500 [0.500, 0.500] | 53.4% | 0.348 | -- | -- |
| all | text_tfidf | 472 | 0.648 [0.599, 0.697] | 64.0% | 0.640 | 0.001 | -- |
| all | signals_scalar | 472 | 0.633 [0.585, 0.687] | 60.4% | 0.603 | 0.035 | 0.132 |

## v1 set, target: initial answer wrong (diagnostic)

| Generator | Predictor | n | AUC [95% CI] | Accuracy | Macro F1 | p vs majority | p vs TF-IDF |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | majority | 177 | 0.500 [0.500, 0.500] | 69.5% | 0.410 | -- | -- |
| qwen2.5-1.5b | text_tfidf | 177 | 0.768 [0.690, 0.843] | 72.9% | 0.710 | 0.572 | -- |
| qwen2.5-1.5b | signals_scalar | 177 | 0.769 [0.685, 0.848] | 75.1% | 0.725 | 0.275 | 0.523 |
| smollm2-1.7b | majority | 295 | 0.500 [0.500, 0.500] | 59.3% | 0.372 | -- | -- |
| smollm2-1.7b | text_tfidf | 295 | 0.603 [0.538, 0.671] | 59.3% | 0.570 | 1.000 | -- |
| smollm2-1.7b | signals_scalar | 295 | 0.616 [0.550, 0.675] | 58.6% | 0.579 | 0.932 | 0.931 |
| all | majority | 472 | 0.500 [0.500, 0.500] | 51.5% | 0.340 | -- | -- |
| all | text_tfidf | 472 | 0.696 [0.650, 0.740] | 64.4% | 0.643 | 0.000 | -- |
| all | signals_scalar | 472 | 0.679 [0.633, 0.726] | 64.8% | 0.648 | 0.000 | 0.936 |
