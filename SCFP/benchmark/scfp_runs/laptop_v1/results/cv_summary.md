# 5-fold grouped cross-validation (out-of-fold predictions on all labeled traces)

| Model | n | Accuracy [95% CI] | Macro F1 | AUC | p vs TF-IDF | p vs majority |
|---|---|---|---|---|---|---|
| tfidf_lr | 716 | 57.3% [53.5, 60.8] | 0.571 | 0.577 | -- | 0.142 |
| length | 716 | 55.4% [51.8, 59.2] | 0.507 | 0.545 | 0.498 | 0.173 |
| confidence | 716 | 55.0% [51.4, 58.4] | 0.547 | 0.556 | 0.329 | 0.504 |
| majority | 716 | 52.9% [49.0, 56.6] | 0.346 | 0.465 | 0.142 | -- |
| random | 716 | 49.9% [46.2, 53.6] | 0.498 | 0.503 | 0.006 | 0.262 |
