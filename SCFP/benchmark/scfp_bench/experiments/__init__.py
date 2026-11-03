"""Experiments on the SCFP benchmark.

All predictors see only the partial trace (P, R0, F), never the final
response Rf. Every run writes per-example predictions to
<run_dir>/predictions/<model>__<task>.jsonl in the format read by
verification/verify_metrics.py:
  {"trace_id", "model", "task", "split", "y_true", "y_pred", "p_fail"}
"""
