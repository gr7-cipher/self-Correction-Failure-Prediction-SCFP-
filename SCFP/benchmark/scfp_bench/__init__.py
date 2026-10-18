"""SCFP benchmark construction pipeline.

Stages (run in order, each reads the previous stage's output under one run dir):
  holdout  -> reserve the routing problems so no later stage can touch them
  filter   -> estimate per-problem accuracy and keep problems in the target band
  generate -> collect (R0, F, Rf) correction traces from each generator model
  label    -> score R0 and Rf against ground truth; derive the binary outcome
  split    -> split by problem into train/val/test and write a hash manifest
  export   -> write failed traces as annotation tasks
  agree    -> import annotations, compute Fleiss' kappa, build final labels
  validate -> refuse to release unless every trace is backed by a logged call
"""

__version__ = "0.1.0"
