# Revision plan (pre-registered)

I append each entry before running the experiment it describes. I don't edit an entry after its run. A correction goes in a new, dated entry.

---

## M4: routing baselines without self-correction (registered 2026-10-31, before any M4 computation)

**Data.** The existing routing log is `SCFP/benchmark/scfp_runs/laptop_v1/routing/traces.jsonl`, with Qwen2.5-1.5B as the base model. The first 50 problems are for tuning and the other 150 for evaluation. The split is unchanged and nothing is generated again.

**New policies** (all computed from logged texts, tokens and latencies):

1. **No correction (R0 only).** Return the answer of R0.
   - Cost: R0 tokens, divided by the existing unit (the mean intrinsic tokens on eval, as in §7.2).
   - Latency: the R0 call only.
2. **Agreement tool.** Generate R0 and the PoT program, but no critique.
   - If the program's answer equals R0's answer, return R0. Otherwise return the program's answer.
   - If the program produced no answer (error or empty output), return R0. I'm adding this fallback because the rule is otherwise undefined.
   - Cost: R0 + program tokens. Latency: R0 + program + execution.
   - Note: whenever the program produces an answer, this policy returns the program's answer, so its accuracy differs from Always Extrinsic only on program failures. I'll report that, not hide it.
3. **Agreement + human.** As in policy 2, but disagreements and program failures go to the human oracle. The oracle uses the same assumed cost and latency as the existing human path, added on top of the R0 + program cost and latency.

**Metrics.**

- Accuracy, with a 95% bootstrap CI over problems (1,000 resamples, seed 0). There is one trace per problem, so a problem-level bootstrap and a trace-level bootstrap are the same here.
- Relative cost and mean latency.
- For each new policy against each existing fixed policy: a paired accuracy difference with a 95% paired bootstrap CI, and an exact McNemar p.

**Decision rule.**

- I'll say a policy is "better" in accuracy than another only if the paired 95% CI of the difference excludes 0.
- I'll say it "dominates" only if it is at least as accurate in point estimate and no more costly or slow, and its accuracy is not significantly lower.
- §7 will name as best the policy with the best accuracy among those not dominated. The cost–accuracy trade-off is shown in the Pareto plot. No single score decides.

**Pareto plot.** All fixed policies are plotted on cost (x) against accuracy (y).

- For the Dynamic (DeBERTa p_fail) and Confidence routers, I sweep all (τ_low < τ_high) pairs on the existing grid (0.10–0.90, step 0.05) and plot every point on the eval set, along with each router's frontier.
- The sweep is descriptive only. The reported operating point is still the one tuned on the 50 tuning problems.

**Risk–coverage.**

- Predictors:
  - DeBERTa p_fail;
  - the confidence score, i.e. the negated critique mean log-prob, as in the Confidence router;
  - random ordering (expected curve);
  - an oracle (wrong answers ordered last).
- The risk is that the intrinsic answer R_f is wrong among the accepted problems. Problems are accepted in ascending order of risk score, and ties are broken by problem order.
- AURC is the mean risk over coverages k/150, k = 1..150. I report it with a bootstrap CI over problems, plus the paired difference against random, whose AURC equals the overall error rate.
- A predictor "helps selective prediction" only if its paired difference CI against random excludes 0.

**Sensitivity of the Overall Score (secondary table).**

- Weights (w_c, w_ℓ, w_a) range over the simplex in steps of 0.1 (66 points).
- Human cost is 5×, 15× or 50×; human latency is 15 s, 45 s or 120 s. That makes 594 settings in total.
- In every setting, the learned routers' thresholds are re-tuned on the 50 tuning problems with that setting, as the paper's procedure does. Then all policies are scored on the 150 eval problems.
- I'll report, for each policy, the share of settings in which it has the best Overall Score. I'll also report the share of settings in which Dynamic beats the best fixed policy.
- The human cost and latency are labeled as assumptions throughout.

**Expected outcome (the reviewer's prediction, not mine).** R0 only is at least as accurate as Always Intrinsic (73.3% vs 70.7%) and cheaper. Whatever happens, the result is reported as is.

---

## M5: domain and base-rate confounding in natural-set AUCs (registered 2026-10-31, before any M5 computation)

**Data:** the existing out-of-fold predictions (`scfp_runs/paper_v7/predictions.jsonl`), on the natural set with the rf_wrong target. Folds and models are unchanged. The only new fit is the identity baseline.

**New baseline: identity only.**
- Logistic regression on one-hot source, with the same grouped 5-fold CV, folds, class balancing and inner CV as the other models.
- It is fit per generator. The pooled version is fit on all generators together, with source and generator one-hots.
- By construction, its within-source AUC is 0.5. I report that as a sanity check.

**Metrics for every predictor (rows of Table 14):**
1. **Within-source AUC (headline).** AUC per source, averaged with weights equal to each source's number of traces. For the pooled rows, the strata are source × generator.
2. **Stratified AUC.** The share of correctly ordered (failure, success) pairs, counting only pairs from the same stratum (ties count ½). This is AUC with the comparison restricted within strata, weighted by pairs instead of traces.
3. **Pooled AUC**, as before, for reference only.
4. **AUPRC** with failure as the positive class (average precision), reported next to the failure rate, which is the chance level.
5. **Brier score** of the predicted probabilities as they are. The models are class-balanced, so the probabilities are not calibrated for the base rate. I'll say so; M7 adds calibration.

**Intervals and tests.**
- 95% cluster bootstrap over problems: 1,000 resamples, seed 0. Within one generator there is one trace per problem; pooled, all traces of a resampled problem go in together.
- Comparison with text: the cluster-bootstrap distribution of the difference in within-source AUC.
- A predictor "beats text" only if that 95% CI excludes 0.
- The existing McNemar accuracy tests stay as they are until M7 replaces them.

**Decision rule.**
- The abstract, §8, §9 and §10 will quote within-source AUC ranges in place of 0.710–0.794 and 0.761.
- If the identity baseline's pooled AUC is within 0.05 of a predictor's pooled AUC, I'll state that most of that predictor's pooled AUC is explained by dataset identity.

**Expectation (not a target):** within-source AUC is lower than pooled AUC, more so for generators whose failure rate differs strongly by source.
