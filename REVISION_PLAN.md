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

### M5 addendum (registered 2026-10-31, after the main M5 run and before these computations)

The main M5 run showed that pooled AUC is strongly confounded by source. Two more tables are pooled across sources, so I'm extending the same treatment to them. Metrics, bootstrap and decision rules are as above.

- **(a) Filter effect (Table 13).**
  - Within-source AUC on SCFP v1.0 and on the natural set, per generator, for text and for signals with the verifier.
  - SCFP v1.0 predictions come from the same saved file (`set == "v1"`). The intervals use the cluster bootstrap over problems.
  - The §8.3 claim ("the filter removes the signal for the filter model") stands only if Qwen2.5-0.5B's within-source text AUC on SCFP v1.0 is below its natural-set within-source text AUC, with the cluster-bootstrap CI of the difference excluding 0 (natural minus v1, resampling each set's problems independently). Otherwise the claim is weakened in the text.
- **(b) Sub-tasks (Table 16).**
  - Out-of-fold predictions for r0_wrong, changed, repair and harm are recomputed with the paper_v7 code: GBM on signals with the verifier, same folds. The results.json aggregates are checked to match first.
  - These predictions are now saved per trace.
  - Within-source AUC is reported next to the pooled AUC.
  - The §8.6 claim ("revision behaviour is predictable, R0 correctness is not") stands only if the within-source AUC for changed, repair and harm exceeds that for r0_wrong for every generator.

### M5 post-hoc check (registered 2026-10-31, after seeing the identity baseline; labeled post hoc in the paper)

**Observation.** The identity baseline's within-source AUC is 0.45–0.48, not the 0.5 expected by construction. A likely cause: each fold's model is fit on the other folds, so a fold with more failures has a lower predicted base rate. That anti-correlates the fold intercept with the labels and biases within-source AUC downward for every predictor.

**Check.** Compute AUC within source × fold strata, i.e. within source × fold × generator for the pooled rows. This removes between-fold intercept differences.

**Reporting.** This is a sensitivity analysis only. The headline stays the pre-registered within-source AUC. If the source × fold version changes which predictor beats text, I'll say so.

---

## Decision recorded 2026-10-31 (author)

After M5, the authors chose to keep the filter finding as a contribution, restated as a general caution: difficulty-filtered problems carry less within-source signal for every generator tested, and metrics pooled over datasets can reward recognizing the dataset. It is no longer stated as specific to the filter model.

## M6: isolating the PoT verifier (registered 2026-10-31, before any M6 generation)

**Generation** (Kaggle T4; approved by the author: about 5.4 GPU-h, k = 1 and k = 3).
- For each of the four generators, on its kept natural-set GSM8K traces (3,118 traces), draw 3 answers with the **same initial prompt** as R0.
- Settings: temperature 0.7, at most 512 new tokens (as for R0), the pinned commits of Appendix B, and calls logged with the existing `CallLog` format. Seeds come from the call id, as in the pipeline.
- Precision: all four run in float16. Qwen2.5-1.5B's R0 was generated in bfloat16, so this is noted as a limitation.
- Sample 1 is the cost-matched baseline: one extra generation, like the PoT program. Samples 1–3 give k = 3.
- An unparsable sample counts as disagreeing with R0.

**Features:**
- `pot_ran`, `pot_agree_r0` (existing);
- `sc1_agree`: sample 1's answer equals R0's answer;
- `sc3_agree_frac`: share of the 3 samples equal to R0;
- `sc3_majority_is_r0`.

**Analyses** (per generator, GSM8K traces only, target rf_wrong; also reported for r0_wrong):
1. **Single binary or score features:** `pot_ran`, `pot_agree_r0`, `sc1_agree`, `sc3_agree_frac`.
   - Metrics: AUC (failure score = 1 − agreement); accuracy of "predict failure iff disagreement"; AUPRC.
2. **Models:** class-balanced logistic regression on the signals without the verifier, plus one of {nothing, PoT, SC1, SC1 + PoT, SC3, SC3 + PoT}.
   - Same LR settings as the paper, with 5-fold CV grouped by problem (seed 42), on the GSM8K traces of each generator.
3. **Paired comparisons:** cluster bootstrap over problems (1,000 resamples, seed 0) of the AUC difference for:
   - (+PoT) − (+SC1), the cost-matched comparison;
   - (+SC1+PoT) − (+SC1), what PoT adds beyond two-sample agreement;
   - (+SC3+PoT) − (+SC3).

**Decision rule.**
- The paper will say that PoT adds information beyond two-sample agreement for a generator only if the CI of (+SC1+PoT) − (+SC1) excludes 0.
- It will say that PoT is better than the cost-matched SC1 only if the CI of (+PoT) − (+SC1) excludes 0.
- The number of generators for which each holds is reported. If it is 0, the paper says the verifier adds nothing beyond two-sample agreement.

---

## M7: statistical rigor (registered 2026-10-31, before any M7 computation or training)

**1. Seeds** (author decision: Kaggle T4).
- Each Table 8 encoder (deberta-scfp, deberta-plain, roberta-base, bert-base) and each Table 9 ablation (no-segments, no-attention, inputs P-R0, P-F, R0-F, deberta-xsmall) is retrained with seeds 1, 2, 3 and 4.
- Everything else is the exact command line of `run_mac_overnight.sh`: same splits, budgets, batch 2 × 8 accumulation and frozen word embeddings.
- Precision is float32 (`--no-amp`), as on the M1. The device is CUDA instead of MPS, and each result records its hardware.
- Seed 42 is the existing M1 run.
- Reported: mean ± sd over the 5 seeds of test accuracy, macro-F1 and AUC, with per-seed results in an appendix. Model selection is on validation macro-F1, as before; the test split is evaluated once per run.

**2. Cluster-aware intervals and tests.**
- All intervals become 95% bootstrap intervals over problems: 1,000 resamples, seed 0, all traces of a resampled problem kept together. That means 45 problems and 108 traces for the SCFP test split, and 2,000 problems pooled for the natural set.
- Paired accuracy tests become a **problem-level paired permutation test**: per-problem summed differences in correctness, signs flipped at random, 10,000 permutations, seed 0, two-sided.
- Natural-set AUC differences: the cluster-bootstrap p-value, two-sided, equal to 2 × min(share of differences ≤ 0, share ≥ 0).

**3. Families and Holm–Bonferroni.**
- One family per table:
  - Table 7: each baseline vs TF-IDF.
  - Table 8: each encoder vs TF-IDF, and each encoder vs majority (two families).
  - Table 9: each ablation vs the full model.
  - Table 11: each policy vs no correction.
  - Table 14: each predictor vs text, within each generator row, with the five rows as one family of 25.
- **Pre-specified** in the submitted paper: Table 7 vs TF-IDF and vs majority; Table 8 vs TF-IDF and vs majority; Table 14 vs TF-IDF.
- **Added in revision:** Table 9 vs the full model, and Table 11 vs no correction.
- Holm-adjusted p-values are reported next to the raw ones.

**4. Power.**
- For the SCFP test split (108 traces in 45 problems), the minimum detectable difference at 80% power and two-sided α = 0.05:
  - for paired accuracy, using the observed discordance rate between the full model and TF-IDF, inflated by the design effect 1 + (m̄ − 1)·ICC, with the ICC of correctness within problems estimated from the test predictions;
  - for AUC against 0.5 and for paired AUC, by Hanley–McNeil with the observed class sizes and inter-model correlation, inflated by the same design effect.
- Both are checked by simulation (resampling problems under an injected difference). The assumptions are stated in the appendix.

**5. Calibration.**
- ECE (15 equal-width bins), Brier score and a reliability diagram for:
  - the best natural-set predictor per generator and pooled (out-of-fold);
  - the DeBERTa p_fail on the SCFP test split and on the routing evaluation set.
- The confidence-router score is a rank, not a probability, so it gets no ECE; I'll say so.
- The class-balanced models are not calibrated to the base rate by design, and this is reported as a finding, not fixed.

**Decision rule.** A claimed difference survives only if its Holm-adjusted p < 0.05. Any claim that loses significance after clustering or Holm correction is rewritten.

### M7 implementation note (2026-10-31, after the first run of m7_stats.py)

I raised the bootstrap resamples behind the natural-set AUC-difference p-values from 1,000 to 10,000. The p-value definition is unchanged.

With 1,000 resamples, the smallest attainable p is 2/1001. Holm over 25 comparisons then gives at least 0.05 for every test, so no comparison could ever reach significance. All 25 adjusted values came out at exactly 0.05. This is a resolution fix, made before looking at any adjusted value from the new run.

The confidence intervals still use 1,000 resamples, as registered.

---

## M9: hidden-state probing at the answer tokens (registered 2026-10-31, before any M9 extraction)

**Extraction** (Kaggle T4, about 0.5 GPU-h; within the author's approved budget for the revision).
- For every kept natural-set trace (6,824), run each generator by teacher forcing on (P, R0) in the chat template, with the pinned commit and float16.
- Store, at **every layer**, the residual-stream state averaged over the tokens of R0's answer span (the value on the last "Final answer:" line, or else the final `\boxed{}`). Also store the state at the last token of R0, for comparison with the existing probe.
- Traces whose answer span cannot be located get no answer-token state; their number is reported.

**Probe.**
- Standardization, then L2-regularized logistic regression with no PCA, class-balanced. C is chosen from {1e-4, 1e-3, 1e-2, 1e-1, 1} by 3-fold inner CV.
- 5-fold CV grouped by problem, using the paper's folds. Target: rf_wrong; r0_wrong is also reported.

**Reporting.**
- Within-source AUC per layer, per generator (figure), with the cluster-bootstrap CI at the best layer. The best layer is identified **after the fact** and labeled as such.
- Comparison: the existing last-token PCA-64 probe (the paper's numbers, recomputed within source if the arrays are available; otherwise its pooled value, labeled as pooled).

**Stacking.**
- The layer is chosen **inside each outer training fold** by inner CV over layers, so the test fold never informs the choice.
- That probe's out-of-fold probability is added as a feature to the best M5 predictor (signals + verifier, LR) for each generator.
- Report the within-source AUC change with a cluster-bootstrap CI.

**Decision rule.** §8.4 will say that hidden states add to the signals only if the stacked model's within-source AUC exceeds that of the signals model, with the CI excluding 0. Otherwise it says they do not.

---

## M3: separating initial-answer error from correction failure (registered 2026-10-31, before any M3 computation)

**CPU part** (from the existing features; same folds; within-source AUC, cluster bootstrap as in M5).
- **(a) Δ-label.** helped (R0 wrong → Rf right), neutral (correctness unchanged), harmed (R0 right → Rf wrong).
  - Class-balanced multinomial logistic regression on the signals with the verifier.
  - Reported: macro one-vs-rest within-source AUC, per-class AUC, and macro-F1.
- **(b) Two-stage gate.**
  - Stage 1 predicts P(R0 correct) from pre-revision signals (signals + verifier, LR, out-of-fold).
  - Stage 2 predicts P(harm) and P(repair) from the same signals plus the stage-1 out-of-fold probability, trained on all traces. Labels: harm = R0 right and Rf wrong; repair = R0 wrong and Rf right.
  - The gate's P(final wrong) = (1 − p̂₀)(1 − P̂(repair | R0 wrong)) + p̂₀ · P̂(harm | R0 right). The conditional models condition on the **predicted** p̂₀ as a feature, never on the true R0 correctness.
  - Compared with the direct predictor by within-source AUC.
- **(c) Answer-span mean log-prob of R0 alone** (existing feature `lp_answer_mean`): within-source AUC for rf_wrong and r0_wrong. Score = −log-prob.

**GPU part** (Kaggle T4). All prompts see only P and R0, or P, R0 and F.
1. **P(True)** from each generator. The prompt is: "Question: … Proposed answer: … Is the proposed answer correct? (A) True (B) False. Answer:". The score is the next-token probability of "A" normalized over {A, B}, and failure score = 1 − P(True).
2. **Verbalized confidence** (0–100) from each generator: greedy decoding, at most 8 new tokens, parsed as the first integer. Unparsable answers are missing and imputed in models but excluded from the single-feature AUC (count reported).
3. **Reward model:** `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` in 4-bit NF4 (bitsandbytes). It scores (P as user, R0 as assistant), and failure score = −reward. The exact commit is recorded.
4. **LLM judge:** `Qwen/Qwen2.5-7B-Instruct` in 4-bit NF4. It sees (P, R0, F) and is asked whether the final answer after revision will be wrong. Score = P("Yes") normalized over {Yes, No}. The exact commit is recorded.

**Reporting** (all items).
- Within-source AUC, stratified AUC, AUPRC and Brier, with cluster-bootstrap CIs, per generator and pooled, for rf_wrong and r0_wrong.
- Holm-adjusted tests against text, as in M7.
- The scores are added to Table 14, or to a new table if that is too wide.

**Decision rule.**
- The paper will say a correctness-prediction baseline matches or beats the generator signals only on the CI of the within-source AUC difference.
- §3.2 will be rewritten in any case, as the reviewer requested, to say that the rf_wrong label largely reduces to verifying R0 and that the Δ-label is what decides whether to self-correct.
