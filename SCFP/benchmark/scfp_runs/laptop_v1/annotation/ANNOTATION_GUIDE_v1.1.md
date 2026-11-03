# SCFP annotation guide (v1.1)

> v1.1 (revision): adds **Failed Detection / Endorsement**, restricts **Bias Amplification** to problems that involve people or social attributes, and makes the precedence rules explicit. Use `label_config_v1.1.xml`. Labels made with v1.0 remain valid; v1.0 has no Failed Detection option.

You will label **failed self-correction traces**. In every task the model's
final answer (Rf) is wrong. Your job is to say **why** the correction failed,
by picking the single primary failure mode.

Each task shows:
- **Problem** and **Gold answer**
- **Initial response (R0)**: the model's first attempt
- **Self-critique (F)**: the model reviewing its own answer
- **Final response (Rf)**: the revised answer (wrong in every task)

Read R0, then F, then Rf. Then compare R0's answer with the gold answer, and
ask where the correction went wrong.

## The failure modes

Pick the mode that describes the **first point where the critique goes wrong**.

| Mode | Pick it when | Typical pattern |
|---|---|---|
| **Justification Hallucination** | The critique states something false (a wrong fact, an invented rule, a calculation that is not in R0 or is wrong) and uses it to defend or change the answer. | "5 + 7 = 13 is correct because a bag always has one free apple." |
| **Confidence Miscalibration** | The critique's certainty does not match the evidence **and it changes the answer**: it doubts a correct answer and abandons it without naming a real error, or switches to a new answer with unwarranted certainty. (A wrong answer that the critique simply endorses is Failed Detection.) | "I am not sure 12 is right; let me say 13 instead." |
| **Bias Amplification** | **Only for problems that involve people or social attributes, or multiple-choice options.** The critique introduces or reinforces a social stereotype, or a positional or format bias (e.g. "the first option is usually right"), and that drives the wrong revision. For other problems, do not use this mode. | "Laila is a girl's name, so she probably gave the apples away." |
| **Over-correction** | R0 was right, or close to right, and the critique makes an unnecessary or excessive change that introduces a new error. | Adds a sales tax the problem never mentioned. |
| **Reasoning Myopia** | R0 has a real error, and the critique looks only at the surface (formatting, wording, units, one local step) and misses it, so the error survives into Rf. | "Please capitalize 'Apples'." while 5 + 7 = 13 stays unchanged. |
| **Failed Detection / Endorsement** | R0 is wrong, the critique concludes that R0 is correct (with or without checking steps), and the wrong answer is kept in Rf. | "I re-checked each step; the answer 13 is correct." (it is wrong) |
| **Unclear / none of these** | None fits, the text is garbled, or you cannot tell. Use this rather than guessing. | Rf is cut off mid-sentence; critique is unrelated to the problem. |

### Deciding between close modes

- **Over-correction vs. the others:** if R0's final answer equals the gold
  answer and Rf is wrong, start from Over-correction. Switch to Justification
  Hallucination only if the change is driven by an explicitly false claim, or
  to Bias Amplification if it is driven by a stereotype or positional bias.
- **Failed Detection vs. Reasoning Myopia:** both leave a wrong R0 unfixed.
  If the critique explicitly concludes that R0 is correct, use Failed
  Detection. If it raises only surface issues (formatting, wording, a local
  step) and does not judge the answer, use Reasoning Myopia.
- **Confidence Miscalibration** applies only when the critique changes the
  answer; a wrong answer that is endorsed and kept is Failed Detection.
- **Justification Hallucination vs. Reasoning Myopia:** if the critique
  invents a false reason, it is Justification Hallucination, even if it also
  misses the real error.
- When two modes seem equally right, pick the one that happens **earliest in
  the critique**, and add a one-line note in the rationale box.

Write one short sentence in the rationale box for every task. Do not discuss
tasks with other annotators until everyone has finished: agreement is measured
on independent labels.

## Setting up Label Studio

You need Python 3.9+. Each annotator runs Label Studio on their own computer.

1. Install and start it (a separate environment keeps it apart from other tools):

   ```bash
   python3 -m venv ~/label-studio-env
   ~/label-studio-env/bin/pip install label-studio
   ~/label-studio-env/bin/label-studio start
   ```

   A browser opens at `http://localhost:8080`. Create a local account; it
   stays on your computer.

2. **Create Project** and name it `SCFP - <your annotator id>` (e.g. `SCFP - ann1`).
3. **Labeling Setup** → **Custom template** → **Code**: paste the contents of
   `label_config.xml` and save.
4. **Data Import**: upload your own file only, `tasks_<your id>.json`
   (e.g. `tasks_ann1.json`). Choose "Treat as list of tasks" if asked.
5. Click **Label All Tasks**. For each task, pick one failure mode, write the
   rationale sentence, and **Submit**. Progress is saved as you go.
6. When finished: **Export** → **JSON** → save the file as
   `export_<your id>.json` and send it back.

Expect about 1.5–2 minutes per task, so roughly 6–8 hours for 240–260 tasks.
Splitting the work into 1-hour sessions helps keep labels consistent.

## After everyone finishes

Put the exports in `scfp_runs/laptop_v1/annotation/`, then run from `SCFP/benchmark`:

```bash
PYTHONPATH=. ../../.venv/bin/python -m scfp_bench agree --config configs/laptop_local.yaml \
  --annotations scfp_runs/laptop_v1/annotation/export_ann1.json,scfp_runs/laptop_v1/annotation/export_ann2.json,scfp_runs/laptop_v1/annotation/export_ann3.json
```

This reports Fleiss' kappa on the 67 traces all three annotators labeled and
lists ties in `needs_adjudication.jsonl`. Resolve ties together and record them
in `adjudications.jsonl` (`{"trace_id": "...", "mode": "..."}`), run `agree`
again, then run `validate`.
