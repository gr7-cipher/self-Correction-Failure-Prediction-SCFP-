#!/usr/bin/env bash
# SCFP larger-model experiments on one GPU (Kaggle T4/P100 or Colab T4, 16 GB).
#
#   bash run_gpu.sh            # full run, about 5-7 hours on a T4
#   QUICK=1 bash run_gpu.sh    # 15-20 minute smoke test on CPU (tiny model, few examples)
#
# Re-running skips every fold or run that already finished, so it resumes after a
# session ends: copy the scfp_runs/ folder back in and run the same command.
# Everything is written under benchmark/scfp_runs/scfp_v1/ and zipped into
# scfp_gpu_results.zip at the end.
set -uo pipefail
cd "$(dirname "$0")/benchmark"
export PYTHONPATH=.
CFG=configs/gpu.yaml
RD=scfp_runs/scfp_v1
PY=${PY:-python}

if [ -n "${QUICK:-}" ]; then
  echo ">>> QUICK smoke test: tiny model, 30 examples per split, 1 epoch, CPU. Numbers are meaningless."
  LARGE=microsoft/deberta-v3-xsmall; RLARGE=microsoft/deberta-v3-xsmall; SMALL=microsoft/deberta-v3-xsmall
  COMMON="--max-len 128 --batch-size 4 --grad-accum 1 --epochs-binary 1 --limit 30 --device cpu"
  RD_QUICK=scfp_runs/quick_test; rm -rf $RD_QUICK; mkdir -p $RD_QUICK; cp -R $RD/splits $RD/features $RD_QUICK/
  sed "s#run_dir: scfp_runs/scfp_v1#run_dir: $RD_QUICK#" configs/gpu.yaml > configs/quick.yaml
  CFG=configs/quick.yaml; RD=$RD_QUICK; SEEDS="42"
else
  LARGE=microsoft/deberta-v3-large; RLARGE=FacebookAI/roberta-large; SMALL=microsoft/deberta-v3-small
  # effective batch 16; fp16 autocast and gradient checkpointing are on by default on CUDA
  COMMON="--max-len 768 --batch-size 4 --grad-accum 4"
  SEEDS="42 43 44"
  nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "No GPU found. Use QUICK=1 for a CPU smoke test."; exit 1; }
fi
$PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

echo "=== 1. Cross-validated CPU baselines (majority, random, length, confidence, TF-IDF)"
$PY -m scfp_bench.experiments.cv tfidf --config $CFG > /dev/null

echo "=== 2. Cross-validated encoders (5 folds each, all 716 traces predicted once)"
cvrun() { $PY -m scfp_bench.experiments.cv run --config $CFG --name "$1" -- ${@:2} $COMMON; }
cvrun deberta-large-scfp   --arch scfp  --model $LARGE
cvrun deberta-large-plain  --arch plain --model $LARGE
cvrun roberta-large        --arch plain --model $RLARGE        # capped at its 512-token limit
cvrun deberta-small-scfp   --arch scfp  --model $SMALL         # same model as the paper, on GPU

echo "=== 3. Paper's fixed split (108 test traces), DeBERTa-v3-large, 3 seeds"
for s in $SEEDS; do
  n=fixed-deberta-large-scfp-seed$s
  if [ -f $RD/results/$n.json ]; then echo "$n: done, skipping"; continue; fi
  $PY -m scfp_bench.experiments.train_encoder --config $CFG --name $n --arch scfp --model $LARGE --seed $s $COMMON
done

echo "=== 4. Summary"
$PY -m scfp_bench.experiments.cv summary --config $CFG
$PY - <<EOF
import json, glob, os
rows = []
for f in sorted(glob.glob("$RD/results/fixed-*.json")):
    r = json.load(open(f)); t = r["test"]
    rows.append(f"| {os.path.basename(f)[:-5]} | {100*t['accuracy']:.1f}% [{100*t['ci95']['accuracy'][0]:.1f}, {100*t['ci95']['accuracy'][1]:.1f}] | {t['macro_f1']:.3f} | {t.get('auc', float('nan')):.3f} | {r.get('gpu')} | {r['wall_clock_s']/60:.0f} min |")
if rows:
    txt = "| Run | Accuracy [95% CI] | Macro F1 | AUC | GPU | Time |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
    open("$RD/results/fixed_split_summary.md", "w").write("# Fixed split (108 test traces)\n\n" + txt + "\n")
    print(txt)
EOF

cd ..
OUT=scfp_gpu_results.zip
[ -n "${QUICK:-}" ] && OUT=scfp_quick_test_results.zip
rm -f $OUT
zip -qr $OUT benchmark/$RD -x "*.pt"
echo ">>> Done. Send back: $(pwd)/$OUT"
