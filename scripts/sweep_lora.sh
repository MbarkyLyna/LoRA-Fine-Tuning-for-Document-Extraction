#!/usr/bin/env bash
# LoRA configuration sweep (rank x target modules) on a train subset; scored on val with the SAME evaluator.
# Run on a GPU (Colab T4 is enough for the 0.5B model). Results land in results/lora_sweep/*.json
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
NTRAIN=${NTRAIN:-1000}
mkdir -p results/lora_sweep runs
for TARGETS in attn all; do
  for RANK in 4 8 16 32; do
    RUN="r${RANK}_${TARGETS}"
    python -m docextract.llm.train_lora --model "$MODEL" --out "runs/$RUN" --rank "$RANK" --targets "$TARGETS" \
        --n-train "$NTRAIN" --epochs 2
    python -m docextract.llm.infer --model "$MODEL" --adapter "runs/$RUN" --name "sweep_$RUN" \
        --testset val --limit 150 --out "runs/$RUN/preds"
    python - <<PY
import json
from docextract.evaluate import read_preds, score
from docextract.metrics import aggregate
from docextract.synth import load
log = json.load(open("runs/$RUN/train_log.json"))
preds = read_preds("runs/$RUN/preds/sweep_${RUN}__val.jsonl")
gold = [d for d in load("data/val.jsonl") if d["id"] in preds]
agg = aggregate(score(preds, gold))
json.dump({"rank": $RANK, "targets": "$TARGETS", "trainable_params": log["trainable_params"],
           "train_time_s": log["train_time_s"], "peak_mem_gb": log["peak_mem_gb"],
           "val_field_em": round(agg["field_em_macro"], 4), "val_item_f1": round(agg["item_f1_strict"], 4)},
          open("results/lora_sweep/$RUN.json", "w"), indent=1)
PY
  done
done
python -m docextract.report
