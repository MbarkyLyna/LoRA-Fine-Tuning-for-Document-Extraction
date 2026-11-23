#!/usr/bin/env bash
# End-to-end LLM benchmark on a GPU: (1) base few-shot baseline, (2) LoRA fine-tune, (3) evaluate on all test sets.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
NAME=${NAME:-qwen05b}
LIMIT=${LIMIT:-200}
SETS="test_id_n00 test_id_n01 test_id_n02 test_id_n05 test_ood_layout_n02 test_ood_lang_n02"
python -m docextract.synth --out data
# 1) benchmark the *untuned* base model first: LoRA must beat this to justify itself
python -m docextract.llm.infer --model "$MODEL" --name "${NAME}-fewshot2" --shots 2 --testset $SETS --limit "$LIMIT"
# 2) fine-tune
python -m docextract.llm.train_lora --model "$MODEL" --out "runs/${NAME}-lora" --rank 16 --epochs 2
# 3) evaluate adapter
python -m docextract.llm.infer --model "$MODEL" --adapter "runs/${NAME}-lora" --name "${NAME}-lora-r16" --testset $SETS --limit "$LIMIT"
python -m docextract.report
