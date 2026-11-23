"""Greedy JSON extraction with a base model (zero/few-shot) or base+LoRA adapter.
Writes results/preds/<name>__<testset>.jsonl in the format the report expects, including a
sequence-level confidence (exp of mean token log-prob) for selective prediction.

    python -m docextract.llm.infer --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/r16 \
        --name qwen05b-lora-r16 --testset test_id_n02 --limit 200
    python -m docextract.llm.infer --model Qwen/Qwen2.5-0.5B-Instruct --name qwen05b-fewshot2 --shots 2 ...
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from ..schema import coerce
from ..synth import load
from .prompt import build_messages, extract_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True, help="system name used in report, e.g. qwen05b-lora-r16")
    ap.add_argument("--data", default="data")
    ap.add_argument("--testset", nargs="+", required=True)
    ap.add_argument("--out", default="results/preds")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples from train (base-model baseline)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if dev == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()  # no adapter overhead at inference
    model.eval()
    shots = load(Path(a.data) / "train.jsonl")[: a.shots] if a.shots else None

    for ts in a.testset:
        docs = load(Path(a.data) / f"{ts}.jsonl")[: a.limit]
        recs = []
        for i in range(0, len(docs), a.bs):
            batch = docs[i:i + a.bs]
            prompts = [tok.apply_chat_template(build_messages(d["text"], shots), tokenize=False, add_generation_prompt=True)
                       for d in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
            dt = (time.perf_counter() - t0) / len(batch)
            gen = out.sequences[:, enc["input_ids"].shape[1]:]
            lp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            for d, g, l in zip(batch, gen, lp):
                mask = g != tok.pad_token_id
                raw = tok.decode(g, skip_special_tokens=True)
                conf = math.exp(float(l[mask].mean())) if mask.any() else 0.0
                recs.append({"id": d["id"], "pred": coerce(extract_json(raw)), "confidence": conf,
                             "latency_s": dt, "raw": raw})
            print(f"{ts}: {min(i + a.bs, len(docs))}/{len(docs)}", flush=True)
        path = Path(a.out) / f"{a.name}__{ts}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
