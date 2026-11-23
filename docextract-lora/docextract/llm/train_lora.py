"""LoRA / QLoRA fine-tuning for JSON extraction (plain PyTorch loop; no Trainer/TRL dependency drift).

Runs on a free Colab/Kaggle T4 (16 GB) with Qwen2.5-0.5B/1.5B-Instruct. NOT executed in the authoring
sandbox (no GPU, no Hugging Face access) - see docs/LIMITATIONS.md. Everything is logged to
<out>/train_log.json (trainable params, loss curve, wall time, peak GPU memory) for the benchmark report.

    python -m docextract.llm.train_lora --data data --model Qwen/Qwen2.5-0.5B-Instruct --out runs/r16 --rank 16
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from ..synth import load
from .prompt import build_messages, target_json

TARGETS = {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
           "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def encode(tok, doc, max_len):
    prompt = tok.apply_chat_template(build_messages(doc["text"]), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target_json(doc["gold"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = p_ids + t_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + t_ids}  # loss on the answer only


def collate(batch, pad_id):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k], lab[i, :k], att[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return ids, att, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None, help="default 2*rank")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", choices=list(TARGETS), default="all")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-train", type=int, default=None, help="subsample train set (quick sweeps)")
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base weights (needs bitsandbytes)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if dev == "cuda" else torch.float32)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": dtype}
    if a.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if a.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=a.grad_ckpt)
    elif a.grad_ckpt:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha or 2 * a.rank, lora_dropout=a.dropout,
                     target_modules=TARGETS[a.targets], task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, cfg)
    if not a.qlora:
        model.to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.3f}%)")

    train = [e for e in (encode(tok, d, a.max_len) for d in load(Path(a.data) / "train.jsonl")[: a.n_train]) if e]
    val = [e for e in (encode(tok, d, a.max_len) for d in load(Path(a.data) / "val.jsonl")[:100]) if e]
    steps_per_epoch = math.ceil(len(train) / (a.bs * a.grad_accum))
    total = int(steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def val_loss():
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), a.bs):
                ids, att, lab = (x.to(dev) for x in collate(val[i:i + a.bs], tok.pad_token_id))
                with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
                    tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.item(); n += 1
        model.train(); return tot / max(n, 1)

    log = {"model": a.model, "rank": a.rank, "alpha": a.alpha or 2 * a.rank, "targets": a.targets, "qlora": a.qlora,
           "trainable_params": n_tr, "total_params": n_all, "n_train": len(train), "loss": [], "val_loss": []}
    model.train(); t0 = time.time(); step = 0
    while step < total:
        random.shuffle(train)
        for i in range(0, len(train) - a.bs * a.grad_accum + 1, a.bs * a.grad_accum):
            run = 0.0
            for j in range(a.grad_accum):
                ids, att, lab = (x.to(dev) for x in collate(train[i + j * a.bs: i + (j + 1) * a.bs], tok.pad_token_id))
                with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.grad_accum
                scaler.scale(loss).backward(); run += loss.item()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            log["loss"].append(run)
            if step % 25 == 0 or step == total:
                v = val_loss(); log["val_loss"].append([step, v])
                print(f"step {step}/{total} loss {run:.4f} val {v:.4f} ({time.time() - t0:.0f}s)")
            if step >= total:
                break
    log["train_time_s"] = round(time.time() - t0, 1)
    log["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    print("saved adapter to", out)


if __name__ == "__main__":
    main()
