"""Score a predictions JSONL against a gold JSONL.  python -m docextract.evaluate --pred P --gold G"""
from __future__ import annotations

import argparse
import json

from .metrics import aggregate, eval_doc
from .schema import coerce
from .synth import load
from .validate import repair


def read_preds(path):
    return {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8"))}


def score(preds: dict, gold_docs: list, use_repair=False):
    rows = []
    for d in gold_docs:
        rec = preds.get(d["id"])
        p = coerce(rec["pred"]) if rec and rec.get("pred") is not None else None
        if use_repair:
            p, _ = repair(p)
        rows.append(eval_doc(p, d["gold"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--repair", action="store_true")
    a = ap.parse_args()
    gold = load(a.gold)
    preds = read_preds(a.pred)
    gold = [g for g in gold if g["id"] in preds]  # allow --limit runs
    print(json.dumps(aggregate(score(preds, gold, a.repair)), indent=2))
