"""Run the CPU baselines (rules, CRF) with benchmark-driven model selection on *val*.

    python -m docextract.bench            # generates data if missing, writes results/preds/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .baselines import rules
from .baselines.crf import CRFExtractor
from .evaluate import score
from .metrics import aggregate, risk_coverage
from .schema import coerce
from .synth import build_all, load
from .validate import checks, repair

TEST_SETS = ["test_id_n00", "test_id_n01", "test_id_n02", "test_id_n05", "test_ood_layout_n02", "test_ood_lang_n02"]


def write_preds(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(data_dir="data", out_dir="results", seed=1234):
    data, out = Path(data_dir), Path(out_dir)
    if not (data / "train.jsonl").exists():
        build_all(data, seed=seed)
    train, val = load(data / "train.jsonl"), load(data / "val.jsonl")
    log = {}

    # ---- rules: no training; validated on val only to catch bugs -------------------------------
    log["rules_val"] = aggregate(score({d["id"]: {"pred": rules.predict(d["text"])} for d in val}, val))

    # ---- CRF: hyper-parameter grid chosen on val (never on test) --------------------------------
    grid, results = [(0.01, 0.01), (0.05, 0.05), (0.2, 0.2), (1.0, 0.1)], []
    for c1, c2 in grid:
        t = time.perf_counter()
        m = CRFExtractor(c1, c2, 100).fit(train)
        fit_s = time.perf_counter() - t
        preds = {d["id"]: {"pred": m.predict(d["text"])[0]} for d in val}
        a = aggregate(score(preds, val))
        results.append({"c1": c1, "c2": c2, "fit_s": round(fit_s, 1), "val_field_em": a["field_em_macro"],
                        "val_item_f1": a["item_f1_strict"], "val_doc_em": a["doc_em"], "_m": m})
        print(f"CRF c1={c1} c2={c2}: field_em={a['field_em_macro']:.4f} item_f1={a['item_f1_strict']:.4f} fit={fit_s:.0f}s")
    best = max(results, key=lambda r: r["val_field_em"] + r["val_item_f1"])
    crf = best["_m"]
    log["crf_grid"] = [{k: v for k, v in r.items() if k != "_m"} for r in results]
    log["crf_selected"] = {"c1": best["c1"], "c2": best["c2"]}

    # ---- confidence variant (min vs mean marginal) chosen by val AURC on critical fields ---------
    from .metrics import eval_doc
    aurc = {}
    for variant in ("conf_min", "conf_mean"):
        confs, ok = [], []
        for d in val:
            p, meta = crf.predict(d["text"])
            p = coerce(p)
            confs.append(meta[variant] * checks(repair(p)[0])["score"])
            ok.append(eval_doc(repair(p)[0], d["gold"])["critical_ok"])
        aurc[variant] = risk_coverage(confs, ok)[2]
    conf_variant = min(aurc, key=aurc.get)
    log["crf_conf_variant"], log["crf_val_aurc"] = conf_variant, aurc

    # ---- predict all test sets ------------------------------------------------------------------
    for ts in TEST_SETS:
        docs = load(data / f"{ts}.jsonl")
        recs_r, recs_x, recs_c = [], [], []
        for d in docs:
            t = time.perf_counter()
            p = rules.predict(d["text"])
            recs_r.append({"id": d["id"], "pred": p, "confidence": None, "latency_s": time.perf_counter() - t})
            t = time.perf_counter()
            p = rules.predict(d["text"], extended=True)
            recs_x.append({"id": d["id"], "pred": p, "confidence": None, "latency_s": time.perf_counter() - t})
            t = time.perf_counter()
            p, meta = crf.predict(d["text"])
            recs_c.append({"id": d["id"], "pred": p, "confidence": meta[conf_variant],
                           "latency_s": time.perf_counter() - t})
        write_preds(out / "preds" / f"rules__{ts}.jsonl", recs_r)
        write_preds(out / "preds" / f"rules-ext__{ts}.jsonl", recs_x)
        write_preds(out / "preds" / f"crf__{ts}.jsonl", recs_c)
        print("wrote preds for", ts)
    (out / "baseline_selection.json").write_text(json.dumps(log, indent=2))
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    main(a.data, a.out, a.seed)
