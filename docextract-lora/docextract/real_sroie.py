"""Real-data check on public SROIE receipts (human-transcribed OCR lines + key fields).

Closes the 'everything is synthetic' gap for the three fields SROIE annotates: company -> vendor,
date, total.  Systems compared on a fixed held-out split:
  rules / rules-ext      hand-written regex (built for synthetic docs; no SROIE tuning)
  crf-synth              CRF trained ONLY on synthetic data  -> measures the synthetic->real gap
  crf-sroie              CRF trained on the SROIE train split with weak token labels
  <any>__sroie_test.jsonl  predictions from other systems (e.g. LoRA LLM) dropped into results/preds_real/

    bash scripts/prepare_sroie.sh && python -m docextract.real_sroie --evaluate
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from rapidfuzz import fuzz

from .baselines import rules
from .baselines.crf import CRFExtractor
from .metrics import _eq_field, anls, bootstrap_ci
from .normalize import norm_text, parse_amount, parse_date
from .synth import load

FIELDS = ["vendor", "date", "total"]


def read_box(path: Path):
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split(",", 8)
        if len(parts) == 9 and parts[8].strip():
            lines.append(parts[8].strip())
    return lines


def weak_labels(lines, key):
    """Token-level labels derived by matching the gold strings back into the text."""
    toks = [l.split() for l in lines]
    labs = [["O"] * len(t) for t in toks]
    comp = norm_text(key["company"])
    best = max(range(min(len(lines), 12)), key=lambda i: fuzz.ratio(norm_text(lines[i]), comp), default=None)
    if best is not None and fuzz.ratio(norm_text(lines[best]), comp) >= 90:
        labs[best] = ["vendor"] * len(toks[best])
    raw_date = key["date"].strip()
    for i, t in enumerate(toks):  # contiguous token window that equals the raw date string
        for a in range(len(t)):
            for b in range(a + 1, min(len(t), a + 4) + 1):
                if " ".join(t[a:b]) == raw_date or "".join(t[a:b]) == raw_date.replace(" ", ""):
                    for k in range(a, b):
                        labs[i][k] = "date"
    tot = parse_amount(re.sub(r"[A-Za-z]", "", key["total"]))
    cands = [(i, j) for i, t in enumerate(toks) for j, w in enumerate(t)
             if tot is not None and re.search(r"\d", w) and parse_amount(re.sub(r"[A-Za-z]", "", w)) == tot]
    if cands:
        pref = [c for c in cands if "total" in lines[c[0]].lower() or (c[0] > 0 and "total" in lines[c[0] - 1].lower())]
        i, j = (pref or cands)[-1]
        labs[i][j] = "total"
    return [[[w, l] for w, l in zip(t, lb)] for t, lb in zip(toks, labs)]


def convert(src: Path, out: Path, seed=0, n_test=126):
    ids = sorted(p.stem for p in (src / "key").glob("*.json"))
    docs = []
    for i in ids:
        key = json.loads((src / "key" / f"{i}.json").read_text(encoding="utf-8", errors="ignore"))
        box = src / "box" / f"{i}.csv"
        if not box.exists() or not all(k in key for k in ("company", "date", "total")):
            continue
        lines = read_box(box)
        gold = {"vendor": key["company"].strip(), "date": parse_date(key["date"].strip()),
                "total": parse_amount(re.sub(r"[A-Za-z]", "", key["total"]))}
        if gold["date"] is None or gold["total"] is None or not lines:
            continue  # date/total we cannot normalise: excluded and counted below
        docs.append({"id": f"sroie-{i}", "text": "\n".join(lines), "lines": weak_labels(lines, key), "gold": gold})
    random.Random(seed).shuffle(docs)
    test, train = docs[:n_test], docs[n_test:]
    out.mkdir(parents=True, exist_ok=True)
    for name, ds in (("sroie_train", train), ("sroie_test", test)):
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in ds) + "\n", encoding="utf-8")
    print(f"converted {len(docs)} of {len(ids)} receipts (dropped {len(ids) - len(docs)} with unparseable date/total); "
          f"train={len(train)} test={len(test)}")


def score_real(preds: dict, docs):
    rows = []
    for d in docs:
        p = (preds.get(d["id"]) or {}).get("pred") or {}
        r = {f: _eq_field(f, p.get(f), d["gold"][f]) for f in FIELDS}
        r["anls_vendor"] = anls(p.get("vendor"), d["gold"]["vendor"])
        r["all3"] = all(r[f] for f in FIELDS)
        rows.append(r)
    out = {"n": len(rows)}
    for f in FIELDS + ["all3"]:
        v = [float(r[f]) for r in rows]
        out[f] = sum(v) / len(v)
        out[f + "_ci95"] = list(bootstrap_ci(v))
    out["anls_vendor"] = sum(r["anls_vendor"] for r in rows) / len(rows)
    out["field_em_macro"] = sum(out[f] for f in FIELDS) / 3
    return out


def evaluate(data="data", results="results", synth_train="data/train.jsonl"):
    from .schema import coerce
    data, results = Path(data), Path(results)
    test, train = load(data / "sroie_test.jsonl"), load(data / "sroie_train.jsonl")
    systems = {
        "rules": lambda t: rules.predict(t),
        "rules-ext": lambda t: rules.predict(t, extended=True),
        "crf-synth": CRFExtractor(0.01, 0.01, 100).fit(load(synth_train)).predict,
        "crf-sroie": CRFExtractor(0.05, 0.05, 100).fit(train).predict,
    }
    scores, preds_dir = {}, results / "preds_real"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in systems.items():
        recs = {}
        for d in test:
            out = fn(d["text"])
            out = out[0] if isinstance(out, tuple) else out
            recs[d["id"]] = {"id": d["id"], "pred": coerce(out)}
        (preds_dir / f"{name}__sroie_test.jsonl").write_text("\n".join(json.dumps(r) for r in recs.values()) + "\n")
    for f in sorted(preds_dir.glob("*__sroie_test.jsonl")):  # includes externally produced (e.g. LLM) predictions
        recs = {json.loads(l)["id"]: json.loads(l) for l in f.read_text().splitlines() if l.strip()}
        docs = [d for d in test if d["id"] in recs]
        scores[f.name.split("__")[0]] = score_real(recs, docs)
    md = [f"# Real-data check: SROIE receipts (held-out n={len(test)})\n",
          "Fields available in SROIE: vendor (company), date, total. Exact match after the same normalisation as the "
          "synthetic benchmark; 95% bootstrap CI in brackets. `all3` = all three fields right.\n",
          "| system | vendor EM | date EM | total EM | all3 | field EM (macro) |", "|---|---|---|---|---|---|"]
    for n, s in scores.items():
        c = lambda k: f"{100 * s[k]:.1f} [{100 * s[k + '_ci95'][0]:.0f}-{100 * s[k + '_ci95'][1]:.0f}]"  # noqa: E731
        md.append(f"| {n} | {c('vendor')} | {c('date')} | {c('total')} | {c('all3')} | {100 * s['field_em_macro']:.1f} |")
    (results / "real_sroie.md").write_text("\n".join(md) + "\n")
    (results / "real_sroie.json").write_text(json.dumps(scores, indent=2))
    readme = Path("README.md")
    if readme.exists():
        block = "<!-- REAL:START -->\n" + "\n".join(md[1:]) + "\n<!-- REAL:END -->"
        readme.write_text(re.sub(r"<!-- REAL:START -->.*?<!-- REAL:END -->", lambda m: block, readme.read_text(), flags=re.S))
    print("\n".join(md))
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/_sroie/data")
    ap.add_argument("--out", default="data")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.evaluate:
        evaluate(a.out)
    else:
        convert(Path(a.src), Path(a.out), a.seed)
