"""Evaluation: field EM, ANLS, line-item F1 (Hungarian matching), hallucination, CIs."""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .normalize import norm_id, norm_text
from .schema import EQ_TOL, MONEY, NULLABLE, SCALARS
from .validate import checks


def _eq_field(name, p, g) -> bool:
    if p is None or g is None:
        return p is None and g is None
    if name in MONEY:
        return abs(p - g) <= EQ_TOL
    if name == "invoice_number":
        return norm_id(p) == norm_id(g)
    if name == "vendor":
        return norm_text(p) == norm_text(g)
    return p == g


def anls(p, g, tau=0.5) -> float:
    if p is None or g is None:
        return float(p is None and g is None)
    a, b = norm_text(p), norm_text(g)
    d = Levenshtein.normalized_distance(a, b)
    return 1.0 - d if d < tau else 0.0


def _item_match(pi, gi, strict):
    if pi.get("amount") is None or gi["amount"] is None or abs(pi["amount"] - gi["amount"]) > EQ_TOL:
        return 0.0
    if fuzz.ratio(norm_text(pi.get("description") or ""), norm_text(gi["description"])) < 80:
        return 0.0
    if strict:
        for k in ("qty", "unit_price"):
            if pi.get(k) is None or abs(pi[k] - gi[k]) > EQ_TOL:
                return 0.0
    return 1.0


def item_tp(pred_items, gold_items, strict) -> int:
    if not pred_items or not gold_items:
        return 0
    m = np.array([[_item_match(p, g, strict) for g in gold_items] for p in pred_items])
    r, c = linear_sum_assignment(-m)
    return int(m[r, c].sum())


def eval_doc(pred, gold) -> dict:
    valid = pred is not None
    row = {"valid": valid}
    for f in SCALARS:
        row[f] = _eq_field(f, pred.get(f) if valid else None, gold[f]) if valid else False
    row["anls_vendor"] = anls(pred.get("vendor"), gold["vendor"]) if valid else 0.0
    row["anls_invoice"] = anls(pred.get("invoice_number"), gold["invoice_number"]) if valid else 0.0
    pi = pred.get("items", []) if valid else []
    row["n_pred"], row["n_gold"] = len(pi), len(gold["items"])
    row["tp_loose"] = item_tp(pi, gold["items"], strict=False)
    row["tp_strict"] = item_tp(pi, gold["items"], strict=True)
    row["hallu"] = int(valid and any(f in NULLABLE and gold[f] is None and pred.get(f) is not None for f in NULLABLE))
    row["n_null_gold"] = sum(gold[f] is None for f in NULLABLE)
    row["n_hallu"] = sum(valid and gold[f] is None and pred.get(f) is not None for f in NULLABLE)
    row["items_ok"] = row["tp_strict"] == row["n_gold"] == row["n_pred"]
    row["doc_em"] = bool(all(row[f] for f in SCALARS) and row["items_ok"])
    row["critical_ok"] = bool(valid and all(row[f] for f in ("vendor", "date", "currency", "total")))
    row["money_ok"] = bool(valid and all(row[f] for f in ("subtotal", "tax", "total")))
    row["consistent"] = bool(valid and checks(pred)["consistent"])
    return row


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    s = {"n": n, "doc_em": float(np.mean([r["doc_em"] for r in rows]))}
    for f in SCALARS:
        s[f"em_{f}"] = float(np.mean([r[f] for r in rows]))
    s["field_em_macro"] = float(np.mean([s[f"em_{f}"] for f in SCALARS]))
    tp_s, tp_l = sum(r["tp_strict"] for r in rows), sum(r["tp_loose"] for r in rows)
    npred, ngold = sum(r["n_pred"] for r in rows), sum(r["n_gold"] for r in rows)
    s["item_f1_strict"] = prf(tp_s, npred, ngold)[2]
    s["item_f1_loose"] = prf(tp_l, npred, ngold)[2]
    s["anls_vendor"] = float(np.mean([r["anls_vendor"] for r in rows]))
    s["anls_invoice"] = float(np.mean([r["anls_invoice"] for r in rows]))
    nn = sum(r["n_null_gold"] for r in rows)
    s["hallucination_rate"] = (sum(r["n_hallu"] for r in rows) / nn) if nn else 0.0
    s["json_valid"] = float(np.mean([r["valid"] for r in rows]))
    s["arith_consistent"] = float(np.mean([r["consistent"] for r in rows]))
    lo, hi = bootstrap_ci([float(r["doc_em"]) for r in rows])
    s["doc_em_ci95"] = [lo, hi]
    return s


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def risk_coverage(conf, correct):
    """Selective prediction: accept docs with conf >= t. Returns (coverage, risk, aurc)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    cov, risk, prev, aurc = [], [], 0.0, 0.0
    for t in np.unique(conf)[::-1]:
        acc = conf >= t
        c, r = acc.sum() / n, 1.0 - correct[acc].mean()
        cov.append(c)
        risk.append(r)
        aurc += r * (c - prev)
        prev = c
    return np.array(cov), np.array(risk), float(aurc)


def coverage_at_risk(cov, risk, target):
    ok = [c for c, r in zip(cov, risk) if r <= target]
    return float(max(ok)) if ok else 0.0
