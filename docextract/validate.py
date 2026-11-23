"""Verifier + repair layer: exploit invoice arithmetic as free supervision.

checks()  -> which invariants hold (usable as a confidence signal)
repair()  -> deterministic fixes for missing / inconsistent numeric fields
Both are model-agnostic: they act on predictions, so any system can be wrapped.
"""
from __future__ import annotations

import copy

from .normalize import KNOWN_CURRENCIES
from .schema import EQ_TOL

STD_RATES = [0, 5, 7, 10, 19, 20, 21]


def _eq(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= EQ_TOL


def _sum_items(items):
    a = [i.get("amount") for i in items]
    return None if not a or any(x is None for x in a) else round(sum(a), 3)


def checks(p: dict) -> dict:
    res: dict[str, bool] = {}
    items = p.get("items") or []
    full = [i for i in items if None not in (i.get("qty"), i.get("unit_price"), i.get("amount"))]
    if full:
        res["item_arith"] = all(_eq(round(i["qty"] * i["unit_price"], 3), i["amount"]) for i in full)
    s = _sum_items(items)
    sub, tax, tot = p.get("subtotal"), p.get("tax"), p.get("total")
    if sub is not None and s is not None:
        res["items_sum"] = _eq(s, sub)
    if None not in (sub, tax, tot):
        res["totals"] = _eq(round(sub + tax, 3), tot)
    elif sub is None and None not in (s, tax, tot):
        res["gross"] = _eq(round(s + tax, 3), tot)
    if p.get("date") is not None:
        res["date_valid"] = "2000" <= p["date"][:4] <= "2035"
    if p.get("currency") is not None:
        res["currency_valid"] = p["currency"] in KNOWN_CURRENCIES
    res["required_present"] = all(p.get(k) is not None for k in ("vendor", "date", "total", "currency"))
    score = sum(res.values()) / len(res) if res else 0.0
    return {"detail": res, "score": score, "consistent": bool(res) and all(res.values())}


def _plausible_rate(tax, sub) -> bool:
    if tax is None or not sub:
        return False
    r = 100.0 * tax / sub
    return any(abs(r - s) < 0.6 for s in STD_RATES)


def repair(pred: dict | None):
    """Return (repaired_pred, actions). Never invents text fields; only numeric fields."""
    if pred is None:
        return None, []
    p = copy.deepcopy(pred)
    acts: list[str] = []
    items = p.get("items") or []
    # 1) per-item: fill the single missing factor
    for k, it in enumerate(items):
        q, u, a = it.get("qty"), it.get("unit_price"), it.get("amount")
        miss = [x is None for x in (q, u, a)]
        if sum(miss) == 1:
            if a is None:
                it["amount"] = round(q * u, 3)
            elif u is None and q:
                it["unit_price"] = round(a / q, 3)
            elif q is None and u:
                it["qty"] = float(round(a / u))
            else:
                continue
            acts.append(f"item{k}:fill")
    # 2) per-item inconsistency: use subtotal as the arbiter between candidate fixes
    sub = p.get("subtotal")
    if sub is not None:
        for k, it in enumerate(items):
            q, u, a = it.get("qty"), it.get("unit_price"), it.get("amount")
            if None in (q, u, a) or _eq(round(q * u, 3), a):
                continue
            others = [i.get("amount") for j, i in enumerate(items) if j != k]
            if any(x is None for x in others):
                continue
            base = sum(others)
            cands = {"amount": (round(q * u, 3), u, q)}
            if q:
                cands["unit"] = (a, round(a / q, 3), q)
            if u and abs(a / u - round(a / u)) < 1e-6 and round(a / u) >= 1:
                cands["qty"] = (a, u, float(round(a / u)))
            good = [name for name, (na, _, _) in cands.items() if _eq(round(base + na, 3), sub)]
            if len(good) == 1:
                na, nu, nq = cands[good[0]]
                it["amount"], it["unit_price"], it["qty"] = na, nu, nq
                acts.append(f"item{k}:fix_{good[0]}")
    # 3) fill missing top-level numbers
    sub, tax, tot = p.get("subtotal"), p.get("tax"), p.get("total")
    s = _sum_items(items)
    if tot is None and None not in (sub, tax):
        p["total"] = tot = round(sub + tax, 3); acts.append("total:fill")
    elif tax is None and None not in (sub, tot) and tot >= sub:
        p["tax"] = tax = round(tot - sub, 3); acts.append("tax:fill")
    elif sub is None and None not in (tax, tot) and s is not None and _eq(round(s + tax, 3), tot):
        pass  # receipt without subtotal line, already consistent: leave null (gold is null)
    # 4) inconsistent totals
    sub, tax, tot = p.get("subtotal"), p.get("tax"), p.get("total")
    if None not in (sub, tax, tot) and not _eq(round(sub + tax, 3), tot):
        items_ok = s is not None and _eq(s, sub)
        if items_ok:  # subtotal trusted -> decide between tax and total using rate plausibility
            if _plausible_rate(tax, sub):
                p["total"] = round(sub + tax, 3); acts.append("total:recompute")
            elif _plausible_rate(tot - sub, sub):
                p["tax"] = round(tot - sub, 3); acts.append("tax:recompute")
            else:
                p["total"] = round(sub + tax, 3); acts.append("total:recompute_default")
        elif s is not None and _eq(round(s + tax, 3), tot):
            p["subtotal"] = s; acts.append("subtotal:from_items")
    return p, acts
