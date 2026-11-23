"""Hand-written regex baseline (EN/FR/DE keyword lists). Deliberately *not* tuned on OOD
layouts/languages: that brittleness is exactly what the benchmark should expose."""
from __future__ import annotations

import re

from ..normalize import parse_amount, parse_currency, parse_date

NUM = r"\d{1,3}(?:[ .,]\d{3})+[.,]\d{2,3}|\d+(?:[.,]\d+)?"
RE_NUM = re.compile(NUM)
RE_SUB = re.compile(r"sub-?total|sous-?total|total ht|zwischensumme|netto", re.I)
RE_TAX = re.compile(r"\b(vat|tax|tva|mwst|ust)\b", re.I)
RE_TOTAL = re.compile(r"total|gesamt|summe|amount due", re.I)
RE_DUE = re.compile(r"due|échéance|echeance|fällig|zahlungsziel|vencimiento", re.I)
RE_SKIP = re.compile(r"^\s*(tel\b|iban\b)", re.I)
RE_INV = re.compile(r"(?:invoice\s*(?:no\.?|#)|inv\.?\s*no\.?|facture\s*n°|n°\s*facture|rechnung\s*nr\.?|rechnungsnummer)\s*:?\s*(\S+)", re.I)
RE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{4}|\d{1,2} [A-Za-z]{3} \d{4}|[A-Z][a-z]+ \d{1,2}, \d{4}")
# --- extended (v2) patterns: written AFTER inspecting the held-out layout/language, so v2 scores on the
# OOD sets are NOT a held-out measurement (they show the cost of manual rule maintenance). v1 stays frozen.
RE_INV_X = re.compile(r"(?:invoice\s*(?:no\.?|#)|inv\.?\s*no\.?|facture\s*n°|n°\s*facture|rechnung\s*nr\.?|rechnungsnummer"
                      r"|factura\s*n[º°o]\.?|n[º°]\s*de\s*factura|bill\s*#)\s*:?\s*(\S+)", re.I)
RE_SUB_X = re.compile(RE_SUB.pattern + r"|base imponible|\bnet\b(?!\s*total)", re.I)
RE_TAX_X = re.compile(r"\b(vat|tax|tva|mwst|ust|iva)\b", re.I)
RE_ITEM_X = re.compile(r"^\s*(\d{1,2})\s*x\s+(.+?)\s*@\s*(" + NUM + r")\s*=\s*(" + NUM + r")\s*$")
RE_DATE_X = re.compile(RE_DATE.pattern + r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}[ -][A-Za-z]{3,9}[ -]\d{2,4}|\d{4}/\d{2}/\d{2}")
RE_ONLY_AMOUNT = re.compile(r"^\s*(?:[A-Za-z$€]{0,3}\s*)?(" + NUM + r")\s*(?:[A-Za-z$€]{0,3})\s*$")
RE_TOTAL_SKIP = re.compile(r"qty|quantity|items?\b", re.I)
RE_TOTAL_PREF = re.compile(r"incl|round|after|grand|net total|amount due|importe", re.I)
RE_TOTAL_LOW = re.compile(r"excl", re.I)
RE_REG = re.compile(r"\s*\([^()]*\d[^()]*\)\s*$")  # trailing company registration no., e.g. (139386 X)
RE_CUR = re.compile(r"€|\$|\bEUR\b|\bUSD\b|\bTND\b|\bDT\b")


def _last_amount(line: str):
    line = re.sub(r"\(?\d+(?:[.,]\d+)?\s*%\)?", " ", line)  # drop tax rates like (19%)
    nums = RE_NUM.findall(line)
    return parse_amount(nums[-1]) if nums else None


def _item(line: str):
    nums = list(RE_NUM.finditer(line))
    if len(nums) < 3:
        return None
    q, u, a = nums[-3], nums[-2], nums[-1]
    if not re.fullmatch(r"\d{1,2}", q.group()):
        return None
    desc = line[: q.start()].replace("|", " ").replace(" x ", " ").strip()
    return {"description": desc, "qty": parse_amount(q.group()), "unit_price": parse_amount(u.group()),
            "amount": parse_amount(a.group())}


def predict(text: str, extended: bool = False) -> dict:
    re_inv, re_sub, re_tax = (RE_INV_X, RE_SUB_X, RE_TAX_X) if extended else (RE_INV, RE_SUB, RE_TAX)
    lines = [l for l in text.split("\n") if l.strip()]
    body = [l for l in lines if not re.fullmatch(r"[A-ZÄÖÜ]{4,}", l.strip())]  # skip 'INVOICE'-style titles
    out = {"vendor": (RE_REG.sub("", body[0].strip()) if extended else body[0].strip()) if body else None, "invoice_number": None, "date": None, "currency": None,
           "subtotal": None, "tax": None, "total": None, "items": []}
    best_total = (-1, -1)
    re_date = RE_DATE_X if extended else RE_DATE
    for idx, l in enumerate(lines):
        nxt = None
        if extended and idx + 1 < len(lines):  # real receipts often print the value on the NEXT line/box
            mm = RE_ONLY_AMOUNT.match(lines[idx + 1])
            nxt = parse_amount(mm.group(1)) if mm else None
        m = re_inv.search(l)
        if m and out["invoice_number"] is None:
            out["invoice_number"] = m.group(1).rstrip("|")
        if out["date"] is None and not RE_DUE.search(l):
            d = re_date.search(l)
            if d:
                out["date"] = parse_date(d.group())
        if RE_SKIP.search(l) or (RE_DUE.search(l) and not RE_TOTAL.search(l)):
            continue
        la = _last_amount(l)
        if la is None and extended:
            la = nxt
        if re_sub.search(l):
            out["subtotal"] = la
        elif re_tax.search(l):
            out["tax"] = la
        elif RE_TOTAL.search(l):
            if extended:
                if RE_TOTAL_SKIP.search(l) or la is None:
                    continue
                prio = 0 if RE_TOTAL_LOW.search(l) else (2 if RE_TOTAL_PREF.search(l) else 1)
                if (prio, idx) < best_total:
                    continue
                best_total = (prio, idx)
            out["total"] = la
            c = RE_CUR.search(l)
            out["currency"] = parse_currency(c.group()) if c else None
        else:
            mx = RE_ITEM_X.match(l) if extended else None
            it = ({"description": mx.group(2), "qty": parse_amount(mx.group(1)), "unit_price": parse_amount(mx.group(3)),
                   "amount": parse_amount(mx.group(4))} if mx else _item(l))
            if it:
                out["items"].append(it)
    return out
