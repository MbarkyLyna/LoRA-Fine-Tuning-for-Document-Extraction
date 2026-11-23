"""Target schema, JSON coercion and shared constants."""
from __future__ import annotations

from .normalize import parse_amount, parse_currency, parse_date

SCALARS = ["vendor", "invoice_number", "date", "currency", "subtotal", "tax", "total"]
MONEY = {"subtotal", "tax", "total"}
ITEM_FIELDS = ["description", "qty", "unit_price", "amount"]
NULLABLE = {"invoice_number", "subtotal"}  # legitimately absent on some layouts
EQ_TOL = 0.0006  # amounts compared at 3-decimal resolution (TND)

SCHEMA_TEXT = (
    '{"vendor": string, "invoice_number": string|null, "date": "YYYY-MM-DD", '
    '"currency": "EUR"|"USD"|"TND", "subtotal": number|null, "tax": number, "total": number, '
    '"items": [{"description": string, "qty": number, "unit_price": number, "amount": number}]}'
)


def coerce(pred) -> dict | None:
    """Turn a raw model dict into a clean, typed prediction (None if not a dict)."""
    if not isinstance(pred, dict):
        return None
    out: dict = {}
    out["vendor"] = None if pred.get("vendor") in (None, "") else str(pred["vendor"]).strip()
    inv = pred.get("invoice_number")
    out["invoice_number"] = None if inv in (None, "", "null") else str(inv).strip()
    out["date"] = parse_date(pred.get("date"))
    out["currency"] = parse_currency(pred.get("currency"))
    for k in ("subtotal", "tax", "total"):
        out[k] = parse_amount(pred.get(k))
    items = []
    raw_items = pred.get("items")
    for it in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(it, dict):
            continue
        items.append({
            "description": None if it.get("description") is None else str(it["description"]).strip(),
            "qty": parse_amount(it.get("qty")),
            "unit_price": parse_amount(it.get("unit_price")),
            "amount": parse_amount(it.get("amount")),
        })
    out["items"] = items
    return out
