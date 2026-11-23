"""Prompt construction + robust JSON extraction. Pure python (no torch) so it is unit-tested in CI."""
from __future__ import annotations

import json
import re

from ..schema import SCHEMA_TEXT

SYSTEM = (
    "You extract structured data from noisy OCR text of invoices and receipts. "
    "Reply with ONLY one JSON object matching this schema: " + SCHEMA_TEXT + ". "
    "Dates must be ISO YYYY-MM-DD (numeric dates are day-first). Amounts are plain numbers using '.' as decimal mark. "
    "Currency is an ISO code. Use null for fields that are absent; never invent values."
)


def build_messages(text: str, shots: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots or []:  # few-shot examples (used for the zero/few-shot *base model* benchmark)
        msgs += [{"role": "user", "content": f"OCR text:\n{ex['text']}"},
                 {"role": "assistant", "content": target_json(ex["gold"])}]
    msgs.append({"role": "user", "content": f"OCR text:\n{text}"})
    return msgs


def target_json(gold: dict) -> str:
    order = ["vendor", "invoice_number", "date", "currency", "subtotal", "tax", "total", "items"]
    g = {k: gold.get(k) for k in order}
    g["items"] = [{k: it[k] for k in ("description", "qty", "unit_price", "amount")} for it in gold["items"]]
    return json.dumps(g, ensure_ascii=False, separators=(",", ":"))


def extract_json(raw: str):
    """Pull the first balanced {...} from model output and parse it; tolerate fences/trailing commas."""
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = s[start:i + 1]
                for cand in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                    try:
                        return json.loads(cand)
                    except json.JSONDecodeError:
                        pass
                return None
    return None  # truncated generation
