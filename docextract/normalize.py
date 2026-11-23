"""Shared normalisation used identically for every system (fair comparison)."""
from __future__ import annotations

import datetime as dt
import re
import unicodedata

CURRENCY_MAP = {"€": "EUR", "EUR": "EUR", "$": "USD", "USD": "USD", "DT": "TND", "TND": "TND"}
KNOWN_CURRENCIES = {"EUR", "USD", "TND"}
_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
                 "%d/%m/%y", "%d-%m-%y", "%d.%m.%y", "%d-%b-%Y", "%d-%b-%y", "%d %B %Y", "%d %b %y", "%Y/%m/%d"]
# characters OCR commonly confuses with digits
_OCR_DIGIT = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2"})


def _digit_ratio(s: str) -> float:
    s = s.replace(" ", "")
    return sum(c.isdigit() for c in s) / len(s) if s else 0.0


def parse_amount(s) -> float | None:
    """Parse '1,234.56' / '1.234,56' / '1 234,56' / '12,345' (TND) into a float.

    Light OCR-confusable repair (O->0, l->1, ...) is applied only when the string is
    mostly digits. Rule: a single separator followed by 1-3 digits is a decimal mark
    (our documents always print decimals, so thousands-only strings do not occur).
    """
    if s is None or isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        return round(float(s), 3)
    t = str(s).replace("\u00a0", " ").strip()
    core = re.sub(r"[€$]|EUR|USD|TND|DT", "", t)
    if _digit_ratio(re.sub(r"[^\w]", "", core)) >= 0.5:
        core = core.translate(_OCR_DIGIT)
    t = re.sub(r"[^\d,.]", "", core)
    if not re.search(r"\d", t):
        return None
    has_c, has_d = "," in t, "." in t
    dec = None
    if has_c and has_d:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
    elif has_c or has_d:
        sep = "," if has_c else "."
        parts = t.split(sep)
        if len(parts) == 2 and 1 <= len(parts[1]) <= 3:
            dec = sep
    if dec:
        ip, fp = t.rsplit(dec, 1)
        val = f"{re.sub(r'[,.]', '', ip) or '0'}.{fp}"
    else:
        val = re.sub(r"[,.]", "", t)
    try:
        return round(float(val), 3)
    except ValueError:
        return None


def parse_date(s) -> str | None:
    """Return ISO date (YYYY-MM-DD) or None. Day-first convention for numeric dates."""
    if s is None:
        return None
    t = re.sub(r"\s+", " ", str(s).replace(",", " ")).strip()
    if _digit_ratio(re.sub(r"[^\w]", "", t)) >= 0.5 and not re.search(r"[A-Za-z]{3,}", t):
        t = t.translate(_OCR_DIGIT)
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_currency(s) -> str | None:
    if s is None:
        return None
    return CURRENCY_MAP.get(str(s).strip().upper()) or CURRENCY_MAP.get(str(s).strip())


def norm_text(s) -> str:
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return re.sub(r"\s+", " ", s).strip(" .,:;|-")


def norm_id(s) -> str:
    return re.sub(r"[\s.:;,|]", "", unicodedata.normalize("NFKC", str(s))).upper()
