"""Token-level OCR noise. Operates per token so gold labels stay aligned."""
from __future__ import annotations

import random

CONFUSE = {
    "0": "O", "O": "0", "1": "l", "l": "1", "I": "1", "5": "S", "S": "5", "8": "B", "B": "8",
    "2": "Z", "Z": "2", "6": "b", "9": "g", "e": "c", "c": "e", "a": "o", "o": "a",
    ",": ".", ".": ",", "m": "rn",
}


def corrupt_token(tok: str, p: float, rng: random.Random) -> list[str]:
    """Return 1 piece (possibly corrupted) or 2 pieces (spurious space inserted)."""
    if p <= 0:
        return [tok]
    out = []
    for c in tok:
        if rng.random() < p:
            op = rng.choice(["sub", "sub", "sub", "del", "dup"])
            if op == "sub":
                out.append(CONFUSE.get(c, c))
            elif op == "dup":
                out.append(c + c)
            elif len(tok) > 1:
                continue  # delete
            else:
                out.append(c)
        else:
            out.append(c)
    s = "".join(out) or tok
    if len(s) >= 3 and rng.random() < p * 0.5:
        k = rng.randint(1, len(s) - 1)
        return [s[:k], s[k:]]
    return [s]


def apply_noise(lines: list[list[tuple[str, str]]], p: float, rng: random.Random):
    if p <= 0:
        return lines
    new = []
    for line in lines:
        nl = []
        for tok, lab in line:
            for piece in corrupt_token(tok, p, rng):
                nl.append((piece, lab))
        new.append(nl)
    return new
