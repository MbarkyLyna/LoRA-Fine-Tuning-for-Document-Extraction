"""Token-level CRF tagger (sklearn-crfsuite). Classic strong, cheap, CPU-only baseline
with calibrated per-token marginals -> usable confidence for selective prediction."""
from __future__ import annotations

import re

import sklearn_crfsuite

from ..normalize import parse_amount, parse_currency, parse_date

RE_AMT = re.compile(r"^[\d.,]+$")


def _shape(w: str) -> str:
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1+", r"\1\1", s)


def doc_features(lines: list[list[str]]):
    n_lines = len(lines)
    feats, index = [], []
    for li, toks in enumerate(lines):
        low = [t.lower() for t in toks]
        words = {t for t in low if t.isalpha()}
        prev_words = {t.lower() for t in lines[li - 1] if t.isalpha()} if li > 0 else set()
        next_words = {t.lower() for t in lines[li + 1] if t.isalpha()} if li + 1 < n_lines else set()
        n_num = sum(any(c.isdigit() for c in t) for t in toks)
        for ti, w in enumerate(toks):
            f = {"bias": 1.0, "w": low[ti], "shape": _shape(w), "len": min(len(w), 12),
                 "isdig": w.isdigit(), "hasdig": any(c.isdigit() for c in w), "amt": bool(RE_AMT.match(w)),
                 "pre2": low[ti][:2], "suf3": low[ti][-3:], "title": w.istitle(), "upper": w.isupper(),
                 "pos": min(ti, 6), "rpos": min(len(toks) - 1 - ti, 6), "line_first": low[0],
                 "line_rel": round(10 * li / max(n_lines, 1)), "line_len": min(len(toks), 10),
                 "line_nnum": min(n_num, 5), "line_colon": any(":" in t for t in toks),
                 "line_pipe": "|" in toks, "li_first": li == 0, "li_small": li < 3}
            for wd in words:
                f[f"lw:{wd}"] = 1.0
            for wd in prev_words:  # label often sits on the previous line (real receipts, table-style layouts)
                f[f"pl:{wd}"] = 1.0
            for wd in next_words:
                f[f"nl:{wd}"] = 1.0
            for d in (-2, -1, 1, 2):
                j = ti + d
                if 0 <= j < len(toks):
                    f[f"w{d}"] = low[j]
                    f[f"s{d}"] = _shape(toks[j])
                else:
                    f[f"w{d}"] = "<pad>"
            feats.append(f)
            index.append((li, ti))
    return feats, index


class CRFExtractor:
    def __init__(self, c1=0.05, c2=0.05, max_iter=100):
        self.crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iter,
                                        all_possible_transitions=True)

    def fit(self, docs):
        X, y = [], []
        for d in docs:
            toks = [[t for t, _ in ln] for ln in d["lines"]]
            f, _ = doc_features(toks)
            X.append(f)
            y.append([lab for ln in d["lines"] for _, lab in ln])
        self.crf.fit(X, y)
        return self

    def predict(self, text: str):
        lines = [l.split() for l in text.split("\n")]
        lines = [l for l in lines if l]
        feats, index = doc_features(lines)
        labels = self.crf.predict_single(feats)
        marg = self.crf.predict_marginals_single(feats)
        conf = [m[lab] for m, lab in zip(marg, labels)]
        return assemble(lines, index, labels, conf)


def _groups(lines, index, labels, conf, field):
    groups, cur = [], None
    for (li, ti), lab, c in zip(index, labels, conf):
        if lab == field:
            if cur and cur["line"] == li and cur["end"] == ti - 1:
                cur["toks"].append(lines[li][ti]); cur["conf"].append(c); cur["end"] = ti
            else:
                cur = {"line": li, "end": ti, "toks": [lines[li][ti]], "conf": [c]}
                groups.append(cur)
        else:
            cur = cur if cur and cur["line"] == li and lab == field else None
    return groups


def assemble(lines, index, labels, conf):
    out = {"vendor": None, "invoice_number": None, "date": None, "currency": None, "subtotal": None,
           "tax": None, "total": None, "items": []}
    confs = []
    parsers = {"vendor": lambda s: s.strip() or None, "invoice_number": lambda s: s.replace(" ", "") or None,
               "date": parse_date, "currency": parse_currency, "subtotal": parse_amount, "tax": parse_amount,
               "total": parse_amount}
    for f, parse in parsers.items():
        gs = _groups(lines, index, labels, conf, f)
        if not gs:
            continue
        g = max(gs, key=lambda g: sum(g["conf"]) / len(g["conf"]))
        out[f] = parse(" ".join(g["toks"]))
        confs.append(sum(g["conf"]) / len(g["conf"]))
    by_line: dict[int, dict[str, list]] = {}
    for (li, ti), lab, c in zip(index, labels, conf):
        if lab.startswith("item_"):
            by_line.setdefault(li, {}).setdefault(lab, []).append((lines[li][ti], c))
            confs.append(c)
    for li in sorted(by_line):
        d = by_line[li]
        txt = lambda k: " ".join(t for t, _ in d.get(k, []))  # noqa: E731
        out["items"].append({"description": txt("item_desc") or None, "qty": parse_amount(txt("item_qty")),
                             "unit_price": parse_amount(txt("item_unit")), "amount": parse_amount(txt("item_amount"))})
    meta = {"conf_min": min(confs) if confs else 0.0, "conf_mean": sum(confs) / len(confs) if confs else 0.0}
    return out, meta
