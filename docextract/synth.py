"""Synthetic OCR-style invoice/receipt generator with exact gold labels.

Why synthetic: no restricted data, perfect labels (token level + JSON), controllable
noise and *held-out layouts/languages* for out-of-distribution testing. The honest
limitation (synthetic != real scans) is discussed in docs/LIMITATIONS.md.

Splits
  train / val           layouts A,B,C (en/fr/de), train vocabulary pool, mixed noise
  test_id_nXX           same layouts, *disjoint* vendor/product vocabulary, per-char noise XX% (00,01,02,05)
  test_ood_layout_n02   unseen layout E (English labels never seen in training)
  test_ood_lang_n02     seen layout A but unseen language (Spanish)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

from .noise import apply_noise

ADJ = ["Atlas", "Nova", "Delta", "Blue", "Sahara", "Carthage", "Orion", "Vertex", "Lumen", "Cedar",
       "Zenith", "Pioneer", "Harbor", "Summit", "Aurora", "Falcon", "Olive", "Quantum", "Nimbus", "Mistral",
       "Granite", "Polar", "Amber", "Coral", "Helix", "Ivory", "Jade", "Kestrel", "Lotus", "Meridian"]
NOUN = ["Logistics", "Trading", "Systems", "Foods", "Energy", "Textiles", "Digital", "Medical", "Consulting",
        "Hardware", "Motors", "Print", "Solutions", "Supplies", "Telecom", "Labs", "Steel", "Paper", "Cargo", "Media"]
SUFFIX = {"en": ["Ltd", "Inc", "LLC"], "fr": ["SARL", "SA", "SAS"], "de": ["GmbH", "AG", "KG"], "es": ["S.L.", "S.A."]}
PRODUCTS = ["Wireless Mouse", "USB-C Cable 2m", "A4 Paper Ream", "Standing Desk", "LED Monitor 27in", "Office Chair",
            "Laptop Stand", "Toner Cartridge", "HDMI Adapter", "Notebook Pack", "Desk Lamp", "Webcam HD", "Router AX3000",
            "SSD 1TB", "Keyboard Mechanical", "Cleaning Kit", "Server Rack 42U", "Label Printer", "Cable Tray",
            "Power Strip 6-way", "Whiteboard 90x120", "Stapler Heavy", "Shredder P4", "Headset Pro", "Docking Station",
            "Ethernet Cable 5m", "Mouse Pad XL", "Flash Drive 64GB", "Projector Lamp", "Surge Protector", "Tablet Case",
            "Phone Charger 30W", "Badge Holder", "Paper Cutter", "Ink Bottle Set", "Cable Ties 100pk", "Speaker Compact",
            "Monitor Arm", "UPS 1500VA", "Barcode Scanner"]
STREETS = ["Rue de la Liberté", "Main Street", "Hauptstraße", "Avenue Habib Bourguiba", "Park Lane", "Calle Mayor"]

SPLIT_POOLS = {  # disjoint vocabularies between train and test
    "train": (ADJ[:20], NOUN[:14], PRODUCTS[:28]),
    "test": (ADJ[20:], NOUN[14:], PRODUCTS[28:]),
}

LABELS = {
    "en": dict(inv=["Invoice No", "Invoice #", "Inv. No."], date=["Date", "Invoice Date"], due=["Due Date", "Payment due"],
               sub=["Subtotal"], tax=["VAT", "Tax"], total=["Total", "Total Due", "Amount Due"],
               hdr="Description Qty Unit Price Amount"),
    "fr": dict(inv=["Facture N°", "N° Facture"], date=["Date", "Date d'émission"], due=["Échéance", "Date d'échéance"],
               sub=["Sous-total", "Total HT"], tax=["TVA"], total=["Total TTC", "Total"], hdr="Désignation Qté P.U. Montant"),
    "de": dict(inv=["Rechnung Nr.", "Rechnungsnummer"], date=["Datum", "Rechnungsdatum"], due=["Fällig am", "Zahlungsziel"],
               sub=["Zwischensumme", "Netto"], tax=["MwSt", "USt"], total=["Gesamtbetrag", "Summe", "Gesamt"],
               hdr="Beschreibung Menge Einzelpreis Betrag"),
    "es": dict(inv=["Factura Nº", "Nº de factura"], date=["Fecha", "Fecha de emisión"], due=["Vencimiento"],
               sub=["Subtotal", "Base imponible"], tax=["IVA"], total=["Total", "Importe total"],
               hdr="Descripción Cant. Precio Importe"),
    "en_e": dict(inv=["Bill #"], date=["Issued"], sub=["Net"], tax=["Sales tax"], total=["Grand total"]),
}
AMOUNT_STYLES = {"en": ["en", "en", "en", "eu"], "fr": ["fr", "fr", "eu", "en"], "de": ["eu", "eu", "eu", "en"],
                 "es": ["eu", "eu", "fr"], "en_e": ["en"]}
DATE_FMTS = ["%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d %b %Y", "%B %d, %Y"]
CUR_TOKENS = {"EUR": ["EUR", "€"], "USD": ["USD", "$"], "TND": ["TND", "DT"]}


def T(text: str, label: str = "O"):
    return [(w, label) for w in text.split()]


def fmt_amount(v: float, style: str, decimals: int) -> str:
    s = f"{v:,.{decimals}f}"  # en: 1,234.56
    if style == "en":
        return s
    if style == "eu":
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s.replace(",", " ").replace(".", ",")  # fr: 1 234,56


def _fake_phone(rng):
    return "+" + str(rng.randint(30, 216)) + " " + " ".join(str(rng.randint(10, 99)) for _ in range(4))


def _iban(rng):
    return "IBAN " + rng.choice(["FR", "DE", "TN", "ES", "GB"]) + str(rng.randint(10, 99)) + " " + \
        " ".join(str(rng.randint(1000, 9999)) for _ in range(4))


def make_gold(rng, pool, layout, currency):
    adjs, nouns, prods = pool
    decimals = 3 if currency == "TND" else 2
    lang_for_suffix = rng.choice(["en", "fr", "de"])
    vendor = f"{rng.choice(adjs)} {rng.choice(nouns)} {rng.choice(SUFFIX[lang_for_suffix])}"
    items = []
    for _ in range(rng.randint(1, 7)):
        qty = rng.randint(1, 9)
        unit = round(rng.uniform(1, 60) if rng.random() < 0.8 else rng.uniform(100, 2500), decimals)
        items.append({"description": rng.choice(prods), "qty": qty, "unit_price": unit,
                      "amount": round(qty * unit, decimals)})
    sub = round(sum(i["amount"] for i in items), decimals)
    rate = rng.choice([0, 5, 7, 10, 19, 20, 21])
    tax = round(sub * rate / 100, decimals)
    date = dt.date(2022, 1, 1) + dt.timedelta(days=rng.randint(0, 1640))
    inv = None
    if layout != "receipt":
        inv = rng.choice([f"INV-{rng.randint(2019, 2026)}-{rng.randint(1, 99999):05d}",
                          f"{rng.randint(2019, 2026)}/{rng.randint(1, 9999):04d}",
                          f"FA{rng.randint(100000, 999999)}", f"{rng.randint(10000, 9999999)}"])
    gold = {"vendor": vendor, "invoice_number": inv, "date": date.isoformat(), "currency": currency,
            "subtotal": sub, "tax": tax, "total": round(sub + tax, decimals), "items": items}
    if layout == "receipt" and rng.random() < 0.3:
        gold["subtotal"] = None  # receipt without subtotal line
    return gold, rate, decimals, date


def render(gold, layout, lang, style, rate, decimals, date, rng):
    L = LABELS[lang]
    fa = lambda v: fmt_amount(v, style, decimals)  # noqa: E731
    dfmt = rng.choice(DATE_FMTS)
    ds = date.strftime(dfmt)
    cur = rng.choice(CUR_TOKENS[gold["currency"]])
    cur_before = rng.random() < 0.25
    due_s = (date + dt.timedelta(days=rng.randint(15, 60))).strftime(dfmt)
    pick = lambda k: rng.choice(L[k])  # noqa: E731

    def money_line(label, value, field, with_cur=False, rate_txt=""):
        line = T(label + rate_txt + ":" if layout != "receipt" else label + rate_txt)
        amt = T(fa(value), field)
        c = T(cur, "currency") if with_cur else []
        return line + (c + amt if cur_before else amt + c)

    def tot_lines():
        out = []
        if gold["subtotal"] is not None:
            out.append(money_line(pick("sub"), gold["subtotal"], "subtotal"))
        out.append(money_line(pick("tax"), gold["tax"], "tax", rate_txt=f" ({rate}%)"))
        out.append(money_line(pick("total"), gold["total"], "total", with_cur=True))
        return out

    lines = [T(gold["vendor"], "vendor"), T(f"{rng.randint(1, 250)} {rng.choice(STREETS)}"), T("Tel: " + _fake_phone(rng))]
    if layout == "invoice":  # A
        lines += [T(pick("inv") + ":") + T(gold["invoice_number"], "invoice_number"),
                  T(pick("date") + ":") + T(ds, "date"), T(pick("due") + ":") + T(due_s), T(L["hdr"])]
        for it in gold["items"]:
            lines.append(T(it["description"], "item_desc") + T(str(it["qty"]), "item_qty") +
                         T(fa(it["unit_price"]), "item_unit") + T(fa(it["amount"]), "item_amount"))
        lines += tot_lines() + [T(_iban(rng))]
    elif layout == "receipt":  # B (no invoice number, 'x' separators)
        lines += [T(f"{pick('date')} ") + T(ds, "date") + T(f"{rng.randint(8, 21)}:{rng.randint(10, 59)}")]
        for it in gold["items"]:
            lines.append(T(it["description"], "item_desc") + T(str(it["qty"]), "item_qty") + T("x") +
                         T(fa(it["unit_price"]), "item_unit") + T(fa(it["amount"]), "item_amount"))
        lines += tot_lines() + [T("Thank you!")]
    elif layout == "compact":  # C (pipes, invoice number + date on same line)
        lines = [T(rng.choice({"en": ["INVOICE"], "fr": ["FACTURE"], "de": ["RECHNUNG"]}[lang]))] + lines
        lines += [T(pick("inv") + " ") + T(gold["invoice_number"], "invoice_number") + T("|") +
                  T(pick("date") + " ") + T(ds, "date"), T(pick("due") + " " + due_s)]
        for it in gold["items"]:
            lines.append(T(it["description"], "item_desc") + T("|") + T(str(it["qty"]), "item_qty") + T("|") +
                         T(fa(it["unit_price"]), "item_unit") + T("|") + T(fa(it["amount"]), "item_amount"))
        lines += tot_lines()
    elif layout == "ood_e":  # E: totals first, items as 'qty x desc @ unit = amount'
        lines = [T(gold["vendor"], "vendor"), T(f"{rng.randint(1, 250)} {rng.choice(STREETS)}")]
        lines += [money_line(L["total"][0], gold["total"], "total", with_cur=True),
                  money_line(L["sub"][0], gold["subtotal"], "subtotal"),
                  money_line(L["tax"][0], gold["tax"], "tax", rate_txt=f" ({rate}%)"),
                  T(L["inv"][0] + " ") + T(gold["invoice_number"], "invoice_number"),
                  T(L["date"][0] + " ") + T(ds, "date")]
        for it in gold["items"]:
            lines.append(T(str(it["qty"]), "item_qty") + T("x") + T(it["description"], "item_desc") + T("@") +
                         T(fa(it["unit_price"]), "item_unit") + T("=") + T(fa(it["amount"]), "item_amount"))
    return lines


def make_doc(rng, split_pool, layout, lang, noise_p, doc_id, split):
    currency = rng.choices(["EUR", "USD", "TND"], [0.45, 0.3, 0.25])[0]
    gold, rate, decimals, date = make_gold(rng, SPLIT_POOLS[split_pool], layout, currency)
    style = rng.choice(AMOUNT_STYLES[lang])
    lines = render(gold, layout, lang, style, rate, decimals, date, rng)
    lines = apply_noise(lines, noise_p, rng)
    text = "\n".join(" ".join(t for t, _ in ln) for ln in lines)
    return {"id": doc_id, "text": text, "lines": [[list(t) for t in ln] for ln in lines], "gold": gold,
            "meta": {"layout": layout, "lang": lang, "noise": noise_p, "style": style, "split": split}}


ID_MIX = [("invoice", "en"), ("invoice", "fr"), ("invoice", "de"), ("receipt", "en"), ("receipt", "fr"),
          ("compact", "en"), ("compact", "fr"), ("compact", "de")]
TRAIN_NOISE = [0.0, 0.01, 0.02, 0.04]


def build_split(name, n, seed, pool, mix, noise_choices):
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        layout, lang = rng.choice(mix)
        docs.append(make_doc(rng, pool, layout, lang, rng.choice(noise_choices), f"{name}-{i:05d}", name))
    return docs


def build_all(out: Path, n_train=3000, n_val=500, n_test=500, n_ood=300, seed=1234):
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": build_split("train", n_train, seed, "train", ID_MIX, TRAIN_NOISE),
        "val": build_split("val", n_val, seed + 1, "train", ID_MIX, TRAIN_NOISE),
    }
    for tag, p in [("n00", 0.0), ("n01", 0.01), ("n02", 0.02), ("n05", 0.05)]:
        splits[f"test_id_{tag}"] = build_split(f"test_id_{tag}", n_test, seed + 10, "test", ID_MIX, [p])
    splits["test_ood_layout_n02"] = build_split("test_ood_layout_n02", n_ood, seed + 20, "test",
                                                [("ood_e", "en_e")], [0.02])
    splits["test_ood_lang_n02"] = build_split("test_ood_lang_n02", n_ood, seed + 30, "test", [("invoice", "es")], [0.02])
    for name, docs in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return list(splits)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-ood", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    print("wrote:", build_all(Path(a.out), a.n_train, a.n_val, a.n_test, a.n_ood, a.seed))
