import json
import random

import pytest

from docextract import synth
from docextract.baselines import rules
from docextract.llm.prompt import build_messages, extract_json, target_json
from docextract.metrics import aggregate, eval_doc, item_tp, risk_coverage
from docextract.normalize import parse_amount, parse_currency, parse_date
from docextract.schema import coerce
from docextract.validate import checks, repair


@pytest.mark.parametrize("s,v", [("1,234.56", 1234.56), ("1.234,56", 1234.56), ("1 234,56", 1234.56), ("12,345", 12.345),
                                 ("17,693.712", 17693.712), ("€ 9,99", 9.99), ("1O5.50", 105.5), ("abc", None)])
def test_parse_amount(s, v):
    assert parse_amount(s) == v


def test_parse_date_and_currency():
    assert parse_date("26.12.2022") == "2022-12-26"
    assert parse_date("December 26, 2022") == "2022-12-26"
    assert parse_date("2022-12-26") == "2022-12-26"
    assert parse_date("nonsense") is None
    assert parse_currency("€") == "EUR" and parse_currency("dt") == "TND"


def test_generator_labels_match_gold():
    """Data-quality gate: tokens labelled 'total'/'date' must parse back to the gold values."""
    rng = random.Random(0)
    for i in range(300):
        layout, lang = rng.choice(synth.ID_MIX + [("ood_e", "en_e"), ("invoice", "es")])
        d = synth.make_doc(rng, "train", layout, lang, 0.0, f"x{i}", "t")
        by = {}
        for ln in d["lines"]:
            for tok, lab in ln:
                by.setdefault(lab, []).append(tok)
        assert parse_amount(" ".join(by["total"])) == d["gold"]["total"]
        assert parse_date(" ".join(by["date"])) == d["gold"]["date"]
        assert " ".join(by["vendor"]).strip() == d["gold"]["vendor"]
        n_item_lines = sum(any(lab == "item_amount" for _, lab in ln) for ln in d["lines"])
        assert n_item_lines == len(d["gold"]["items"])
        assert checks(d["gold"])["consistent"]  # generator output obeys invoice arithmetic


def test_generator_deterministic(tmp_path):
    synth.build_all(tmp_path / "a", 5, 2, 2, 2, seed=7)
    synth.build_all(tmp_path / "b", 5, 2, 2, 2, seed=7)
    assert (tmp_path / "a" / "train.jsonl").read_text() == (tmp_path / "b" / "train.jsonl").read_text()


def test_train_test_vocab_disjoint():
    tr, te = synth.SPLIT_POOLS["train"], synth.SPLIT_POOLS["test"]
    for a, b in zip(tr, te):
        assert not set(a) & set(b)


def test_rules_clean_doc():
    rng = random.Random(1)
    d = synth.make_doc(rng, "test", "invoice", "en", 0.0, "r0", "t")
    p = coerce(rules.predict(d["text"]))
    r = eval_doc(p, d["gold"])
    assert r["total"] and r["date"] and r["invoice_number"]


def test_repair_fixes_total_and_item():
    g = {"vendor": "A", "date": "2024-01-01", "currency": "EUR", "subtotal": 100.0, "tax": 20.0, "total": 120.0,
         "items": [{"description": "x", "qty": 2, "unit_price": 30.0, "amount": 60.0},
                   {"description": "y", "qty": 1, "unit_price": 40.0, "amount": 40.0}]}
    bad = json.loads(json.dumps(g)); bad["total"] = 129.0
    fixed, acts = repair(bad)
    assert fixed["total"] == 120.0 and acts
    bad2 = json.loads(json.dumps(g)); bad2["items"][0]["amount"] = 66.0
    fixed2, _ = repair(bad2)
    assert fixed2["items"][0]["amount"] == 60.0
    assert checks(fixed2)["consistent"]


def test_repair_never_touches_consistent_pred():
    g = {"vendor": "A", "date": "2024-01-01", "currency": "EUR", "subtotal": 100.0, "tax": 20.0, "total": 120.0,
         "items": [{"description": "x", "qty": 2, "unit_price": 50.0, "amount": 100.0}]}
    fixed, acts = repair(g)
    assert fixed == g and acts == []


def test_item_matching_and_aggregate():
    gold = [{"description": "Wireless Mouse", "qty": 2, "unit_price": 5.0, "amount": 10.0}]
    pred = [{"description": "Wireles Mouse", "qty": 2, "unit_price": 5.0, "amount": 10.0}]
    assert item_tp(pred, gold, strict=True) == 1
    pred[0]["qty"] = 3
    assert item_tp(pred, gold, strict=True) == 0 and item_tp(pred, gold, strict=False) == 1
    rows = [eval_doc(None, {"vendor": "a", "invoice_number": None, "date": "2024-01-01", "currency": "EUR",
                            "subtotal": 1.0, "tax": 0.0, "total": 1.0, "items": gold})]
    assert aggregate(rows)["doc_em"] == 0.0 and aggregate(rows)["json_valid"] == 0.0


def test_risk_coverage_monotone_confidence_helps():
    conf = [0.9, 0.8, 0.7, 0.2, 0.1]
    good = [1, 1, 1, 0, 0]
    cov, risk, aurc = risk_coverage(conf, good)
    assert risk[0] == 0.0 and aurc < 0.4 * 0.6 + 1e-9
    _, _, aurc_bad = risk_coverage(conf[::-1], good)
    assert aurc < aurc_bad


def test_extract_json_robust():
    assert extract_json('Sure! ```json\n{"a": 1,}\n``` done') == {"a": 1}
    assert extract_json('{"a": "x}y", "b": {"c": 2}} trailing') == {"a": "x}y", "b": {"c": 2}}
    assert extract_json('{"a": 1') is None and extract_json("no json") is None


def test_prompt_roundtrip():
    d = synth.make_doc(random.Random(3), "train", "invoice", "fr", 0.0, "p", "t")
    msgs = build_messages(d["text"], shots=[d])
    assert msgs[0]["role"] == "system" and msgs[-1]["content"].endswith(d["text"])
    back = coerce(extract_json(target_json(d["gold"])))
    assert eval_doc(back, d["gold"])["doc_em"]


def test_rules_ext_handles_heldout_layout_and_nextline_amount():
    d = synth.make_doc(random.Random(5), "test", "ood_e", "en_e", 0.0, "e", "t")
    r = eval_doc(coerce(rules.predict(d["text"], extended=True)), d["gold"])
    assert r["invoice_number"] and r["subtotal"] and r["items_ok"]
    p = rules.predict("ACME SDN BHD\nDATE: 02-04-18\nTOTAL\n6.70", extended=True)
    assert p["total"] == 6.7 and p["date"] == "2018-04-02"


def test_sroie_weak_labels():
    from docextract.real_sroie import weak_labels
    lines = ["FTOF NOODLE HOUSE", "TOTAL", "RM 31.60", "DATE: 21/05/2018"]
    labs = weak_labels(lines, {"company": "FTOF NOODLE HOUSE", "date": "21/05/2018", "total": "RM31.60"})
    flat = {w: l for ln in labs for w, l in ln}
    assert flat["HOUSE"] == "vendor" and flat["31.60"] == "total" and flat["21/05/2018"] == "date"
