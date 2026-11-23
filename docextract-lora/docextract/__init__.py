"""docextract: benchmark-first structured extraction from OCR'd invoices/receipts.

Systems: rule-based baseline, CRF token tagger, and LoRA-fine-tuned small LLMs,
all scored by the same evaluator, plus an arithmetic verifier/repair layer and
selective-prediction (risk-coverage) analysis.
"""
__version__ = "0.1.0"
