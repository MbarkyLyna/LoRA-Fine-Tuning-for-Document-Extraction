# Limitations and honest caveats

* **Synthetic data, plus one small real check.** The main benchmark is synthetic. The SROIE check covers only three fields on
  126 Malaysian receipts whose OCR lines are human-transcribed (no OCR noise), with weak token labels for the CRF; it is a sanity
  check on real layouts, not a real-world accuracy estimate. Invoices with line items, multiple currencies and noisy scans are untested.
* **Synthetic data.** Templates are few (4 layouts, 4 languages) and noise is character-level only. Real scans add
  layout shifts, merged/split lines, missing lines, logos, tables with cell borders, handwriting. Perfect CRF scores on
  clean in-distribution text mostly show that the synthetic task is easy when noise is zero, not that the problem is solved.
* **The 5% noise setting is brutal**: any single corrupted digit in a long document breaks doc-level exact match.
  Prefer field EM / item F1 for comparison and treat doc EM as a strict upper-bound metric.
* **Rules baseline.** `rules` (v1) is frozen: written for EN/FR/DE layouts A-C only, never patched for held-out data.
  `rules-ext` (v2) adds Spanish/layout-E patterns after inspecting them, plus next-line values, date formats and vendor cleanup
  tuned on the SROIE train split only; its OOD scores are therefore not held-out. A maintained rule system would be extended, which is precisely
  the maintenance cost the OOD columns illustrate. One known weakness: space-grouped thousands ("8 373.32") are ambiguous
  between quantity and price in a regex.
* **LoRA results are absent until you run the notebook.** The training/inference code follows standard `transformers`/`peft`
  APIs but has not been executed by the author of this repository snapshot; expect to fix small version-specific issues
  (e.g. argument names) on first run. The unit tests cover everything except the torch code paths.
* **Repair heuristics encode invoice arithmetic** (items sum to subtotal, subtotal + tax = total, standard VAT rates).
  They cannot fix vendor/date/invoice-number errors, and they can in principle "repair" a correct-but-unusual document
  (e.g. discounts, rounding at line level). The report measures the net effect; always re-measure on your own data.
* **Numbers are from one seed.** Bootstrap CIs capture test-set sampling noise only, not training-seed variance.
* **Day-first dates.** Numeric dates are interpreted as DD/MM/YYYY throughout.
