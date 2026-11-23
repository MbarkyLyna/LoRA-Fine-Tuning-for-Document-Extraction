# Real-data check: SROIE receipts (held-out n=126)

Fields available in SROIE: vendor (company), date, total. Exact match after the same normalisation as the synthetic benchmark; 95% bootstrap CI in brackets. `all3` = all three fields right.

| system | vendor EM | date EM | total EM | all3 | field EM (macro) |
|---|---|---|---|---|---|
| crf-sroie | 58.7 [50-67] | 93.7 [89-98] | 52.4 [44-61] | 35.7 [28-44] | 68.3 |
| crf-synth | 5.6 [2-10] | 18.3 [12-25] | 7.1 [3-11] | 0.0 [0-0] | 10.3 |
| rules-ext | 72.2 [64-79] | 97.6 [94-100] | 61.1 [52-70] | 40.5 [32-49] | 77.0 |
| rules | 50.0 [40-59] | 59.5 [51-68] | 4.8 [2-9] | 4.8 [2-9] | 38.1 |
