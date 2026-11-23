#!/usr/bin/env bash
# Fetch ONLY the text boxes + key labels of the public SROIE (ICDAR 2019) receipts (~few MB, not the 600MB images).
# Data is not redistributed in this repo; check the dataset's terms before commercial use.
set -euo pipefail
mkdir -p data
if [ ! -d data/_sroie ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/zzzDavid/ICDAR-2019-SROIE data/_sroie
  (cd data/_sroie && git sparse-checkout set data/box data/key)
fi
python -m docextract.real_sroie --src data/_sroie/data --out data
