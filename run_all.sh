#!/usr/bin/env bash
# Reproduce every result and figure. Requires data in data_raw/ (see README).
set -euo pipefail
pip install -q pandas numpy scipy scikit-learn lightgbm xgboost shap pyarrow matplotlib
mkdir -p data results figures
python -c "import sys; sys.path.insert(0,'src'); from load import *; load_phase1().to_parquet('data/phase1.parquet'); load_phase2().to_parquet('data/phase2.parquet')"
python src/phase1.py      > results/phase1.log 2>&1 || true   # 75/25 block may exceed 8 GB; phase1b reruns it lean
python src/phase1b.py     > results/phase1b.log 2>&1
python src/phase1_ablation.py > results/phase1_ablation.log 2>&1
python src/phase2.py      > results/phase2.log 2>&1 || true   # rolling block is redone in phase2b
python src/phase2b.py     > results/phase2b.log 2>&1
python src/india.py       > results/india.log 2>&1
python src/figures.py && python src/figures_india.py
