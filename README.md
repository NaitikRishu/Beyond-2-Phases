# Beyond Two Phases — reproducible code

Companion code for *"Beyond Two Phases: Calibrated, Leakage-Audited and Economically Evaluated P2P Credit Risk Models,
with an Extension to India's CIBIL Ecosystem"* (Naitik Rishu, November 2026). Base paper: Turiel J.D. & Aste T. (2020),
*R. Soc. Open Sci.* 7:191649, doi:10.1098/rsos.191649.

## Data
The original authors' cleaned files (Git LFS, ~160 MB zipped) — identical to those used in the base paper:

```
mkdir -p data_raw && cd data_raw
curl -L -O https://media.githubusercontent.com/media/jeremyDT/P2P-lending-with-AI/master/data/input_file_1.csv.zip
curl -L -O https://media.githubusercontent.com/media/jeremyDT/P2P-lending-with-AI/master/data/input_file_2.csv.zip
unzip input_file_1.csv.zip && unzip input_file_2.csv.zip
```
Edit `RAW` in `src/load.py` if you put them elsewhere, then run `python src/load.py`-style caching (see `run_all.sh`).

## Run order (≈ 1 h on 2 CPU cores, 8 GB RAM)
| Script | What it does | Main outputs |
|---|---|---|
| `src/load.py` (via run_all) | caches CSVs to parquet | `data/phase1.parquet`, `data/phase2.parquet` |
| `src/phase1.py` | Phase 1 replication (LR) + LightGBM, 90/10 split | `results/phase1_*.csv`, log |
| `src/phase1b.py` | memory-lean 75/25 split + merges results | `results/phase1_results.json` |
| `src/phase1_ablation.py` | employment-length ablation | `results/phase1_ablation.json` |
| `src/phase2.py` | Phase 2: replication, leakage audit, LightGBM/XGBoost, calibration, economics, 3-band, conformal, IPW, SHAP, small business | `results/phase2_*` |
| `src/phase2b.py` | rolling-origin validation + final assembly | `results/phase2_results.json`, `phase2_rolling.csv` |
| `src/india.py` | CIBIL-world semi-synthetic extension + sensitivity sweep | `results/india_*` |
| `src/figures.py`, `src/figures_india.py` | all figures | `figures/*.png` |

Set `GRID_JSON=results/phase2_grid.json` to skip the 6-model LightGBM tuning grid (≈10 min).

## Modules
* `src/metrics.py` — DeLong AUC CIs and test (fast Sun–Xu algorithm), KS, ECE, Brier, recall helpers.
* `src/nn.py` — NumPy re-implementation of the paper's DNN (tanh, dropout, Adam, cross-entropy).

## Honesty notes
* The India extension is **semi-synthetic**: borrowers and outcomes are real (Lending Club); the FICO→CIBIL and DTI→FOIR
  maps are deterministic; NTC status and Account-Aggregator/UPI features are simulated with explicit, swept parameters
  (`gamma`, `beta`). Treat its numbers as scenario analysis, not estimates for India.
* Profit figures assume loans run to term (upper bound) and an assumed LGD (0.65; 0.4–0.9 sensitivity).
