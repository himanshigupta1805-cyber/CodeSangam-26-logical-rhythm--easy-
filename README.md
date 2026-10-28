# The Portal Fare Prophecy: total_amount regression

Hybrid **XGBoost 80% / LightGBM 10% / CatBoost 10%** pipeline.

## Run
```bash
pip install -r requirements.txt
# put train.csv, test.csv (and sample_submission.csv if provided) in ./data
python -m src.train                 # -> outputs/submission.csv, run_log.json, oof_predictions.csv
python -m src.diagnostics           # -> EDA + residual plots, residual_report.md
```
On Kaggle: add `--data-dir /kaggle/input/<competition-slug> --out-dir /kaggle/working`, and raise `N_SEEDS` to 3 in `src/config.py`.
The submission uses the header of `sample_submission.csv` (falls back to `<id>,total_amount`).

## Pipeline
| Stage | What happens |
|---|---|
| Hygiene | duplicates, zero/negative fares, negative distance, bad duration, impossible speed removed from **train only**; test rows are never dropped, NaNs left to the trees + missing-indicators |
| Leakage guard | fare components (`fare_amount`, `tip_amount`, ...) dropped if present (`DROP_FARE_COMPONENTS` in config) |
| Temporal | duration, hour (+sin/cos), dow, month, weekend, rush-hour, night |
| Spatial | speed, min/mile, zone & zone-pair frequency, unsupervised zone-profile clusters, cluster pairs |
| Target encoding | PU, DO, pair, hour×dow on log-fare and fare-per-mile; **nested OOF**, recomputed inside every fold |
| CV | `StratifiedKFold` on target deciles, so each fold mirrors the distribution |
| Models | log1p target, early stopping, shrinkage (lr 0.03-0.05), L1/L2, row/col subsampling, min-child limits |
| Blend | 0.8·XGB + 0.1·LGBM + 0.1·Cat on the dollar scale, clipped to the training minimum |
| Logging | every fold's params, best iteration, train-vs-val gap -> `outputs/run_log.json` |

## Overfitting controls
Early stopping per fold · low learning rate · strong regularisation · winsorised training target ·
leak-free target encoding · fold-averaged test predictions (and optional multi-seed) · train/val gap logged.
Note: early stopping uses the validation fold, so OOF scores are very slightly optimistic; the public leaderboard is the honest check.

## Notes
* Weights are in `config.BLEND_WEIGHTS` (assumed order XGBoost/LightGBM/CatBoost = 80/10/10).
* "ExiBoost" was read as **XGBoost**.
* `demo_smoke_test/` holds a run on **synthetic** data (sklearn stand-in models) used only to verify the code.
  Its CSV is not a real prediction and must not be submitted.
