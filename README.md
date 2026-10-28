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
| Hygiene | only duplicates and zero/missing totals removed; refund rows (negative totals) and swapped timestamps (38% of rows) are kept and handled as features |
| Components | tip/tolls/surcharges are provided in test.csv, so they are used (`DROP_FARE_COMPONENTS` in config) |
| Temporal | duration, hour (+sin/cos), dow, month, weekend, rush-hour, night |
| Spatial | speed, min/mile, zone & zone-pair frequency, unsupervised zone-profile clusters, cluster pairs |
| Linear prior | Ridge on distance-per-rate-code, tolls, card-tip (fitted inside each fold); trees can't extrapolate to the largest fares, so boosters learn only the **residual** of this prior |
| Target | base fare = refund_sign × (total − known components); total = sign × base + components |
| CV | `StratifiedKFold` on target deciles, so each fold mirrors the distribution |
| Models | log1p target, early stopping, shrinkage (lr 0.03-0.05), L1/L2, row/col subsampling, min-child limits |
| Blend | 0.8·XGB + 0.1·LGBM + 0.1·Cat on the dollar scale, clipped to the training minimum |
| Logging | every fold's params, best iteration, train-vs-val gap -> `outputs/run_log.json` |

## Overfitting controls
Early stopping per fold · low learning rate · strong regularisation · winsorised training target ·
leak-free target encoding · fold-averaged test predictions (and optional multi-seed) · train/val gap logged.
Note: early stopping uses the validation fold, so OOF scores are very slightly optimistic; the public leaderboard is the honest check.

## Notes
* Target encoding (`add_target_encodings`) is implemented but disabled: it did not help on this data (zone IDs are noisy). Feature groups tip/toll/dist/dur/surch were tested and rejected (change ≤0.016 RMSE = noise).
* Weights are in `config.BLEND_WEIGHTS` (assumed order XGBoost/LightGBM/CatBoost = 80/10/10).
* "ExiBoost" was read as **XGBoost**.
