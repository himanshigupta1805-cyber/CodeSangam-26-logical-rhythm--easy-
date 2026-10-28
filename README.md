# Tabular Fare Prediction

A phased, reproducible regression project for predicting NYC taxi `total_amount` from trip data. The project uses only the supplied files in `data/raw/` and keeps the modeling, feature engineering, evaluation, and inference code in a small Python package.

## Project phases and completed work

### Phase 1 — Project setup and data contract (complete)

Created the `fare_prediction` package, train and predict command-line entry points, dependency lists, and local input/output folders. The project reads the supplied `train.csv`, `test.csv`, and `sample_submission.csv`. Training uses `total_amount` as the target; prediction preserves the test `id` column.

### Phase 2 — Data hygiene and feature engineering (complete)

The pipeline drops only rows with invalid targets and reports how many were removed. It replaces infinite feature values with missing values, imputes numeric and categorical missing values inside each training fold, and caps numeric feature tails using quantile bounds fitted on that fold's training portion. It derives trip duration, pickup hour/day/month/weekday, weekend and rush-hour flags, and cyclic hour features. The supplied data contains pickup and dropoff zone IDs but no coordinates, so the pipeline builds origin/destination route interactions rather than inventing physical coordinates or geographic clusters. Route, zone, vendor, rate-code, and payment categories get fold-safe frequency and smoothed target-mean aggregates. Target aggregates are cross-fitted within each outer training fold to avoid using validation targets.

### Phase 3 — Baselines and cross-validation (complete)

The package includes scikit-learn HistGradientBoosting and ExtraTrees models, plus optional LightGBM, XGBoost, and CatBoost candidates. The training command reports 5-fold RMSE and MAE. The completed final run used HistGradientBoosting with all engineered features and the fold-safe aggregate encoder. It achieved **RMSE 8.99 ± 1.61** and **MAE 3.39** across the 5 validation folds. A fold-matched mean-only baseline scored RMSE 26.04 and MAE 16.74.

LightGBM and CatBoost were available in the environment and are selectable with `--models`; XGBoost is listed in the optional booster requirements but was not installed in the environment. PyTorch Tabular was not installed, so it was not used for this run.

### Phase 4 — Model selection, blending, and test predictions (complete)

The CLI can select the lowest-RMSE candidate or blend the top candidates with inverse-RMSE weights using `--ensemble`. For the completed dataset run, the validated HistGradientBoosting model was refit on all 8,000 labeled rows. It generated 2,000 predictions from `test.csv` in `outputs/submission.csv`. The output columns and ID order were checked against `sample_submission.csv`; all 2,000 target predictions are present.

### Phase 5 — Evaluation and residual diagnostics (complete)

Saved fold metrics, out-of-fold predictions, residuals, and a residual scatter plot. The residual CSVs allow error review by actual amount, predicted amount, and trip attributes by joining on row order. No leaderboard score is claimed; the reported score is the local shuffled 5-fold validation result.

## Use the supplied dataset

Python 3.10 or later is recommended. Install core dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the same 5-fold HistGradientBoosting training used for the included model artifact:

```bash
python -m fare_prediction train --train data/raw/train.csv --target total_amount --id-column id --models hist_gradient_boosting --folds 5 --output-dir outputs
```

Generate the test predictions:

```bash
python -m fare_prediction predict --model outputs/model.joblib --input data/raw/test.csv --output outputs/submission.csv --id-column id
```

To compare all locally installed candidates and create a weighted blend from the top three:

```bash
python -m pip install -r requirements-boosters.txt
python -m fare_prediction train --train data/raw/train.csv --models all --ensemble --ensemble-top 3 --folds 5 --output-dir outputs
```

Run `python -m fare_prediction train --help` and `python -m fare_prediction predict --help` for options. Common TLC pickup/dropoff timestamp columns are detected automatically. Coordinate-based haversine distance is derived only if latitude/longitude columns are present.

## Artifacts

- `outputs/model.joblib` — fitted model and expected feature schema.
- `outputs/metrics.json` — cross-validation scores, fold spread, and selected model.
- `outputs/oof_predictions.csv` — actual and out-of-fold predicted target values.
- `outputs/residuals.csv` and `outputs/residuals.png` — residual table and plot.
- `outputs/submission.csv` — 2,000 test predictions in the sample submission format.

Input data and generated outputs are git-ignored so the project does not accidentally commit local datasets or model artifacts.

## Repository layout

```text
fare_prediction/          Feature engineering, model definitions, and CLI
requirements.txt           Core packages
requirements-boosters.txt  Optional LightGBM, XGBoost, and CatBoost
 data/raw/                 Supplied train, test, and sample submission CSV files
 outputs/                  Model, metrics, diagnostics, and submission
```
