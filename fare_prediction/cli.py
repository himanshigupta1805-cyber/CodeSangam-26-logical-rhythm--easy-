from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import engineer_features
from .models import available_models, make_pipeline


def _read(path):
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"CSV not found: {file}. Add the dataset file or pass its path with --train/--input.")
    return pd.read_csv(file, low_memory=False)


def train(args):
    train_df = _read(args.train)
    if args.target not in train_df.columns:
        raise ValueError(f"Target column {args.target!r} is missing. Columns: {list(train_df.columns)}")
    y = pd.to_numeric(train_df[args.target], errors="coerce")
    valid = y.notna() & np.isfinite(y)
    removed = int((~valid).sum())
    train_df, y = train_df.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if len(y) < args.folds:
        raise ValueError(f"Need at least {args.folds} valid rows for {args.folds}-fold CV; found {len(y)}.")
    identifiers = train_df[args.id_column].reset_index(drop=True) if args.id_column in train_df else pd.Series(np.arange(len(y)), name="row_id")
    X = train_df.drop(columns=[args.target, args.id_column], errors="ignore")
    X = engineer_features(X, args.pickup_column, args.dropoff_column)
    X = X.replace([np.inf, -np.inf], np.nan)
    models = available_models(args.seed)
    requested = args.models.split(",") if args.models != "all" else list(models)
    missing = sorted(set(requested) - set(models))
    if missing:
        raise ValueError(f"Unavailable model(s): {', '.join(missing)}. Installed candidates: {', '.join(models)}. Install requirements-boosters.txt for optional boosters.")
    if not requested:
        raise ValueError("No model candidates selected.")
    splits = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X))
    scores, oof = {}, {}
    for name in requested:
        predictions = np.full(len(y), np.nan)
        fold_rmse, fold_mae = [], []
        for fit_idx, val_idx in splits:
            pipeline = make_pipeline(clone(models[name]), X.iloc[fit_idx])
            pipeline.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            pred = np.asarray(pipeline.predict(X.iloc[val_idx]), dtype=float)
            predictions[val_idx] = pred
            fold_rmse.append(float(np.sqrt(mean_squared_error(y.iloc[val_idx], pred))))
            fold_mae.append(float(mean_absolute_error(y.iloc[val_idx], pred)))
        oof[name] = predictions
        scores[name] = {"rmse_mean": float(np.mean(fold_rmse)), "rmse_std": float(np.std(fold_rmse)),
                        "mae_mean": float(np.mean(fold_mae)), "fold_rmse": fold_rmse}
        print(f"{name}: RMSE {scores[name]['rmse_mean']:.4f} +/- {scores[name]['rmse_std']:.4f}; MAE {scores[name]['mae_mean']:.4f}")
    ranked = sorted(requested, key=lambda key: scores[key]["rmse_mean"])
    selected = ranked[: min(args.ensemble_top, len(ranked))] if args.ensemble else ranked[:1]
    weights = np.array([1 / max(scores[name]["rmse_mean"], 1e-9) for name in selected])
    weights /= weights.sum()
    ensemble_oof = sum(weight * oof[name] for name, weight in zip(selected, weights))
    scores["selected_oof"] = {"models": selected, "weights": weights.tolist(),
                              "rmse": float(np.sqrt(mean_squared_error(y, ensemble_oof))),
                              "mae": float(mean_absolute_error(y, ensemble_oof))}
    fitted = [(name, make_pipeline(clone(models[name]), X)) for name in selected]
    estimator = fitted[0][1] if len(fitted) == 1 else VotingRegressor(fitted, weights=weights.tolist(), n_jobs=1)
    estimator.fit(X, y)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": estimator, "features": X.columns.tolist(), "target": args.target,
                 "selected_models": selected, "weights": weights.tolist(), "pickup_column": args.pickup_column,
                 "dropoff_column": args.dropoff_column}, out_dir / "model.joblib")
    pd.DataFrame({"id": identifiers, "actual": y, "prediction": ensemble_oof,
                  "residual": y - ensemble_oof}).to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({"actual": y, "prediction": ensemble_oof,
                  "residual": y - ensemble_oof}).to_csv(out_dir / "residuals.csv", index=False)
    metrics = {"rows_used": len(y), "rows_dropped_for_invalid_target": removed, "folds": args.folds,
               "target": args.target, "models": scores}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        residual = y.to_numpy() - ensemble_oof
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(ensemble_oof, residual, s=8, alpha=0.35)
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1)
        ax.set(xlabel="Out-of-fold prediction", ylabel="Residual (actual - prediction)", title="Out-of-fold residual diagnostics")
        fig.tight_layout()
        fig.savefig(out_dir / "residuals.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; skipping residuals.png")
    print(f"Selected model(s): {', '.join(selected)}. Artifacts saved to {out_dir.resolve()}")


def predict(args):
    artifact = joblib.load(args.model)
    estimator = artifact["model"]
    data = _read(args.input)
    ids = data[args.id_column].copy() if args.id_column and args.id_column in data else pd.Series(np.arange(len(data)), name="row_id")
    X = data.drop(columns=[args.id_column], errors="ignore")
    X = engineer_features(X, artifact.get("pickup_column"), artifact.get("dropoff_column"))
    X = X.reindex(columns=artifact["features"], fill_value=np.nan).replace([np.inf, -np.inf], np.nan)
    predictions = estimator.predict(X)
    result = pd.DataFrame({args.id_column or "row_id": ids, artifact["target"]: predictions})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Wrote {len(result)} predictions to {destination.resolve()}")


def main():
    parser = argparse.ArgumentParser(prog="fare_prediction", description="Cross-validated tabular fare regression pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train", help="cross-validate candidate models, fit final model, and write diagnostics")
    tr.add_argument("--train", required=True, help="labeled CSV path")
    tr.add_argument("--target", default="total_amount")
    tr.add_argument("--id-column", default="id")
    tr.add_argument("--pickup-column", default="tpep_pickup_datetime")
    tr.add_argument("--dropoff-column", default="tpep_dropoff_datetime")
    tr.add_argument("--folds", type=int, default=5)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--models", default="hist_gradient_boosting,extra_trees", help="comma-separated names or all")
    tr.add_argument("--ensemble", action="store_true", help="weighted blend selected by CV RMSE")
    tr.add_argument("--ensemble-top", type=int, default=3)
    tr.add_argument("--output-dir", default="outputs")
    tr.set_defaults(func=train)
    pr = commands.add_parser("predict", help="predict an unlabeled CSV with a saved model")
    pr.add_argument("--model", required=True)
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default="outputs/submission.csv")
    pr.add_argument("--id-column", default="id")
    pr.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
