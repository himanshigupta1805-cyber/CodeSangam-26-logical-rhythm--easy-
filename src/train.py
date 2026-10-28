"""End-to-end training: clean -> features -> stratified K-Fold -> 3 GBMs -> 80/10/10 blend -> submission."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .data import load_data, find_id_col, drop_leaky_columns, clean_train, add_base_target
from .features import build_features, add_target_encodings, feature_columns
from .models import BoostModel

rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main(args):
    t0 = time.time()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train, test, sample = load_data(args.data_dir)
    id_col = find_id_col(test, sample)
    test_ids = test[id_col].copy()
    print(f"train={train.shape} test={test.shape} id_col={id_col!r}")

    train = drop_leaky_columns(train); test = drop_leaky_columns(test)
    train, clean_report = clean_train(train)
    print("cleaning:", clean_report)

    train = add_base_target(train); test = add_base_target(test)

    # Features are built on train+test jointly (unsupervised stats only -> no label leakage).
    train["_is_train"] = 1; test["_is_train"] = 0
    full = pd.concat([train, test], ignore_index=True, sort=False)
    full = build_features(full)
    tr_all = full[full["_is_train"] == 1].reset_index(drop=True)
    te_all = full[full["_is_train"] == 0].reset_index(drop=True)

    # Model target: base fare (total minus known components, refund sign removed).
    # Back-transform: total = refund_sign * base_pred + component_sum
    y_raw = tr_all[C.TARGET].values
    y_mod = tr_all[C.BASE_TARGET].values
    y_mod = np.clip(y_mod, None, np.quantile(y_mod, C.TARGET_WINSOR_Q))
    sign_tr, comp_tr = tr_all["refund_sign"].values, tr_all["component_sum"].values
    sign_te, comp_te = te_all["refund_sign"].values, te_all["component_sum"].values
    to_total = lambda p, s, c: s * p + c

    # Stratify the regression target by quantile bins so every fold mirrors the distribution
    bins = pd.qcut(y_raw, q=10, labels=False, duplicates="drop")
    models = list(C.BLEND_WEIGHTS)
    oof = {m: np.zeros(len(tr_all)) for m in models}
    tst = {m: np.zeros(len(te_all)) for m in models}
    log = {"cleaning": clean_report, "folds": [], "weights": C.BLEND_WEIGHTS,
           "backend": args.backend}
    feats = None
    runs = 0

    for seed_i in range(args.seeds):
        seed = C.SEED + seed_i
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        for fold, (a, b) in enumerate(skf.split(tr_all, bins)):
            tr_f, va_f = tr_all.iloc[a].reset_index(drop=True), tr_all.iloc[b].reset_index(drop=True)
            tr_f, (va_f, te_f) = add_target_encodings(tr_f, [va_f, te_all], seed)
            if feats is None:
                feats = feature_columns(tr_f, id_col)
                print(f"{len(feats)} features")
            Xtr, Xva, Xte = tr_f[feats], va_f[feats], te_f[feats]
            ytr, yva = y_mod[a], y_mod[b]
            rec = {"seed": seed, "fold": fold}
            for m in models:
                bm = BoostModel(m, seed, args.backend).fit(Xtr, ytr, Xva, yva)
                p_va = bm.predict(Xva)
                oof[m][b] += to_total(p_va, sign_tr[b], comp_tr[b]) / args.seeds
                tst[m] += to_total(bm.predict(Xte), sign_te, comp_te) / (args.folds * args.seeds)
                # overfitting monitor: train vs validation RMSE (log space)
                sub = np.random.RandomState(0).choice(len(Xtr), min(len(Xtr), 50000), replace=False)
                rec[m] = {"best_iter": bm.best_iteration,
                          "train_rmse_base": rmse(ytr[sub], bm.predict(Xtr.iloc[sub])),
                          "val_rmse_base": rmse(yva, p_va),
                          "val_rmse_usd": rmse(y_raw[b], to_total(p_va, sign_tr[b], comp_tr[b])),
                          "params": bm.loggable_params()}
                print(f"seed{seed} fold{fold} {m:9s} val_rmse=${rec[m]['val_rmse_usd']:.3f} "
                      f"iters={bm.best_iteration} gap(base)={rec[m]['val_rmse_base']-rec[m]['train_rmse_base']:.3f}")
            log["folds"].append(rec); runs += 1

    # ---- Hybrid blend (default 80 : 10 : 10) ----
    w = C.BLEND_WEIGHTS
    blend_oof = sum(w[m] * oof[m] for m in models)
    blend_te = sum(w[m] * tst[m] for m in models)

    scores = {m: {"rmse": rmse(y_raw, oof[m]), "mae": mae(y_raw, oof[m])} for m in models}
    scores["blend_80_10_10"] = {"rmse": rmse(y_raw, blend_oof), "mae": mae(y_raw, blend_oof)}
    log["oof_scores"] = scores
    print("\nOOF scores:"); [print(f"  {k:16s} RMSE={v['rmse']:.4f} MAE={v['mae']:.4f}") for k, v in scores.items()]

    # ---- Outputs ----
    sub_cols = list(sample.columns) if sample is not None else [id_col, C.TARGET]
    submission = pd.DataFrame({sub_cols[0]: test_ids.values, sub_cols[1]: blend_te})
    submission.to_csv(out / "submission.csv", index=False)
    pd.DataFrame({"y_true": y_raw, "y_pred": blend_oof,
                  **{f"oof_{m}": oof[m] for m in models}}).join(
        tr_all[["hour", "dow", "PU", "DO", C.DIST, "duration_min", "RatecodeID", "is_refund"]]
    ).to_csv(out / "oof_predictions.csv", index=False)
    log["runtime_sec"] = round(time.time() - t0, 1); log["n_features"] = len(feats)
    log["features"] = feats
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"\nWrote {out/'submission.csv'} ({len(submission)} rows) in {log['runtime_sec']}s")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default=str(C.DEFAULT_OUT_DIR))
    ap.add_argument("--folds", type=int, default=C.N_FOLDS)
    ap.add_argument("--seeds", type=int, default=C.N_SEEDS)
    ap.add_argument("--backend", choices=["real", "sklearn"], default="real",
                    help="'sklearn' = smoke-test stand-ins when xgboost/lightgbm/catboost are unavailable")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse())
