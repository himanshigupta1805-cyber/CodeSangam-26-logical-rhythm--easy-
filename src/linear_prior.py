"""Linear fare prior with denoised tip and Wallet regime modeling.

Trees cannot extrapolate: on this data the largest fares (rate codes 4/5, very long trips)
sit outside the range the leaves have seen, so tree-only models under-predict them and
those few rows carry ~70% of the squared error. A small, heavily-constrained Ridge model
(distance slope per rate-code regime, tolls, denoised card-tip, and Wallet regime)
captures the broad structure and extrapolates linearly; the boosters then learn only
the *residual*:

    base_pred = linear_prior(x) + boosted_residual(x)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

RATE_REGIMES = [2, 3, 4, 5, 99]


def design_matrix(d):
    rc = d["RatecodeID"].fillna(-1).values
    dist = d["trip_distance_clean"].fillna(0).values
    is_card = (d["payment_type"] == 0).values.astype(float)
    is_wallet = (d["payment_type"] == 2).values.astype(float)
    is_cw = ((is_card > 0) | (is_wallet > 0)).astype(float)

    raw_tip = d["tip_amount"].fillna(0).values
    tip_act = raw_tip * is_cw
    tip_clean = np.maximum(0.0, tip_act - 2.5)

    X = {
        "dist": dist,
        "zero_dist": (dist == 0).astype(float),
        "is_card": is_card,
        "is_wallet": is_wallet,
        "dist_x_wallet": dist * is_wallet,
        "dist_x_card": dist * is_card,
        "tip_act": tip_act,
        "tip_clean": tip_clean,
        "tip_x_wallet": tip_act * is_wallet,
    }
    for k in RATE_REGIMES:
        m = (rc == k).astype(float)
        X[f"rate_{k}"] = m
        X[f"dist_x_rate_{k}"] = dist * m
        X[f"zero_dist_x_rate_{k}"] = (dist == 0).astype(float) * m
        X[f"tip_x_rate_{k}"] = tip_act * m

    tolls = np.abs(d["tolls_amount"].fillna(0).values)
    X["tolls"] = tolls
    X["tolls_pos"] = (tolls > 0).astype(float)
    X["airport_fee"] = np.abs(d["Airport_fee"].fillna(0).values)
    return pd.DataFrame(X, index=d.index)


def fit_linear_prior(train_df, target_col, alpha=3.0):
    return Ridge(alpha=alpha).fit(design_matrix(train_df), train_df[target_col].values)


def predict_linear_prior(model, df):
    return model.predict(design_matrix(df))
