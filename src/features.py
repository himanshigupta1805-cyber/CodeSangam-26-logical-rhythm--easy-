"""Temporal & spatial alchemy + denoised tip & rate-regime features."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from . import config as C
from .data import first_present

TE_KEYS = ["PU", "DO", "PAIR", "HOUR_DOW"]


def build_features(df):
    """Stateless + unsupervised features. Safe to run on train+test concatenated
    (no target information is used)."""
    d = df.copy()
    pu_col = first_present(d, C.PICKUP_TS_CANDIDATES)
    do_col = first_present(d, C.DROPOFF_TS_CANDIDATES)
    pu = pd.to_datetime(d[pu_col])
    do = pd.to_datetime(d[do_col])

    raw_dur = (do - pu).dt.total_seconds() / 60.0
    d["dur_neg"] = (raw_dur < 0).astype(int)           # dropoff before pickup (~38% of rows)
    d["raw_dur_clipped"] = raw_dur.clip(-120, 240)
    d["duration_min"] = raw_dur.abs()
    d.loc[(d["duration_min"] == 0) | (d["duration_min"] > C.MAX_DURATION_MIN), "duration_min"] = np.nan
    d["dur_pos_only"] = raw_dur.where((raw_dur > 0) & (raw_dur <= C.MAX_DURATION_MIN))

    d["drop_hour"] = do.dt.hour
    d["hour"] = pu.dt.hour
    d["dow"] = pu.dt.dayofweek
    d["month"] = pu.dt.month
    d["day"] = pu.dt.day
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["is_rush"] = ((d["dow"] < 5) & (d["hour"].between(7, 9) | d["hour"].between(16, 19))).astype(int)
    d["is_night"] = ((d["hour"] >= 20) | (d["hour"] < 6)).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24.0)

    dist = d[C.DIST].where(d[C.DIST] >= 0)
    d["trip_distance_clean"] = dist
    d["log_dist"] = np.log1p(dist)
    d["sqrt_dist"] = np.sqrt(dist)
    d["zero_dist"] = (dist == 0).astype(int)
    d["short_trip"] = (dist < 0.3).astype(int)

    speed = dist / (d["duration_min"] / 60.0)
    d["speed_mph"] = speed.where(speed <= C.MAX_SPEED_MPH)
    d["min_per_mile"] = (d["duration_min"] / (dist + 0.1)).clip(0, 60)

    # payment type (string in this dataset) -> stable integer codes
    if "payment_type" in d and not pd.api.types.is_numeric_dtype(d["payment_type"]):
        order = ["Credit Card", "Cash", "Wallet", "UPI", "unknown"]
        d["payment_type"] = d["payment_type"].map({k: i for i, k in enumerate(order)}).fillna(len(order))

    d["is_card"] = (d["payment_type"] == 0).astype(int)
    d["is_cash_or_upi"] = d["payment_type"].isin([1, 3, 4]).astype(int)
    d["is_wallet"] = (d["payment_type"] == 2).astype(int)

    # Denoised tip features: Cash/UPI/unknown have $0 real tip in total_amount (their tip_amount column is Uniform(0,5) noise)
    if "tip_amount" in d:
        raw_tip = d["tip_amount"].fillna(0)
        is_cw = d["payment_type"].isin([0, 2])
        d["tip_active"] = np.where(is_cw, raw_tip, 0.0)
        d["tip_denoised"] = np.where(is_cw, np.maximum(0.0, raw_tip - 2.5), 0.0)
        d["tip_per_mile"] = d["tip_denoised"] / (dist + 1.0)
        d["log_tip_denoised"] = np.log1p(d["tip_denoised"])

    # Known fare components (given as inputs) and refund indicator
    for c in ["tolls_amount", "extra", "congestion_surcharge", "Airport_fee"]:
        if c in d:
            d[f"{c}_abs"] = d[c].fillna(0).abs()
            d[f"{c}_nonzero"] = (d[c].fillna(0) != 0).astype(int)

    d["is_refund"] = (d[C.REFUND_FLAG_COL] < 0).astype(int) if C.REFUND_FLAG_COL in d else 0
    if "component_sum" in d:
        d["component_abs"] = d["component_sum"].abs()
        d["component_per_mile"] = d["component_abs"] / (dist + 1.0)

    if "RatecodeID" in d:
        rc = d["RatecodeID"].fillna(-1)
        d["is_airport_rate"] = rc.isin([2, 3, 4]).astype(int)
        d["is_jfk_rate"] = (rc == 2).astype(int)
        d["is_newark_rate"] = (rc == 3).astype(int)
        d["is_nassau_rate"] = (rc == 4).astype(int)
        d["is_negotiated_rate"] = (rc == 5).astype(int)
        d["is_rate_99"] = (rc == 99).astype(int)
        # Expected standard NYC meter prior
        d["expected_nyc_meter"] = np.where(
            rc == 2, 70.0,
            np.where(rc == 3, 20.0 + 4.5 * dist,
            np.where(rc == 4, 3.0 + 5.2 * dist,
            3.0 + 3.8 * dist))
        )
        if "component_abs" in d:
            d["expected_card_tip"] = np.where(
                d["is_card"] == 1,
                0.20 * (d["expected_nyc_meter"] + d["component_abs"]),
                0.0
            )

    if "store_and_fwd_flag" in d:
        d["store_and_fwd_flag"] = d["store_and_fwd_flag"].map({"Y": 1, "N": 0}).fillna(-1)

    # Spatial keys kept for diagnostics & clustering, while noisy raw IDs are excluded from tree splits
    d["PU"] = d[C.PU_ZONE]
    d["DO"] = d[C.DO_ZONE]
    d["PAIR"] = d["PU"] * 10000 + d["DO"]
    d["HOUR_DOW"] = d["dow"] * 24 + d["hour"]
    d["same_zone"] = (d["PU"] == d["DO"]).astype(int)
    n = len(d)
    for k in ["PU", "DO"]:
        d[f"{k}_freq"] = d[k].map(d[k].value_counts()) / n

    for side in ["PU", "DO"]:
        prof = d.groupby(side).agg(
            dist=("trip_distance_clean", "mean"), dur=("duration_min", "mean"),
            spd=("speed_mph", "mean"), freq=(f"{side}_freq", "first"),
        ).fillna(0)
        k = min(C.N_ZONE_CLUSTERS, len(prof))
        if k >= 2:
            z = StandardScaler().fit_transform(prof.values)
            lab = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit_predict(z)
            d[f"{side}_cluster"] = d[side].map(dict(zip(prof.index, lab)))
        else:
            d[f"{side}_cluster"] = 0
    return d


_GROUP_REGISTRY = {}


def add_extra_features(d):
    """Extra engineered interaction features."""
    d = d.copy()
    dist = d["trip_distance_clean"]
    rc = d["RatecodeID"].fillna(-1)
    tl = d["tolls_amount"].fillna(0).abs()
    d["toll_pos"] = (tl > 0).astype(int)
    d["rate_flat"] = rc.isin([2, 3, 4]).astype(int)
    d["rate_neg"] = rc.isin([5, 99]).astype(int)
    d["dist_x_flat"] = dist * d["rate_flat"]
    d["dist_x_neg"] = dist * d["rate_neg"]
    d["dist_x_wallet"] = dist * d["is_wallet"]
    d["rate_x_tollpos"] = rc.clip(lower=1) * d["toll_pos"]
    return d, {}


# Drop raw noisy uniform-random zone IDs and raw corrupted tip_amount so trees use denoised tip features
NON_FEATURES = {
    C.TARGET, C.BASE_TARGET, "refund_sign", "PAIR",
    "PULocationID", "DOLocationID", "PU", "DO", "tip_amount",
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "lpep_pickup_datetime",
    "lpep_dropoff_datetime", "pickup_datetime", "dropoff_datetime",
    "id", "ID", "trip_id", "Id", "_is_train",
}


def feature_columns(df, id_col=None):
    drop = set(NON_FEATURES) | ({id_col} if id_col else set())
    cols = [c for c in df.columns if c not in drop
            and not c.startswith("te_") and pd.api.types.is_numeric_dtype(df[c])]
    return cols
