"""Temporal & spatial alchemy + leak-free target encoding."""
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

    raw_dur = (do - pu).dt.total_seconds() / 60
    d["dur_neg"] = (raw_dur < 0).astype(int)           # dropoff before pickup (≈38% of rows)
    d["duration_min"] = raw_dur.abs()
    d.loc[(d["duration_min"] == 0) | (d["duration_min"] > C.MAX_DURATION_MIN), "duration_min"] = np.nan
    d["drop_hour"] = do.dt.hour
    d["hour"] = pu.dt.hour
    d["dow"] = pu.dt.dayofweek
    d["month"] = pu.dt.month
    d["day"] = pu.dt.day
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["is_rush"] = ((d["dow"] < 5) & (d["hour"].between(7, 9) | d["hour"].between(16, 19))).astype(int)
    d["is_night"] = ((d["hour"] >= 22) | (d["hour"] <= 5)).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)

    dist = d[C.DIST].where(d[C.DIST] >= 0)
    d["trip_distance_clean"] = dist
    d["log_dist"] = np.log1p(dist)
    speed = dist / (d["duration_min"] / 60)
    d["speed_mph"] = speed.where(speed <= C.MAX_SPEED_MPH)
    d["min_per_mile"] = d["duration_min"] / (dist + 0.1)

    # payment type (string in this dataset) -> stable integer codes
    if "payment_type" in d and not pd.api.types.is_numeric_dtype(d["payment_type"]):
        order = ["Credit Card", "Cash", "Wallet", "UPI", "unknown"]
        d["payment_type"] = d["payment_type"].map({k: i for i, k in enumerate(order)}).fillna(len(order))

    # Known fare components (given as inputs) and refund indicator
    for c in ["tip_amount", "tolls_amount", "extra", "congestion_surcharge"]:
        if c in d:
            d[f"{c}_nonzero"] = (d[c].fillna(0) != 0).astype(int)
    d["is_refund"] = (d[C.REFUND_FLAG_COL] < 0).astype(int) if C.REFUND_FLAG_COL in d else 0
    if "tip_amount" in d:
        d["tip_per_mile"] = d["tip_amount"] / (dist + 1.0)
    if "component_sum" in d:
        d["component_per_mile"] = d["component_sum"] / (dist + 1.0)
    d["is_airport_rate"] = d["RatecodeID"].isin([2, 3, 4]).astype(int) if "RatecodeID" in d else 0

    # Missing-value indicators for the fields most likely to be NaN
    for c in ["passenger_count", "RatecodeID", "store_and_fwd_flag", "payment_type"]:
        if c in d:
            d[f"{c}_isna"] = d[c].isna().astype(int)
    if "store_and_fwd_flag" in d:
        d["store_and_fwd_flag"] = d["store_and_fwd_flag"].map({"Y": 1, "N": 0})

    # Spatial: zone keys, frequency encodings, pair frequency
    d["PU"] = d[C.PU_ZONE]
    d["DO"] = d[C.DO_ZONE]
    d["PAIR"] = d["PU"] * 10000 + d["DO"]
    d["HOUR_DOW"] = d["dow"] * 24 + d["hour"]
    d["same_zone"] = (d["PU"] == d["DO"]).astype(int)
    n = len(d)
    for k in ["PU", "DO", "PAIR"]:
        d[f"{k}_freq"] = d[k].map(d[k].value_counts()) / n
    d["PU_x_hour_freq"] = d.groupby(["PU", "hour"])["PU"].transform("size") / n

    # Spatial clustering: group zones by unsupervised behaviour profile
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
    d["cluster_pair"] = d["PU_cluster"] * 100 + d["DO_cluster"]
    return d


# ---------- Target encoding (nested, leak-free) ----------
def _enc(keys, y, m):
    prior = y.mean()
    g = pd.DataFrame({"k": keys.values, "y": y.values}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + prior * m) / (g["count"] + m)), prior


def _apply(keys, enc_prior):
    enc, prior = enc_prior
    return keys.map(enc).fillna(prior).values


def te_targets(base, dist):
    """Two encoding targets built from the base fare: log fare and fare-per-mile."""
    return {
        "te_logfare": np.log1p(base.clip(lower=0)),
        "te_fpm": (base / (dist.fillna(0) + 1.0)).clip(-5, 50),
    }


def add_target_encodings(train_df, apply_dfs, seed):
    """Encode `TE_KEYS` using ONLY `train_df` labels.

    * rows in `train_df` get inner out-of-fold encodings (no row sees its own label)
    * rows in each of `apply_dfs` (validation / test) use stats from the full `train_df`
    Returns (train_df_enc, [apply_dfs_enc...]).
    """
    tr = train_df.copy()
    outs = [a.copy() for a in apply_dfs]
    targets = te_targets(tr[C.BASE_TARGET], tr[C.DIST])
    inner = KFold(C.TE_INNER_FOLDS, shuffle=True, random_state=seed)
    for tname, y in targets.items():
        y = y.reset_index(drop=True)
        for key in TE_KEYS:
            col = f"{tname}_{key}"
            keys = tr[key].reset_index(drop=True)
            oof = np.zeros(len(tr))
            for a, b in inner.split(keys):
                ep = _enc(keys.iloc[a], y.iloc[a], C.TE_SMOOTHING)
                oof[b] = _apply(keys.iloc[b], ep)
            tr[col] = oof
            full = _enc(keys, y, C.TE_SMOOTHING)
            for o in outs:
                o[col] = _apply(o[key], full)
    return tr, outs


NON_FEATURES = {
    C.TARGET, C.BASE_TARGET, "refund_sign", "PAIR",  # PAIR kept only through freq/TE; raw id is high-cardinality noise
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "lpep_pickup_datetime",
    "lpep_dropoff_datetime", "pickup_datetime", "dropoff_datetime",
    "id", "ID", "trip_id", "Id", "_is_train",
}


def feature_columns(df, id_col=None):
    drop = set(NON_FEATURES) | ({id_col} if id_col else set())
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return cols
