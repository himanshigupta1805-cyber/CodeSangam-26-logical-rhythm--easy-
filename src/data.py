"""Loading and cleaning (scrying & data hygiene)."""
import numpy as np
import pandas as pd
from . import config as C


def first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data(data_dir):
    data_dir = __import__("pathlib").Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = data_dir / "sample_submission.csv"
    sample = pd.read_csv(sample) if sample.exists() else None
    return train, test, sample


def find_id_col(test, sample):
    if sample is not None:
        return sample.columns[0]
    return first_present(test, C.ID_CANDIDATES) or test.columns[0]


def drop_leaky_columns(df):
    if not C.DROP_FARE_COMPONENTS:
        return df
    return df.drop(columns=[c for c in C.FARE_COMPONENT_COLS if c in df.columns])


def add_base_target(df):
    """Adds refund sign, deterministic surcharge sum and the 'base_fare' modelling target.

    Note: `tip_amount` contains Uniform(0, 5) additive noise and is strictly $0 in
    `total_amount` for Cash/UPI/unknown trips. We therefore exclude `tip_amount` from
    `component_sum` and instead pass denoised tip features to the linear prior & GBMs.
    We also include the implicit $0.50 MTA tax whenever improvement_surcharge != 0.
    """
    d = df.copy()
    comps = [c for c in C.FARE_COMPONENT_COLS if c in d.columns]
    d["refund_sign"] = np.where(d[C.REFUND_FLAG_COL] < 0, -1, 1)
    mta_est = np.where(d[C.REFUND_FLAG_COL].fillna(0) != 0, 0.50 * d["refund_sign"], 0.0)
    d["component_sum"] = d[comps].fillna(0).sum(axis=1) + mta_est
    if C.TARGET in d.columns:
        d[C.BASE_TARGET] = d["refund_sign"] * (d[C.TARGET] - d["component_sum"])
    return d


def clean_train(train):
    """Row-level hygiene for TRAINING data only (test rows are never dropped)."""
    n0 = len(train)
    d = train.drop_duplicates()
    report = {"rows_in": n0, "dup_removed": n0 - len(d)}
    bad = d[C.TARGET].isna() | (d[C.TARGET] == 0)
    report["missing_or_zero_total"] = int(bad.sum())
    d = d[~bad].reset_index(drop=True)

    pu_col = first_present(d, C.PICKUP_TS_CANDIDATES)
    do_col = first_present(d, C.DROPOFF_TS_CANDIDATES)
    if pu_col and do_col:
        dur = (pd.to_datetime(d[do_col]) - pd.to_datetime(d[pu_col])).dt.total_seconds() / 60
        report["dropoff_before_pickup (kept, flagged)"] = int((dur < 0).sum())
        report["abs_duration_gt_max (set NaN)"] = int((dur.abs() > C.MAX_DURATION_MIN).sum())
    report["refund_rows (kept, sign-handled)"] = int((d[C.TARGET] < 0).sum())
    report["rows_out"] = len(d)
    return d, report
