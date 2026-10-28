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
    """Adds refund sign, known-component sum and the 'base_fare' modelling target.

    base_fare = refund_sign * (total_amount - sum(components)); components are known
    inputs. Works for test too (without the target) via `component_sum`/`refund_sign`.
    """
    d = df.copy()
    comps = [c for c in C.FARE_COMPONENT_COLS if c in d.columns]
    d["component_sum"] = d[comps].fillna(0).sum(axis=1)
    d["refund_sign"] = np.where(d[C.REFUND_FLAG_COL] < 0, -1, 1)
    if C.TARGET in d.columns:
        d[C.BASE_TARGET] = d["refund_sign"] * (d[C.TARGET] - d["component_sum"])
    return d


def clean_train(train):
    """Row-level hygiene for TRAINING data only (test rows are never dropped).

    EDA findings that shape the rules:
      * ~38% of rows have dropoff < pickup (same in test) -> this is a data-quality
        pattern of the contest, NOT something to delete. It is handled as a feature
        (abs duration + `dur_neg` flag) instead of dropping 38% of the data.
      * negative totals are legitimate refund rows (also present in test) -> kept,
        handled by `refund_sign`.
    Only truly unusable rows are removed: duplicates, missing target, total_amount == 0.
    """
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
