"""Synthetic NYC-taxi-like data for SMOKE-TESTING the pipeline only (NOT real data)."""
import numpy as np, pandas as pd
from pathlib import Path

def make(n, rs, with_target=True):
    PU = rs.choice(np.arange(1, 80), n, p=np.r_[np.linspace(3, 1, 79)] / np.linspace(3, 1, 79).sum())
    DO = rs.choice(np.arange(1, 80), n)
    start = pd.Timestamp("2025-01-01") + pd.to_timedelta(rs.randint(0, 60 * 24 * 90, n), unit="m")
    dist = np.clip(np.abs(PU - DO) * 0.15 + rs.gamma(2, 0.8, n), 0.05, None)
    hour = start.hour.values
    rush = ((start.dayofweek < 5) & (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19)))).astype(int)
    speed = np.clip(rs.normal(16 - 6 * rush, 3, n), 3, 45)
    dur = dist / speed * 60 + rs.gamma(2, 1.5, n)
    rate = rs.choice([1, 1, 1, 1, 2, 3, 5], n); pay = rs.choice([1, 2, 3, 4], n, p=[.7, .25, .03, .02])
    d = pd.DataFrame({"VendorID": rs.choice([1, 2], n), "tpep_pickup_datetime": start,
        "tpep_dropoff_datetime": start + pd.to_timedelta(dur, unit="m"),
        "passenger_count": rs.choice([1, 1, 1, 2, 3, 4, 5], n).astype(float), "trip_distance": dist.round(2),
        "RatecodeID": rate.astype(float), "store_and_fwd_flag": rs.choice(["N", "N", "N", "Y"], n),
        "PULocationID": PU, "DOLocationID": DO, "payment_type": pay})
    if with_target:
        fare = 3 + 2.7 * dist + 0.5 * dur + 1.0 * rush + 2.5 * (hour >= 20) + 12 * (rate == 2) + 6 * (rate == 3) \
               + 0.04 * (PU % 7) * dist + rs.normal(0, 1.2 + 0.05 * dist, n)
        tip = np.where(pay == 1, fare * rs.uniform(0, .25, n), 0)
        d["total_amount"] = (fare + 0.5 + 1.0 + tip + rs.choice([0, 0, 0, 6.55], n)).round(2)
    return d

if __name__ == "__main__":
    out = Path("data"); out.mkdir(exist_ok=True); rs = np.random.RandomState(7)
    tr = make(30000, rs); te = make(8000, rs, with_target=False)
    # inject real-world dirt
    bad = rs.choice(len(tr), 150, replace=False); tr.loc[bad[:50], "total_amount"] = -tr.loc[bad[:50], "total_amount"]
    tr.loc[bad[50:80], "total_amount"] = 0; tr.loc[bad[80:110], "trip_distance"] = 0
    tr.loc[bad[110:], "passenger_count"] = np.nan; te.loc[te.sample(100, random_state=1).index, "passenger_count"] = np.nan
    tr.insert(0, "id", np.arange(len(tr))); te.insert(0, "id", np.arange(len(tr), len(tr) + len(te)))
    tr.to_csv(out / "train.csv", index=False); te.to_csv(out / "test.csv", index=False)
    pd.DataFrame({"id": te["id"], "total_amount": 0.0}).to_csv(out / "sample_submission.csv", index=False)
    print("synthetic data written", tr.shape, te.shape)
