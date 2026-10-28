from __future__ import annotations

import numpy as np
import pandas as pd

PICKUP_CANDIDATES = ("tpep_pickup_datetime", "pickup_datetime", "pickup_time", "pickup_timestamp")
DROPOFF_CANDIDATES = ("tpep_dropoff_datetime", "dropoff_datetime", "dropoff_time", "dropoff_timestamp")
COORDINATE_PAIRS = (("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"),
                    ("pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"))


def engineer_features(frame: pd.DataFrame, pickup_col: str | None = None,
                      dropoff_col: str | None = None) -> pd.DataFrame:
    """Derive trip time, route-interaction and geographic distance features."""
    data = frame.copy()
    pickup = pickup_col if pickup_col in data.columns else next((c for c in PICKUP_CANDIDATES if c in data), None)
    dropoff = dropoff_col if dropoff_col in data.columns else next((c for c in DROPOFF_CANDIDATES if c in data), None)
    if pickup:
        dt = pd.to_datetime(data[pickup], errors="coerce")
        data["pickup_hour"] = dt.dt.hour
        data["pickup_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        data["pickup_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        data["pickup_day"] = dt.dt.day
        data["pickup_month"] = dt.dt.month
        data["pickup_weekday"] = dt.dt.dayofweek
        data["pickup_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
        data["pickup_rush_hour"] = dt.dt.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
        if dropoff:
            end = pd.to_datetime(data[dropoff], errors="coerce")
            duration = (end - dt).dt.total_seconds().div(60)
            data["trip_duration_minutes"] = duration.where(duration.between(0, 24 * 60))
        data = data.drop(columns=[pickup], errors="ignore")
    if dropoff:
        data = data.drop(columns=[dropoff], errors="ignore")
    if "PULocationID" in data and "DOLocationID" in data:
        data["route_pair"] = data["PULocationID"].astype("string").fillna("NA") + "_" + data["DOLocationID"].astype("string").fillna("NA")
    for col in ("VendorID", "RatecodeID", "PULocationID", "DOLocationID"):
        if col in data:
            data[col] = data[col].astype("string")
    pair = next((p for p in COORDINATE_PAIRS if all(c in data for c in p)), None)
    if pair:
        plat, plon, dlat, dlon = (pd.to_numeric(data[c], errors="coerce") for c in pair)
        lat1, lat2 = np.radians(plat), np.radians(dlat)
        dlat_r, dlon_r = lat2 - lat1, np.radians(dlon - plon)
        a = np.sin(dlat_r / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_r / 2) ** 2
        data["haversine_distance_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
    # Older sklearn imputers expect ordinary object columns with np.nan, not pandas.NA.
    for column in data.select_dtypes(include=["string"]).columns:
        data[column] = data[column].astype(object).where(data[column].notna(), np.nan)
    return data
