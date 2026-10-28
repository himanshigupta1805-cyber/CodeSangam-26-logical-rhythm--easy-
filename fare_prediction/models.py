from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class TripAggregateEncoder(BaseEstimator, TransformerMixin):
    """Add fold-safe route/category frequencies and smoothed target aggregates."""
    def __init__(self, columns=None, smoothing=20.0, inner_folds=5, random_state=42):
        self.columns = columns
        self.smoothing = smoothing
        self.inner_folds = inner_folds
        self.random_state = random_state

    @staticmethod
    def _keys(series):
        return series.astype("string").fillna("__MISSING__").astype(str)

    def _fit_maps(self, X, y):
        prior = float(np.mean(y))
        maps = {}
        for col in self.columns_:
            keys = self._keys(X[col])
            stats = pd.DataFrame({"key": keys.to_numpy(), "target": np.asarray(y, dtype=float)}).groupby("key")["target"].agg(["mean", "count"])
            means = ((stats["mean"] * stats["count"] + prior * self.smoothing) / (stats["count"] + self.smoothing)).to_dict()
            maps[col] = {"frequency": (stats["count"] / len(X)).to_dict(), "target_mean": means}
        return prior, maps

    def _append(self, X, prior, maps):
        result = X.copy()
        for col in self.columns_:
            keys = self._keys(X[col])
            stats = maps.get(col, {})
            result[f"frequency__{col}"] = keys.map(stats.get("frequency", {})).fillna(0.0).to_numpy()
            result[f"target_mean__{col}"] = keys.map(stats.get("target_mean", {})).fillna(prior).to_numpy()
        return result

    def fit(self, X, y):
        defaults = ["PULocationID", "DOLocationID", "route_pair", "VendorID", "RatecodeID", "payment_type", "store_and_fwd_flag"]
        self.columns_ = [c for c in (self.columns or defaults) if c in X.columns]
        self.prior_, self.maps_ = self._fit_maps(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, y).transform(X)
        defaults = ["PULocationID", "DOLocationID", "route_pair", "VendorID", "RatecodeID", "payment_type", "store_and_fwd_flag"]
        self.columns_ = [c for c in (self.columns or defaults) if c in X.columns]
        target = np.asarray(y, dtype=float)
        names = [f"{prefix}__{col}" for col in self.columns_ for prefix in ("frequency", "target_mean")]
        encoded = {name: np.full(len(X), np.nan, dtype=float) for name in names}
        inner = min(self.inner_folds, len(X))
        if inner >= 2:
            for fit_idx, val_idx in KFold(n_splits=inner, shuffle=True, random_state=self.random_state).split(X):
                prior, maps = self._fit_maps(X.iloc[fit_idx], target[fit_idx])
                part = self._append(X.iloc[val_idx], prior, maps)
                for name in names:
                    encoded[name][val_idx] = part[name].to_numpy()
        self.prior_, self.maps_ = self._fit_maps(X, target)
        result = X.copy()
        for name, values in encoded.items():
            result[name] = values if inner >= 2 else self._append(X, self.prior_, self.maps_)[name].to_numpy()
        return result

    def transform(self, X):
        return self._append(X, self.prior_, self.maps_)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Fit per-column quantile bounds on training data and cap outliers."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)


def build_preprocessor(frame):
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    num_pipe = Pipeline([("clip", QuantileClipper()),
                         ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", RobustScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
    return ColumnTransformer([("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)],
                             remainder="drop", sparse_threshold=0.0)


def available_models(seed=42):
    models = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07,
                                                                  l2_regularization=1.0, random_state=seed),
        "extra_trees": ExtraTreesRegressor(n_estimators=350, min_samples_leaf=2, max_features=0.9,
                                            n_jobs=1, random_state=seed),
    }
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(n_estimators=700, learning_rate=0.04, num_leaves=31,
                                            verbosity=-1, n_jobs=1, random_state=seed)
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(n_estimators=700, max_depth=8, learning_rate=0.04,
                                          subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
                                          n_jobs=1, random_state=seed)
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        models["catboost"] = CatBoostRegressor(iterations=700, depth=8, learning_rate=0.04,
                                                verbose=False, random_seed=seed, allow_writing_files=False)
    except ImportError:
        pass
    return models


def make_pipeline(model, frame):
    aggregate = TripAggregateEncoder(random_state=getattr(model, "random_state", 42) or 42)
    aggregate_columns = aggregate.columns or ["PULocationID", "DOLocationID", "route_pair", "VendorID", "RatecodeID", "payment_type", "store_and_fwd_flag"]
    enriched = frame.copy()
    for column in aggregate_columns:
        if column in enriched:
            enriched[f"frequency__{column}"] = 0.0
            enriched[f"target_mean__{column}"] = 0.0
    return Pipeline([("aggregates", aggregate), ("preprocess", build_preprocessor(enriched)), ("model", model)])

