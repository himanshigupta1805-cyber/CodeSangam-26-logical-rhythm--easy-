"""Model wrappers: XGBoost / LightGBM / CatBoost with early stopping.

`backend="sklearn"` swaps in HistGradientBoosting stand-ins when real libraries are unavailable.
"""
import numpy as np
from . import config as C


class BoostModel:
    def __init__(self, name, seed, backend="real"):
        self.name, self.seed, self.backend = name, seed, backend
        self.model = None
        self.best_iteration = None
        self.params = {}

    def fit(self, Xtr, ytr, Xva, yva):
        if self.backend == "sklearn":
            from sklearn.ensemble import HistGradientBoostingRegressor as H
            cfg = {
                "xgboost": dict(max_depth=5, max_leaf_nodes=25, min_samples_leaf=30, l2_regularization=12.0, learning_rate=0.035),
                "lightgbm": dict(max_depth=6, max_leaf_nodes=31, min_samples_leaf=35, l2_regularization=15.0, learning_rate=0.035),
                "catboost": dict(max_depth=4, max_leaf_nodes=16, min_samples_leaf=25, l2_regularization=10.0, learning_rate=0.04),
            }[self.name]
            self.params = dict(max_iter=2000, early_stopping=True,
                               validation_fraction=0.12, n_iter_no_change=50,
                               random_state=self.seed, **cfg)
            self.model = H(**self.params).fit(Xtr, ytr)
            self.best_iteration = int(self.model.n_iter_)
            return self

        p = dict(C.PARAMS[self.name])
        es = C.EARLY_STOPPING_ROUNDS
        if self.name == "xgboost":
            from xgboost import XGBRegressor
            self.params = dict(p, random_state=self.seed, early_stopping_rounds=es,
                               eval_metric="rmse", n_jobs=-1)
            self.model = XGBRegressor(**self.params)
            self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            self.best_iteration = int(self.model.best_iteration)
        elif self.name == "lightgbm":
            import lightgbm as lgb
            self.params = dict(p, random_state=self.seed, n_jobs=-1)
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
                           callbacks=[lgb.early_stopping(es, verbose=False)])
            self.best_iteration = int(self.model.best_iteration_)
        elif self.name == "catboost":
            from catboost import CatBoostRegressor
            self.params = dict(p, random_seed=self.seed, early_stopping_rounds=es,
                               thread_count=-1)
            self.model = CatBoostRegressor(**self.params)
            self.model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
            self.best_iteration = int(self.model.get_best_iteration())
        else:
            raise ValueError(self.name)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def loggable_params(self):
        return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in self.params.items()}
