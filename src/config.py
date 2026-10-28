"""Central configuration for the Portal Fare Prophecy pipeline."""
from pathlib import Path

TARGET = "total_amount"

# Column-name candidates (the pipeline auto-detects which one exists).
PICKUP_TS_CANDIDATES = ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"]
DROPOFF_TS_CANDIDATES = ["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"]
ID_CANDIDATES = ["id", "ID", "trip_id", "Id"]
PU_ZONE = "PULocationID"
DO_ZONE = "DOLocationID"
DIST = "trip_distance"

# Columns that are *components of* total_amount (surcharges, tip, tolls ...).
# In THIS contest they are provided in test.csv, so they are legitimate inputs.
# The model predicts the remaining "base fare" and adds the known components back:
#     total = refund_sign * base_fare + sum(components)
FARE_COMPONENT_COLS = [
    "extra", "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
    "congestion_surcharge", "airport_fee", "Airport_fee",
]
# Refund/reversal trips carry negative surcharges and a negative total_amount.
# improvement_surcharge < 0 identifies them perfectly in train (78/78).
REFUND_FLAG_COL = "improvement_surcharge"
BASE_TARGET = "base_fare"
DROP_FARE_COMPONENTS = False   # set True for a "pure" no-component variant

# Hybrid blend. Order: XGBoost / LightGBM / CatBoost  ->  80 : 10 : 10
BLEND_WEIGHTS = {"xgboost": 0.80, "lightgbm": 0.10, "catboost": 0.10}

# Cross-validation / anti-overfitting
N_FOLDS = 5
N_SEEDS = 1                 # raise to 3+ on Kaggle for seed-averaging
SEED = 42
EARLY_STOPPING_ROUNDS = 200
TE_SMOOTHING = 20           # smoothing strength for target encoding
TE_INNER_FOLDS = 5          # inner OOF folds for leak-free target encoding

# Cleaning
MAX_DURATION_MIN = 360      # abs duration above this is treated as missing
MAX_SPEED_MPH = 80
TARGET_WINSOR_Q = 1.0       # 1.0 = no winsorising (large fares are real: long/airport trips)
N_ZONE_CLUSTERS = 8

# Model hyper-parameters (regularised; all logged to outputs/run_log.json)
PARAMS = {
    "xgboost": dict(
        n_estimators=5000, learning_rate=0.03, max_depth=7, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.5,
        gamma=0.0, tree_method="hist",
    ),
    "lightgbm": dict(
        n_estimators=5000, learning_rate=0.03, num_leaves=63, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
        reg_alpha=0.5, verbose=-1,
    ),
    "catboost": dict(
        iterations=5000, learning_rate=0.05, depth=8, l2_leaf_reg=5.0,
        loss_function="RMSE", verbose=0,
    ),
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "outputs"
