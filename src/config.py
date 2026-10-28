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

# Deterministic fare components (tip_amount excluded from direct sum because it has Uniform(0,5) noise;
# instead, denoised tip features are passed to the linear prior and gradient boosters).
FARE_COMPONENT_COLS = [
    "extra", "mta_tax", "tolls_amount", "improvement_surcharge",
    "congestion_surcharge", "airport_fee", "Airport_fee",
]
REFUND_FLAG_COL = "improvement_surcharge"
BASE_TARGET = "base_fare"
DROP_FARE_COMPONENTS = False   # set True for a "pure" no-component variant

# Hybrid blend. Order: XGBoost / LightGBM / CatBoost
BLEND_WEIGHTS = {"xgboost": 0.45, "lightgbm": 0.30, "catboost": 0.25}

# Cross-validation / anti-overfitting
N_FOLDS = 5
N_SEEDS = 3                 # 3 seeds for multi-seed variance reduction
SEED = 42
EARLY_STOPPING_ROUNDS = 150
LINEAR_ALPHA = 3.0          # Ridge strength of the linear prior
TE_SMOOTHING = 25           # smoothing strength for target encoding
TE_INNER_FOLDS = 5          # inner OOF folds for leak-free target encoding

# Cleaning
MAX_DURATION_MIN = 360      # abs duration above this is treated as missing
MAX_SPEED_MPH = 80
TARGET_WINSOR_Q = 1.0       # 1.0 = no winsorising (large fares are real: long/airport trips)
N_ZONE_CLUSTERS = 6

# Model hyper-parameters (strong anti-overfitting regularisation; logged to outputs/run_log.json)
PARAMS = {
    "xgboost": dict(
        n_estimators=4000, learning_rate=0.025, max_depth=5, min_child_weight=15,
        subsample=0.75, colsample_bytree=0.75, reg_lambda=10.0, reg_alpha=2.0,
        gamma=0.2, tree_method="hist",
    ),
    "lightgbm": dict(
        n_estimators=4000, learning_rate=0.025, num_leaves=31, max_depth=6,
        min_child_samples=35, subsample=0.75, subsample_freq=1,
        colsample_bytree=0.75, reg_lambda=10.0, reg_alpha=2.0, verbose=-1,
    ),
    "catboost": dict(
        iterations=4000, learning_rate=0.03, depth=6, l2_leaf_reg=10.0,
        loss_function="RMSE", verbose=0,
    ),
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "outputs"
