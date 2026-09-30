"""
Configuration module for Retail Store Smart Inventory Management System.
Single source of truth for constants, paths, thresholds, and metadata.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_stock_aware.json"
METADATA_PATH = BASE_DIR / "model_features.json"
SAMPLE_DATA_PATH = BASE_DIR / "data" / "retail_history.csv"

# Dataset Specification
DATASET_REPO_ID = "Dingdong-Inc/FreshRetailNet-50K"
REQUIRED_COLUMNS = [
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt"
]

# Leakage-Safe Feature Contract (Strict Order)
EXPECTED_FEATURES = [
    "lag_1",
    "lag_7",
    "rolling_7",
    "stockout_lag_1",
    "stockout_lag_7",
    "stockout_rolling_7"
]
TARGET_COLUMN = "target_next_day"
PREDICTION_HORIZON = "next_day"

# Inventory Decision-Support Heuristics & Assumptions
# Project heuristics: not optimized Economic Order Quantities (EOQ)
SAFETY_FACTOR = 0.50
PLANNING_HORIZON_DAYS = 7
MIN_REQUIRED_DAYS = 8

# Stockout Risk Assessment Thresholds (Based solely on past 7-day stockout hours: stockout_rolling_7)
# Fixed heuristic thresholds
RISK_THRESHOLD_MEDIUM = 8.0   # hours
RISK_THRESHOLD_HIGH = 24.0    # hours

# Quantile-based configurable alternative (70th and 90th empirical training percentiles)
RISK_QUANTILE_MEDIUM = 29.0   # hours (70th percentile)
RISK_QUANTILE_HIGH = 48.0     # hours (90th percentile)

# Holdout Base Rates for Risk Comparison
OVERALL_STOCKOUT_BASE_RATE = 43.58     # % of holdout rows experiencing actual stockout on day t
OVERALL_MEAN_STOCKOUT_HOURS = 3.29     # mean stockout hours across all holdout observations

RISK_LEVEL_LOW = "LOW"
RISK_LEVEL_MEDIUM = "MEDIUM"
RISK_LEVEL_HIGH = "HIGH"

ACTION_LOW = "NO IMMEDIATE ACTION"
ACTION_MEDIUM = "MONITOR / REPLENISH"
ACTION_HIGH = "REPLENISH NOW"

# Additional Model Artifacts
MODEL_V2_PATH = BASE_DIR / "xgboost_v2_features.json"

# Benchmark Reference Results (Validated 150-product development subset, 214,560 holdout observations)
# Mean Holdout Target Sales: 1.1277 units
BENCHMARK_RESULTS = {
    # Backward compatibility aliases
    "baseline_mae": 0.5746,
    "baseline_rmse": 0.9226,
    "mae_reduction_percent": 23.72,
    # Baselines
    "stale_lag1_mae": 0.5746,
    "stale_lag1_rmse": 0.9226,
    "stale_lag1_wape": 50.95,
    "true_prev_day_mae": 0.5152,
    "true_prev_day_rmse": 0.8282,
    "true_prev_day_wape": 45.69,
    "rolling_7_mae": 0.4469,
    "rolling_7_rmse": 0.7011,
    "rolling_7_wape": 39.63,
    "lag_7_8day_mae": 0.5652,
    "lag_7_8day_rmse": 0.9090,
    "lag_7_8day_wape": 50.12,
    "weekday_aligned_lag_mae": 0.5308,
    "weekday_aligned_lag_rmse": 0.8470,
    "weekday_aligned_lag_wape": 47.07,
    # Models (Frozen v1)
    "rf_mae": 0.4396,
    "rf_rmse": 0.6848,
    "rf_wape": 38.98,
    "xgboost_mae": 0.4383,
    "xgboost_rmse": 0.6829,
    "xgboost_wape": 38.87,
    # v2 Experiment (with day t and shift(6))
    "xgboost_v2_mae": 0.4194,
    "xgboost_v2_rmse": 0.6594,
    "xgboost_v2_wape": 37.19,
    # Relative gains of deployed XGBoost v1
    "gain_vs_stale_lag1_percent": 23.72,
    "gain_vs_true_prev_day_percent": 14.92,
    "gain_vs_rolling_7_percent": 1.91,
    "gain_vs_lag7_8day_percent": 22.45,
    "gain_vs_weekday_aligned_percent": 17.43,
    "gain_vs_rf_percent": 0.28,  # 0.28% unrounded (0.30% from 0.0013 / 0.4396)
    # v2 relative gain over v1
    "v2_gain_over_v1_percent": 4.31,
    "holdout_mean_sales": 1.1277
}

FEATURE_IMPORTANCES_REF = {
    "rolling_7": 79.14,
    "lag_7": 11.04,  # Note: 8-day lag persistence (t-7 to t+1); does not align to same weekday
    "lag_1": 4.37,
    "stockout_lag_7": 2.17,
    "stockout_rolling_7": 2.00,
    "stockout_lag_1": 1.28
}

# Golden Test Values (Store 0, Product 4 cutoff test series)
GOLDEN_VALUES = {
    "store_id": 0,
    "product_id": 4,
    "features": [0.0, 0.9, 0.87142855, 0.0, 0.0, 6.0],
    "predicted_next_day_demand": 0.8284,
    "recommended_stock_coverage": 6.2343,
    "risk_level": "LOW",
    "action_text": "NO IMMEDIATE ACTION",
    "recent_stockout_hours": 6.0
}

