"""
Generate eval_predictions.parquet, reorder_recommendations.csv, and anomalies.csv
to populate the FastAPI interactive dashboard.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "smart_inventory"))

from inventory_logic import engineer_features
from config import MODEL_PATH, EXPECTED_FEATURES

print("1. Loading retail history data...")
csv_path = BASE_DIR / "smart_inventory" / "data" / "retail_history.csv"
df = pd.read_csv(csv_path)
df["dt"] = pd.to_datetime(df["dt"])

# Select top 50 products across stores for snappy dashboard performance
top_products = df["product_id"].unique()[:50]
df_sub = df[df["product_id"].isin(top_products)].copy()
df_sub = df_sub.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
print(f"   Selected {len(df_sub):,} rows across {df_sub['product_id'].nunique()} products.")

print("\n2. Computing features and running XGBoost model...")
featured_df = engineer_features(df_sub, include_target=True)
valid_df = featured_df.dropna(subset=EXPECTED_FEATURES + ["target_next_day"]).reset_index(drop=True)

# Select holdout split (last 16 days)
cutoff = pd.Timestamp("2024-06-08")
eval_df = valid_df[valid_df["dt"] > cutoff].copy().reset_index(drop=True)
print(f"   Holdout evaluation records: {len(eval_df):,}")

# Predict point forecast
booster = xgb.Booster()
booster.load_model(str(MODEL_PATH))
dmat = xgb.DMatrix(eval_df[EXPECTED_FEATURES], feature_names=EXPECTED_FEATURES)
eval_df["pred_point"] = np.clip(booster.predict(dmat), 0, None)

# Group rolling std for demand uncertainty
grp = eval_df.groupby(["store_id", "product_id"], sort=False)
eval_df["sales_roll7_std"] = grp["sale_amount"].transform(lambda x: x.rolling(7, min_periods=2).std()).fillna(0.3)

# Latent demand: when stockout occurs, unconstrain demand slightly
eval_df["pred_latent"] = np.round(np.where(
    eval_df["stockout_lag_1"] > 0,
    eval_df["pred_point"] * (1.0 + eval_df["stockout_lag_1"] / 24.0 * 0.3),
    eval_df["pred_point"]
), 2)

# Quantile forecasts
eval_df["pred_q50"] = np.round(eval_df["pred_point"], 2)
eval_df["pred_q90"] = np.round(eval_df["pred_point"] + 1.28 * eval_df["sales_roll7_std"], 2)
eval_df["pred_q95"] = np.round(np.maximum(eval_df["pred_point"] + 1.645 * eval_df["sales_roll7_std"], eval_df["pred_point"] * 1.3), 2)

# Save eval_predictions.parquet
os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)
pred_parquet_path = BASE_DIR / "data" / "processed" / "eval_predictions.parquet"
keep_cols = [
    "store_id", "product_id", "dt", "sale_amount",
    "pred_point", "pred_latent", "pred_q50", "pred_q90", "pred_q95",
    "sales_roll7_std", "stock_hour6_22_cnt"
]
eval_df[keep_cols].to_parquet(pred_parquet_path, index=False)
print(f"   Saved {len(eval_df):,} predictions to {pred_parquet_path}")

print("\n3. Generating Reorder Recommendations via src/inventory.py...")
from src.inventory import generate_reorder_recommendations
recs = generate_reorder_recommendations(
    predictions_path=str(pred_parquet_path),
    config_path=str(BASE_DIR / "config" / "inventory.yaml"),
    output_path=str(BASE_DIR / "outputs" / "reorder_recommendations.csv")
)
print(f"   Generated {len(recs):,} reorder recommendations in outputs/reorder_recommendations.csv")

print("\n4. Generating Anomaly Detection Feed via src/anomaly.py...")
from src.anomaly import detect_anomalies
detect_anomalies(
    predictions_path=str(pred_parquet_path),
    output_path=str(BASE_DIR / "outputs" / "anomalies.csv")
)
print("   Generated anomalies in outputs/anomalies.csv")
print("\nAll dashboard tables generated successfully!")
