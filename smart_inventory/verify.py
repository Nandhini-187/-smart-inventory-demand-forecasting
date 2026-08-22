#!/usr/bin/env python3
"""
Verification Script for Retail Store Smart Inventory Management System.
Tests model loading, HuggingFace dataset integration, feature contract, and inference.
"""

import sys
import json
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import xgboost as xgb
from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_stock_aware.json"
METADATA_PATH = BASE_DIR / "model_features.json"

REQUIRED_COLUMNS = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]
EXPECTED_FEATURES = [
    "lag_1", "lag_7", "rolling_7",
    "stockout_lag_1", "stockout_lag_7", "stockout_rolling_7"
]

def main():
    print("=" * 60)
    print("SMART INVENTORY SYSTEM - HUGGING FACE VERIFICATION")
    print("=" * 60)

    # 1. Test Metadata Loading
    print("\n1. Testing Metadata Loading...")
    if not METADATA_PATH.exists():
        print(f"❌ FAIL: Metadata file missing at {METADATA_PATH}")
        sys.exit(1)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    features = metadata.get("features", [])
    if features != EXPECTED_FEATURES:
        print(f"❌ FAIL: Feature contract mismatch.\nExpected: {EXPECTED_FEATURES}\nGot: {features}")
        sys.exit(1)
    print("   ✓ Metadata & feature contract verified.")

    # 2. Test XGBoost Model Loading
    print("\n2. Testing XGBoost Model Loading...")
    if not MODEL_PATH.exists():
        print(f"❌ FAIL: Model artifact missing at {MODEL_PATH}")
        sys.exit(1)
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    print("   ✓ XGBoost Booster loaded successfully.")

    # 3. Test Direct Hugging Face Dataset Loading
    print("\n3. Testing Direct HuggingFace Dataset Loading (Dingdong-Inc/FreshRetailNet-50K)...")
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    if "train" not in dataset:
        print("❌ FAIL: 'train' split missing in HuggingFace dataset.")
        sys.exit(1)
    train_ds = dataset["train"]
    print(f"   ✓ Loaded FreshRetailNet-50K train split with {len(train_ds):,} observations.")

    # 4. Test Zero-Copy PyArrow Filtering
    print("\n4. Testing PyArrow Table Zero-Copy Filtering...")
    table = train_ds.data.table
    stores = sorted(pc.unique(table["store_id"]).to_pylist())
    if not stores:
        print("❌ FAIL: No store IDs found in PyArrow table.")
        sys.exit(1)
    test_store = stores[0]
    
    store_table = table.filter(pc.field("store_id") == test_store)
    products = sorted(pc.unique(store_table["product_id"]).to_pylist())
    if not products:
        print("❌ FAIL: No product IDs found for store.")
        sys.exit(1)
    test_product = products[0]
    
    expr = (pc.field("store_id") == test_store) & (pc.field("product_id") == test_product)
    series_table = table.filter(expr)
    df_series = series_table.select(REQUIRED_COLUMNS).to_pandas()
    df_series["dt"] = pd.to_datetime(df_series["dt"])
    df_series = df_series.sort_values("dt").reset_index(drop=True)
    print(f"   ✓ Filtered Store {test_store}, Product {test_product}: {len(df_series)} historical records materialized.")

    # 5. Test Leakage-Safe Feature Engineering
    print("\n5. Testing Feature Engineering...")
    grp = df_series.groupby(["store_id", "product_id"], sort=False)
    df_series["lag_1"] = grp["sale_amount"].shift(1)
    df_series["lag_7"] = grp["sale_amount"].shift(7)
    df_series["rolling_7"] = grp["sale_amount"].transform(lambda x: x.shift(1).rolling(7).mean())
    df_series["stockout_lag_1"] = grp["stock_hour6_22_cnt"].shift(1)
    df_series["stockout_lag_7"] = grp["stock_hour6_22_cnt"].shift(7)
    df_series["stockout_rolling_7"] = grp["stock_hour6_22_cnt"].transform(lambda x: x.shift(1).rolling(7).sum())
    
    valid_rows = df_series.dropna(subset=EXPECTED_FEATURES)
    if valid_rows.empty:
        print("❌ FAIL: No valid feature rows after shift/rolling operations.")
        sys.exit(1)
    print(f"   ✓ Feature engineering succeeded. Valid feature rows: {len(valid_rows)}")

    # 6. Test Model Inference
    print("\n6. Testing Model Inference...")
    latest_row = valid_rows.iloc[-1]
    feature_vec = latest_row[EXPECTED_FEATURES].values.astype(np.float32).reshape(1, -1)
    dmat = xgb.DMatrix(feature_vec, feature_names=EXPECTED_FEATURES)
    pred = float(np.clip(booster.predict(dmat)[0], 0, None))
    print(f"   ✓ Inference prediction for Store {test_store}, Product {test_product}: {pred:.4f} units")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
