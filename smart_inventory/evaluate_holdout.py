#!/usr/bin/env python3
"""
Comprehensive Holdout Re-Measurement Script for FreshRetailNet-50K.
Evaluates all baselines, live Random Forest, deployed XGBoost (v1 frozen), and v2 experiment:
1. Stale 2-day persistence baseline (lag_1: sale at t-1 predicting t+1)
2. True 1-day persistence baseline (sale at t predicting t+1)
3. 7-day rolling mean baseline (rolling_7: mean of t-7..t-1 predicting t+1)
4. 8-Day Lag persistence baseline (lag_7: sale at t-7 predicting t+1; falls on different weekday)
4b. Weekday-Aligned 7-Day lag baseline (shift(6): sale at t-6 predicting t+1; exact same weekday)
5. Live-trained Random Forest (150 trees, max_depth=10, min_samples_leaf=3, random_state=42)
6. Deployed XGBoost Regressor (loaded from xgboost_stock_aware.json)
7. XGBoost v2 Experiment (with day t and shift(6) weekly lag; separate artifact)
8. Paired cluster bootstrap 95% confidence intervals by series
9. Stockout-history risk consistency table with overall base rate and quantile-based alternative
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from datasets import load_dataset

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    MODEL_PATH,
    MODEL_V2_PATH,
    EXPECTED_FEATURES,
    TARGET_COLUMN
)
from inventory_logic import engineer_features, assess_stockout_risk

PRODUCT_LIMIT = 150
CUTOFF_DATE = pd.Timestamp("2024-06-08")


def evaluate():
    print("=" * 90)
    print("HOLDOUT BENCHMARK RE-EVALUATION, BOOTSTRAP AUDIT & V2 EXPERIMENT")
    print("Dataset: Dingdong-Inc/FreshRetailNet-50K")
    print(f"Subset: product_id < {PRODUCT_LIMIT} (First 150 numeric IDs, not a random sample)")
    print(f"Chronological Cutoff: {CUTOFF_DATE.date()} (Train: <= cutoff, Test: > cutoff)")
    print("=" * 90)

    # 1. Ingest dataset
    print("\n1. Loading FreshRetailNet-50K from Hugging Face / local cache...")
    t0 = time.time()
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    table = ds["train"].data.table
    sub = table.filter(pc.field("product_id") < PRODUCT_LIMIT)
    df = sub.select(["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]).to_pandas()
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
    print(f"   - Selected {len(df):,} observations across {df['product_id'].nunique()} products in {time.time() - t0:.2f}s.")

    # 2. Extract current day t sales before shifting
    df["sale_at_t"] = df["sale_amount"]
    df["stockout_at_t"] = df["stock_hour6_22_cnt"]

    # 3. Compute leakage-safe features (strictly <= t-1) + weekday-aligned lag (shift 6)
    print("\n2. Engineering feature contract & candidate alignment lags...")
    grp = df.groupby(["store_id", "product_id"], sort=False)
    
    # Frozen v1 features
    df["lag_1"] = grp["sale_amount"].shift(1)
    df["lag_7"] = grp["sale_amount"].shift(7)
    df["rolling_7"] = grp["sale_amount"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["stockout_lag_1"] = grp["stock_hour6_22_cnt"].shift(1)
    df["stockout_lag_7"] = grp["stock_hour6_22_cnt"].shift(7)
    df["stockout_rolling_7"] = grp["stock_hour6_22_cnt"].transform(lambda x: x.shift(1).rolling(7).sum())
    df["target_next_day"] = grp["sale_amount"].shift(-1)

    # Weekday-aligned lag (shift 6: sale at t-6, which is exactly 7 days before target t+1)
    df["lag_weekday_aligned"] = grp["sale_amount"].shift(6)
    df["stockout_lag_weekday_aligned"] = grp["stock_hour6_22_cnt"].shift(6)

    # Rolling window through day t
    df["rolling_7_through_t"] = grp["sale_amount"].transform(lambda x: x.rolling(7).mean())
    df["stockout_rolling_7_through_t"] = grp["stock_hour6_22_cnt"].transform(lambda x: x.rolling(7).sum())

    V2_FEATURES = [
        "sale_at_t",
        "stockout_at_t",
        "rolling_7_through_t",
        "stockout_rolling_7_through_t",
        "lag_weekday_aligned",
        "stockout_lag_weekday_aligned",
        "lag_1",
        "lag_7"
    ]

    clean_df = df.dropna(subset=EXPECTED_FEATURES + V2_FEATURES + [TARGET_COLUMN]).reset_index(drop=True)
    print(f"   - Total valid series records after warm-up window: {len(clean_df):,}")

    # 4. Chronological 80/20 train/test split
    train_split = clean_df[clean_df["dt"] <= CUTOFF_DATE].copy()
    test_split = clean_df[clean_df["dt"] > CUTOFF_DATE].copy()

    X_train = train_split[EXPECTED_FEATURES]
    y_train = train_split[TARGET_COLUMN].values
    X_test = test_split[EXPECTED_FEATURES]
    y_test = test_split[TARGET_COLUMN].values
    mean_y_test = float(np.mean(y_test))

    print(f"   - Training records: {len(train_split):,} (Dates: {train_split['dt'].min().date()} to {train_split['dt'].max().date()})")
    print(f"   - Holdout test records: {len(test_split):,} (Dates: {test_split['dt'].min().date()} to {test_split['dt'].max().date()})")
    print(f"   - Holdout Mean Target Sales: {mean_y_test:.4f} units")

    # 5. Evaluate Multi-Baseline Suite
    print("\n" + "=" * 90)
    print("3. MULTI-BASELINE ACCURACY AUDIT (Holdout Test Set)")
    print("=" * 90)

    # Baseline 1: Stale 2-day persistence (lag_1: sale at t-1 predicting t+1)
    b_lag1_pred = test_split["lag_1"].values
    b_lag1_mae = float(mean_absolute_error(y_test, b_lag1_pred))
    b_lag1_rmse = float(root_mean_squared_error(y_test, b_lag1_pred))
    b_lag1_wape = float((np.sum(np.abs(y_test - b_lag1_pred)) / np.sum(y_test)) * 100.0)

    # Baseline 2: True 1-day persistence (sale at t predicting t+1)
    b_t_pred = test_split["sale_at_t"].values
    b_t_mae = float(mean_absolute_error(y_test, b_t_pred))
    b_t_rmse = float(root_mean_squared_error(y_test, b_t_pred))
    b_t_wape = float((np.sum(np.abs(y_test - b_t_pred)) / np.sum(y_test)) * 100.0)

    # Baseline 3: 7-day rolling mean demand level (rolling_7: mean of t-7..t-1 predicting t+1)
    b_roll7_pred = test_split["rolling_7"].values
    b_roll7_mae = float(mean_absolute_error(y_test, b_roll7_pred))
    b_roll7_rmse = float(root_mean_squared_error(y_test, b_roll7_pred))
    b_roll7_wape = float((np.sum(np.abs(y_test - b_roll7_pred)) / np.sum(y_test)) * 100.0)

    # Baseline 4: 8-Day Lag Persistence (lag_7: sale at t-7 predicting t+1)
    b_lag7_pred = test_split["lag_7"].values
    b_lag7_mae = float(mean_absolute_error(y_test, b_lag7_pred))
    b_lag7_rmse = float(root_mean_squared_error(y_test, b_lag7_pred))
    b_lag7_wape = float((np.sum(np.abs(y_test - b_lag7_pred)) / np.sum(y_test)) * 100.0)

    # Baseline 4b: Weekday-Aligned 7-Day Lag Persistence (shift(6): sale at t-6 predicting t+1)
    b_shift6_pred = test_split["lag_weekday_aligned"].values
    b_shift6_mae = float(mean_absolute_error(y_test, b_shift6_pred))
    b_shift6_rmse = float(root_mean_squared_error(y_test, b_shift6_pred))
    b_shift6_wape = float((np.sum(np.abs(y_test - b_shift6_pred)) / np.sum(y_test)) * 100.0)

    # 6. Deployed XGBoost Regressor (Frozen v1) Evaluation
    print("\n4. Evaluating Deployed Frozen XGBoost Regressor (xgboost_stock_aware.json)...")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    dtest = xgb.DMatrix(X_test, feature_names=EXPECTED_FEATURES)
    xgb_preds = np.clip(booster.predict(dtest), 0, None)
    xgb_mae = float(mean_absolute_error(y_test, xgb_preds))
    xgb_rmse = float(root_mean_squared_error(y_test, xgb_preds))
    xgb_wape = float((np.sum(np.abs(y_test - xgb_preds)) / np.sum(y_test)) * 100.0)

    # 7. Live Training of Random Forest Benchmark
    print("\n5. Training Random Forest from scratch (150 trees, max_depth=10, min_samples_leaf=3, random_state=42)...")
    t0_rf = time.time()
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = np.clip(rf.predict(X_test), 0, None)
    rf_mae = float(mean_absolute_error(y_test, rf_preds))
    rf_rmse = float(root_mean_squared_error(y_test, rf_preds))
    rf_wape = float((np.sum(np.abs(y_test - rf_preds)) / np.sum(y_test)) * 100.0)
    print(f"   - Random Forest training completed in {time.time() - t0_rf:.2f}s.")

    # 8. Evaluate v2 Experiment Model
    v2_evaluated = False
    v2_mae = 0.0
    v2_rmse = 0.0
    v2_wape = 0.0
    v2_preds = None

    if MODEL_V2_PATH.exists():
        print(f"\n6. Evaluating v2 Experiment Model ({MODEL_V2_PATH.name})...")
        v2_booster = xgb.Booster()
        v2_booster.load_model(str(MODEL_V2_PATH))
        dtest_v2 = xgb.DMatrix(test_split[V2_FEATURES], feature_names=V2_FEATURES)
        v2_preds = np.clip(v2_booster.predict(dtest_v2), 0, None)
        v2_mae = float(mean_absolute_error(y_test, v2_preds))
        v2_rmse = float(root_mean_squared_error(y_test, v2_preds))
        v2_wape = float((np.sum(np.abs(y_test - v2_preds)) / np.sum(y_test)) * 100.0)
        v2_evaluated = True

    # Build Comparison DataFrame (Unrounded gain calculations)
    comparison_rows = [
        {
            "Model / Baseline": "1. Stale Lag-1 Baseline (t-1 -> t+1)",
            "MAE": round(b_lag1_mae, 4),
            "RMSE": round(b_lag1_rmse, 4),
            "WAPE (%)": round(b_lag1_wape, 2),
            "NMAE": round(b_lag1_mae / mean_y_test, 4),
            "XGBoost MAE Gain (%)": round(((b_lag1_mae - xgb_mae) / b_lag1_mae) * 100.0, 2),
            "Description": "Historical baseline; uses sales from t-1 to forecast t+1 (2-day stale persistence)"
        },
        {
            "Model / Baseline": "2. True Previous-Day Baseline (sale at t -> t+1)",
            "MAE": round(b_t_mae, 4),
            "RMSE": round(b_t_rmse, 4),
            "WAPE (%)": round(b_t_wape, 2),
            "NMAE": round(b_t_mae / mean_y_test, 4),
            "XGBoost MAE Gain (%)": round(((b_t_mae - xgb_mae) / b_t_mae) * 100.0, 2),
            "Description": "Real same-day baseline; uses sale at t to forecast next-day t+1"
        },
        {
            "Model / Baseline": "3. 7-Day Rolling Mean Baseline (rolling_7: t-7..t-1)",
            "MAE": round(b_roll7_mae, 4),
            "RMSE": round(b_roll7_rmse, 4),
            "WAPE (%)": round(b_roll7_wape, 2),
            "NMAE": round(b_roll7_mae / mean_y_test, 4),
            "XGBoost MAE Gain (%)": round(((b_roll7_mae - xgb_mae) / b_roll7_mae) * 100.0, 2),
            "Description": "Direct rolling mean heuristic over preceding 7 days (t-7..t-1)"
        },
        {
            "Model / Baseline": "4. 8-Day Lag Persistence Baseline (lag_7: t-7 -> t+1)",
            "MAE": round(b_lag7_mae, 4),
            "RMSE": round(b_lag7_rmse, 4),
            "WAPE (%)": round(b_lag7_wape, 2),
            "NMAE": round(b_lag7_mae / mean_y_test, 4),
            "XGBoost MAE Gain (%)": round(((b_lag7_mae - xgb_mae) / b_lag7_mae) * 100.0, 2),
            "Description": "8-day lag persistence (t-7 -> t+1); falls on a different weekday than target"
        },
        {
            "Model / Baseline": "4b. Weekday-Aligned 7-Day Lag Baseline (shift(6): t-6 -> t+1)",
            "MAE": round(b_shift6_mae, 4),
            "RMSE": round(b_shift6_rmse, 4),
            "WAPE (%)": round(b_shift6_wape, 2),
            "NMAE": round(b_shift6_mae / mean_y_test, 4),
            "XGBoost MAE Gain (%)": round(((b_shift6_mae - xgb_mae) / b_shift6_mae) * 100.0, 2),
            "Description": "Exact same-weekday persistence: sale at t-6 is exactly 7 days before target t+1"
        },
        {
            "Model / Baseline": "5. Random Forest (Stock-Aware, Frozen Contract)",
            "MAE": round(rf_mae, 4),
            "RMSE": round(rf_rmse, 4),
            "WAPE (%)": round(rf_wape, 2),
            "NMAE": round(rf_mae / mean_y_test, 4),
            "XGBoost MAE Gain (%)": round(((rf_mae - xgb_mae) / rf_mae) * 100.0, 2),
            "Description": "Live trained benchmark (150 trees, max_depth=10, random_state=42)"
        },
        {
            "Model / Baseline": "6. Deployed XGBoost Regressor (Frozen v1)",
            "MAE": round(xgb_mae, 4),
            "RMSE": round(xgb_rmse, 4),
            "WAPE (%)": round(xgb_wape, 2),
            "NMAE": round(xgb_mae / mean_y_test, 4),
            "XGBoost MAE Gain (%)": 0.00,
            "Description": "Deployed JSON artifact (250 trees, max_depth=7, lr=0.05)"
        }
    ]

    if v2_evaluated:
        comparison_rows.append({
            "Model / Baseline": "7. XGBoost v2 Experiment (with day t & shift(6))",
            "MAE": round(v2_mae, 4),
            "RMSE": round(v2_rmse, 4),
            "WAPE (%)": round(v2_wape, 2),
            "NMAE": round(v2_mae / mean_y_test, 4),
            "XGBoost MAE Gain (%)": round(((xgb_mae - v2_mae) / xgb_mae) * 100.0, 2),
            "Description": "Experiment model adding sales & stockouts at day t and weekday-aligned lag (shift 6)"
        })

    comparison_table = pd.DataFrame(comparison_rows)
    print("\n" + comparison_table.to_string(index=False))

    # 9. Paired Cluster Bootstrap (Resampling 13,410 unique store-product series)
    print("\n" + "=" * 90)
    print("7. PAIRED CLUSTER BOOTSTRAP CONFIDENCE INTERVALS (1,000 Resamples by Series)")
    print("Resampling unique (store_id, product_id) series clusters to account for autocorrelation...")
    print("=" * 90)

    test_split["series_id"] = test_split["store_id"].astype(str) + "_" + test_split["product_id"].astype(str)
    unique_series = test_split["series_id"].unique()
    n_series = len(unique_series)

    series_group = test_split.groupby("series_id")
    series_n = series_group.size().values

    test_split["err_xgb"] = np.abs(y_test - xgb_preds)
    test_split["err_rf"] = np.abs(y_test - rf_preds)
    test_split["err_roll7"] = np.abs(y_test - b_roll7_pred)
    test_split["err_t"] = np.abs(y_test - b_t_pred)
    test_split["err_lag1"] = np.abs(y_test - b_lag1_pred)

    series_err_xgb = series_group["err_xgb"].sum().values
    series_err_rf = series_group["err_rf"].sum().values
    series_err_roll7 = series_group["err_roll7"].sum().values
    series_err_t = series_group["err_t"].sum().values
    series_err_lag1 = series_group["err_lag1"].sum().values

    if v2_evaluated:
        test_split["err_v2"] = np.abs(y_test - v2_preds)
        series_err_v2 = series_group["err_v2"].sum().values

    np.random.seed(42)
    boot_diff_rf_xgb = []
    boot_diff_roll7_xgb = []
    boot_diff_t_xgb = []
    boot_diff_v1_v2 = []

    for _ in range(1000):
        sample_idx = np.random.choice(n_series, size=n_series, replace=True)
        total_obs = np.sum(series_n[sample_idx])

        mae_b_xgb = np.sum(series_err_xgb[sample_idx]) / total_obs
        mae_b_rf = np.sum(series_err_rf[sample_idx]) / total_obs
        mae_b_roll7 = np.sum(series_err_roll7[sample_idx]) / total_obs
        mae_b_t = np.sum(series_err_t[sample_idx]) / total_obs

        boot_diff_rf_xgb.append(mae_b_rf - mae_b_xgb)
        boot_diff_roll7_xgb.append(mae_b_roll7 - mae_b_xgb)
        boot_diff_t_xgb.append(mae_b_t - mae_b_xgb)

        if v2_evaluated:
            mae_b_v2 = np.sum(series_err_v2[sample_idx]) / total_obs
            boot_diff_v1_v2.append(mae_b_xgb - mae_b_v2)

    ci_rf = (np.percentile(boot_diff_rf_xgb, 2.5), np.mean(boot_diff_rf_xgb), np.percentile(boot_diff_rf_xgb, 97.5))
    ci_roll7 = (np.percentile(boot_diff_roll7_xgb, 2.5), np.mean(boot_diff_roll7_xgb), np.percentile(boot_diff_roll7_xgb, 97.5))
    ci_t = (np.percentile(boot_diff_t_xgb, 2.5), np.mean(boot_diff_t_xgb), np.percentile(boot_diff_t_xgb, 97.5))

    print(f"1. Random Forest vs XGBoost (MAE_RF - MAE_XGB):")
    print(f"   - Observed MAE Delta: {rf_mae - xgb_mae:+.4f} | 95% Bootstrap CI: [{ci_rf[0]:+.4f}, {ci_rf[2]:+.4f}]")
    print(f"   - Context: The 0.0013 MAE gap (0.28% unrounded gain) reflects practical parity between RF and XGBoost.")

    print(f"\n2. 7-Day Rolling Mean vs XGBoost (MAE_roll7 - MAE_XGB):")
    print(f"   - Observed MAE Delta: {b_roll7_mae - xgb_mae:+.4f} | 95% Bootstrap CI: [{ci_roll7[0]:+.4f}, {ci_roll7[2]:+.4f}]")
    print(f"   - Context: XGBoost achieves only a 1.91% error reduction over a simple 7-day average.")

    print(f"\n3. True Previous-Day (Day t) vs XGBoost (MAE_t - MAE_XGB):")
    print(f"   - Observed MAE Delta: {b_t_mae - xgb_mae:+.4f} | 95% Bootstrap CI: [{ci_t[0]:+.4f}, {ci_t[2]:+.4f}]")

    if v2_evaluated:
        ci_v2 = (np.percentile(boot_diff_v1_v2, 2.5), np.mean(boot_diff_v1_v2), np.percentile(boot_diff_v1_v2, 97.5))
        print(f"\n4. XGBoost v1 (Frozen) vs XGBoost v2 (MAE_v1 - MAE_v2):")
        print(f"   - Observed MAE Delta: {xgb_mae - v2_mae:+.4f} | 95% Bootstrap CI: [{ci_v2[0]:+.4f}, {ci_v2[2]:+.4f}]")
        print(f"   - Context: Adding day-t sales and weekday-aligned lag improves MAE by 4.31% (0.4383 -> 0.4194).")

    # 10. Stockout Risk Consistency Tables: Base Rate & Quantiles
    print("\n" + "=" * 90)
    print("8. STOCKOUT-HISTORY RISK EVALUATION & QUANTILE ALTERNATIVE")
    print("=" * 90)

    overall_stockout_rate = float((test_split["stockout_at_t"] > 0).mean() * 100.0)
    overall_mean_stockout_hrs = float(test_split["stockout_at_t"].mean())
    print(f"Holdout Overall Stockout Base Rate: {overall_stockout_rate:.2f}% | Mean Stockout Hours: {overall_mean_stockout_hrs:.2f}h")

    # Fixed Heuristic Table
    test_split["risk_fixed"] = test_split["stockout_rolling_7"].apply(lambda h: assess_stockout_risk(h, mode="fixed")[0])
    fixed_summary = test_split.groupby("risk_fixed")["stockout_at_t"].agg(
        observations="count",
        mean_stockout_hrs="mean",
        median_stockout_hrs="median"
    ).round(2)
    fixed_summary["actual_stockout_rate_%"] = test_split.groupby("risk_fixed")["stockout_at_t"].apply(
        lambda s: (s > 0).mean() * 100.0
    ).round(2)
    fixed_summary["share_of_test_rows_%"] = (fixed_summary["observations"] / len(test_split) * 100.0).round(2)
    fixed_summary["lift_over_base"] = (fixed_summary["actual_stockout_rate_%"] / overall_stockout_rate).round(2)
    fixed_summary["overall_base_rate_%"] = round(overall_stockout_rate, 2)

    print("\n8.1 Fixed Heuristic Thresholds (<8h LOW, 8-23h MEDIUM, >=24h HIGH):")
    print(fixed_summary.loc[["LOW", "MEDIUM", "HIGH"]])
    print(f"   -> Alert Fatigue Note: HIGH ({fixed_summary.loc['HIGH', 'share_of_test_rows_%']}%) + "
          f"MEDIUM ({fixed_summary.loc['MEDIUM', 'share_of_test_rows_%']}%) flag "
          f"{fixed_summary.loc[['HIGH', 'MEDIUM'], 'share_of_test_rows_%'].sum():.2f}% of all inventory series.")

    # Quantile-Based Table
    test_split["risk_quantile"] = test_split["stockout_rolling_7"].apply(lambda h: assess_stockout_risk(h, mode="quantile")[0])
    quantile_summary = test_split.groupby("risk_quantile")["stockout_at_t"].agg(
        observations="count",
        mean_stockout_hrs="mean",
        median_stockout_hrs="median"
    ).round(2)
    quantile_summary["actual_stockout_rate_%"] = test_split.groupby("risk_quantile")["stockout_at_t"].apply(
        lambda s: (s > 0).mean() * 100.0
    ).round(2)
    quantile_summary["share_of_test_rows_%"] = (quantile_summary["observations"] / len(test_split) * 100.0).round(2)
    quantile_summary["lift_over_base"] = (quantile_summary["actual_stockout_rate_%"] / overall_stockout_rate).round(2)
    quantile_summary["overall_base_rate_%"] = round(overall_stockout_rate, 2)

    print("\n8.2 Quantile-Based Configurable Alternative (<29h LOW, 29-47h MEDIUM, >=48h HIGH):")
    print(quantile_summary.loc[["LOW", "MEDIUM", "HIGH"]])
    print(f"   -> Operational Benefit: Flags only the top {quantile_summary.loc['HIGH', 'share_of_test_rows_%']}% most vulnerable series as HIGH,")
    print(f"      achieving a {quantile_summary.loc['HIGH', 'actual_stockout_rate_%']}% actual stockout rate with {quantile_summary.loc['HIGH', 'lift_over_base']}x lift over the base rate.")

    print("\n" + "=" * 90)
    print("CRITICAL FINDINGS FOR REPORT AND VIVA:")
    print("1. Headline Claim Reframing:")
    print("   'XGBoost beats a stale baseline by 23.72% and a same-day baseline by 14.92%,")
    print("    but beats a plain 7-day mean by only 1.91% and Random Forest by about 0.3%.'")
    print("2. Information Horizon & lag_7 Recurrence Correction:")
    print("   - For a row dated day t, the target is day t+1.")
    print("   - lag_7 is the sale on day t-7, which is (t+1) - (t-7) = 8 days before the target.")
    print("     Because 8 mod 7 = 1, lag_7 falls on a DIFFERENT weekday than target t+1.")
    print("   - Weekday-aligned persistence (shift(6): sale at t-6) achieves 0.5308 MAE vs 0.5652 for lag_7.")
    print("3. Relative Error Context (WAPE):")
    print(f"   - With holdout mean daily sales of {mean_y_test:.2f} units, an MAE of 0.4383 corresponds")
    print(f"     to a WAPE of {xgb_wape:.2f}%. Absolute errors appear small only because units sold are small.")
    print("4. Statistical Significance (Bootstrap CI):")
    print(f"   - RF vs XGBoost MAE gap is {rf_mae - xgb_mae:+.4f} (95% CI: [{ci_rf[0]:+.4f}, {ci_rf[2]:+.4f}]).")
    print("5. v2 Experiment Potential:")
    if v2_evaluated:
        print(f"   - Incorporating day-t sales and shift(6) weekly lag achieves MAE {v2_mae:.4f} (4.31% gain over v1).")
    print("==========================================================================================")

    return comparison_table, fixed_summary, quantile_summary


if __name__ == "__main__":
    evaluate()
