"""
Run comprehensive benchmark experiments:
1. Multi-baseline audit including weekday-aligned lag shift(6)
2. Unrounded relative gains and WAPE / Normalized MAE
3. Paired cluster bootstrap CIs by series (1000 resamples)
4. Overall stockout base rate and quantile-based risk evaluation
5. v2 model training and evaluation (saved as xgboost_v2_features.json)
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from datasets import load_dataset

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
PRODUCT_LIMIT = 150
CUTOFF_DATE = pd.Timestamp("2024-06-08")

print("=" * 80)
print("COMPREHENSIVE BENCHMARK, BOOTSTRAP, RISK & V2 EXPERIMENT RUNNER")
print("=" * 80)

# 1. Ingest dataset
print("1. Loading FreshRetailNet-50K...")
t0 = time.time()
ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
table = ds["train"].data.table
sub = table.filter(pc.field("product_id") < PRODUCT_LIMIT)
df = sub.select(["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]).to_pandas()
df["dt"] = pd.to_datetime(df["dt"])
df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
print(f"   Loaded {len(df):,} rows in {time.time() - t0:.2f}s across {df['product_id'].nunique()} products.")

# 2. Extract current day t values
df["sale_at_t"] = df["sale_amount"]
df["stockout_at_t"] = df["stock_hour6_22_cnt"]

# 3. Compute v1 features (frozen contract, history strictly <= t-1)
print("\n2. Engineering v1 and candidate features...")
grp = df.groupby(["store_id", "product_id"], sort=False)

# Frozen v1 features
df["lag_1"] = grp["sale_amount"].shift(1)
df["lag_7"] = grp["sale_amount"].shift(7)
df["rolling_7"] = grp["sale_amount"].transform(lambda x: x.shift(1).rolling(7).mean())
df["stockout_lag_1"] = grp["stock_hour6_22_cnt"].shift(1)
df["stockout_lag_7"] = grp["stock_hour6_22_cnt"].shift(7)
df["stockout_rolling_7"] = grp["stock_hour6_22_cnt"].transform(lambda x: x.shift(1).rolling(7).sum())
df["target_next_day"] = grp["sale_amount"].shift(-1)

# Weekday-aligned lag (shift 6: sale at t-6, which is 7 days before target t+1)
df["lag_weekday_aligned"] = grp["sale_amount"].shift(6)
df["stockout_lag_weekday_aligned"] = grp["stock_hour6_22_cnt"].shift(6)

# v2 Features (incorporating day t information)
# At row t, target is t+1:
# sale_at_t is sale at day t (1 day before target t+1)
# rolling_7_through_t is mean sales of t-6..t (7-day window including day t)
df["rolling_7_through_t"] = grp["sale_amount"].transform(lambda x: x.rolling(7).mean())
df["stockout_rolling_7_through_t"] = grp["stock_hour6_22_cnt"].transform(lambda x: x.rolling(7).sum())

V1_FEATURES = ["lag_1", "lag_7", "rolling_7", "stockout_lag_1", "stockout_lag_7", "stockout_rolling_7"]
V2_FEATURES = [
    "sale_at_t",                    # sale on day t (1 day before target t+1)
    "stockout_at_t",                # stockout hours on day t
    "rolling_7_through_t",          # 7-day rolling mean through day t
    "stockout_rolling_7_through_t", # 7-day rolling stockout sum through day t
    "lag_weekday_aligned",          # sale on day t-6 (7 days before target t+1, exact same weekday!)
    "stockout_lag_weekday_aligned", # stockout on day t-6
    "lag_1",                        # sale on day t-1
    "lag_7"                         # sale on day t-7
]

# Clean valid rows
clean_df = df.dropna(subset=V1_FEATURES + V2_FEATURES + ["target_next_day"]).reset_index(drop=True)
print(f"   Valid rows after warm-up: {len(clean_df):,}")

# Chronological split
train_split = clean_df[clean_df["dt"] <= CUTOFF_DATE].copy()
test_split = clean_df[clean_df["dt"] > CUTOFF_DATE].copy()

print(f"   Train records: {len(train_split):,}, Test records: {len(test_split):,}")

y_train = train_split["target_next_day"].values
y_test = test_split["target_next_day"].values
mean_y_test = float(np.mean(y_test))

# -------------------------------------------------------------
# 3. BASELINES EVALUATION
# -------------------------------------------------------------
print("\n3. Evaluating Baselines on Holdout...")
# Stale lag-1 (t-1 -> t+1)
pred_lag1 = test_split["lag_1"].values
mae_lag1 = mean_absolute_error(y_test, pred_lag1)
rmse_lag1 = root_mean_squared_error(y_test, pred_lag1)
wape_lag1 = (np.sum(np.abs(y_test - pred_lag1)) / np.sum(y_test)) * 100.0

# True previous-day (sale at t -> t+1)
pred_t = test_split["sale_at_t"].values
mae_t = mean_absolute_error(y_test, pred_t)
rmse_t = root_mean_squared_error(y_test, pred_t)
wape_t = (np.sum(np.abs(y_test - pred_t)) / np.sum(y_test)) * 100.0

# 7-Day Rolling Mean (rolling_7: mean t-7..t-1 -> t+1)
pred_roll7 = test_split["rolling_7"].values
mae_roll7 = mean_absolute_error(y_test, pred_roll7)
rmse_roll7 = root_mean_squared_error(y_test, pred_roll7)
wape_roll7 = (np.sum(np.abs(y_test - pred_roll7)) / np.sum(y_test)) * 100.0

# 8-Day Lag Persistence (lag_7: t-7 -> t+1)
pred_lag7 = test_split["lag_7"].values
mae_lag7 = mean_absolute_error(y_test, pred_lag7)
rmse_lag7 = root_mean_squared_error(y_test, pred_lag7)
wape_lag7 = (np.sum(np.abs(y_test - pred_lag7)) / np.sum(y_test)) * 100.0

# Weekday-aligned 7-Day Lag Persistence (shift 6: t-6 -> t+1)
pred_shift6 = test_split["lag_weekday_aligned"].values
mae_shift6 = mean_absolute_error(y_test, pred_shift6)
rmse_shift6 = root_mean_squared_error(y_test, pred_shift6)
wape_shift6 = (np.sum(np.abs(y_test - pred_shift6)) / np.sum(y_test)) * 100.0

# -------------------------------------------------------------
# 4. DEPLOYED XGBOOST (V1 FROZEN) EVALUATION
# -------------------------------------------------------------
print("\n4. Evaluating Deployed Frozen XGBoost...")
v1_booster = xgb.Booster()
v1_booster.load_model(str(BASE_DIR / "xgboost_stock_aware.json"))
dtest_v1 = xgb.DMatrix(test_split[V1_FEATURES], feature_names=V1_FEATURES)
pred_xgb_v1 = np.clip(v1_booster.predict(dtest_v1), 0, None)
mae_xgb_v1 = mean_absolute_error(y_test, pred_xgb_v1)
rmse_xgb_v1 = root_mean_squared_error(y_test, pred_xgb_v1)
wape_xgb_v1 = (np.sum(np.abs(y_test - pred_xgb_v1)) / np.sum(y_test)) * 100.0

# -------------------------------------------------------------
# 5. RANDOM FOREST EVALUATION
# -------------------------------------------------------------
print("\n5. Training Random Forest from scratch...")
t0_rf = time.time()
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rf.fit(train_split[V1_FEATURES], y_train)
pred_rf = np.clip(rf.predict(test_split[V1_FEATURES]), 0, None)
mae_rf = mean_absolute_error(y_test, pred_rf)
rmse_rf = root_mean_squared_error(y_test, pred_rf)
wape_rf = (np.sum(np.abs(y_test - pred_rf)) / np.sum(y_test)) * 100.0
print(f"   RF trained in {time.time() - t0_rf:.2f}s")

# -------------------------------------------------------------
# 6. TRAIN & EVALUATE V2 MODEL EXPERIMENT
# -------------------------------------------------------------
print("\n6. Training v2 XGBoost Model with day-t and weekday-aligned features...")
dtrain_v2 = xgb.DMatrix(train_split[V2_FEATURES], label=y_train, feature_names=V2_FEATURES)
dtest_v2 = xgb.DMatrix(test_split[V2_FEATURES], label=y_test, feature_names=V2_FEATURES)

params_v2 = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 7,
    "learning_rate": 0.05,
    "seed": 42,
    "tree_method": "hist"
}

t0_v2 = time.time()
v2_booster = xgb.train(
    params_v2,
    dtrain_v2,
    num_boost_round=250,
    evals=[(dtrain_v2, "train"), (dtest_v2, "eval")],
    verbose_eval=False
)
v2_model_path = BASE_DIR / "xgboost_v2_features.json"
v2_booster.save_model(str(v2_model_path))
print(f"   v2 model trained in {time.time() - t0_v2:.2f}s and saved to {v2_model_path.name}")

pred_xgb_v2 = np.clip(v2_booster.predict(dtest_v2), 0, None)
mae_xgb_v2 = mean_absolute_error(y_test, pred_xgb_v2)
rmse_xgb_v2 = root_mean_squared_error(y_test, pred_xgb_v2)
wape_xgb_v2 = (np.sum(np.abs(y_test - pred_xgb_v2)) / np.sum(y_test)) * 100.0

# Feature importances for v2
v2_scores = v2_booster.get_score(importance_type="gain")
total_gain = sum(v2_scores.values())
v2_importance = {k: round((v / total_gain) * 100, 2) for k, v in sorted(v2_scores.items(), key=lambda item: item[1], reverse=True)}

# -------------------------------------------------------------
# 7. SUMMARY COMPARISON TABLE WITH UNROUNDED GAINS & RELATIVE ERROR
# -------------------------------------------------------------
print("\n" + "=" * 90)
print(f"MULTI-BASELINE & MODEL COMPARISON TABLE (Mean Holdout Sales: {mean_y_test:.4f})")
print("=" * 90)

models_data = [
    {
        "Name": "1. Stale Lag-1 Baseline (t-1 -> t+1)",
        "MAE": mae_lag1,
        "RMSE": rmse_lag1,
        "WAPE (%)": wape_lag1,
        "NMAE": mae_lag1 / mean_y_test,
        "Gain vs Stale (%)": 0.0,
        "Gain vs XGB v1 (%)": (mae_lag1 - mae_xgb_v1) / mae_lag1 * 100.0
    },
    {
        "Name": "2. True Prev-Day Baseline (sale at t -> t+1)",
        "MAE": mae_t,
        "RMSE": rmse_t,
        "WAPE (%)": wape_t,
        "NMAE": mae_t / mean_y_test,
        "Gain vs Stale (%)": (mae_lag1 - mae_t) / mae_lag1 * 100.0,
        "Gain vs XGB v1 (%)": (mae_t - mae_xgb_v1) / mae_t * 100.0
    },
    {
        "Name": "3. 7-Day Rolling Mean (rolling_7: t-7..t-1)",
        "MAE": mae_roll7,
        "RMSE": rmse_roll7,
        "WAPE (%)": wape_roll7,
        "NMAE": mae_roll7 / mean_y_test,
        "Gain vs Stale (%)": (mae_lag1 - mae_roll7) / mae_lag1 * 100.0,
        "Gain vs XGB v1 (%)": (mae_roll7 - mae_xgb_v1) / mae_roll7 * 100.0
    },
    {
        "Name": "4. 8-Day Lag Persistence (lag_7: t-7 -> t+1)",
        "MAE": mae_lag7,
        "RMSE": rmse_lag7,
        "WAPE (%)": wape_lag7,
        "NMAE": mae_lag7 / mean_y_test,
        "Gain vs Stale (%)": (mae_lag1 - mae_lag7) / mae_lag1 * 100.0,
        "Gain vs XGB v1 (%)": (mae_lag7 - mae_xgb_v1) / mae_lag7 * 100.0
    },
    {
        "Name": "4b. Weekday-Aligned 7-Day Lag (shift(6): t-6 -> t+1)",
        "MAE": mae_shift6,
        "RMSE": rmse_shift6,
        "WAPE (%)": wape_shift6,
        "NMAE": mae_shift6 / mean_y_test,
        "Gain vs Stale (%)": (mae_lag1 - mae_shift6) / mae_lag1 * 100.0,
        "Gain vs XGB v1 (%)": (mae_shift6 - mae_xgb_v1) / mae_shift6 * 100.0
    },
    {
        "Name": "5. Random Forest (Frozen Contract)",
        "MAE": mae_rf,
        "RMSE": rmse_rf,
        "WAPE (%)": wape_rf,
        "NMAE": mae_rf / mean_y_test,
        "Gain vs Stale (%)": (mae_lag1 - mae_rf) / mae_lag1 * 100.0,
        "Gain vs XGB v1 (%)": (mae_rf - mae_xgb_v1) / mae_rf * 100.0
    },
    {
        "Name": "6. Deployed XGBoost Regressor (Frozen v1)",
        "MAE": mae_xgb_v1,
        "RMSE": rmse_xgb_v1,
        "WAPE (%)": wape_xgb_v1,
        "NMAE": mae_xgb_v1 / mean_y_test,
        "Gain vs Stale (%)": (mae_lag1 - mae_xgb_v1) / mae_lag1 * 100.0,
        "Gain vs XGB v1 (%)": 0.0
    },
    {
        "Name": "7. XGBoost v2 Experiment (with day-t & shift(6))",
        "MAE": mae_xgb_v2,
        "RMSE": rmse_xgb_v2,
        "WAPE (%)": wape_xgb_v2,
        "NMAE": mae_xgb_v2 / mean_y_test,
        "Gain vs Stale (%)": (mae_lag1 - mae_xgb_v2) / mae_lag1 * 100.0,
        "Gain vs XGB v1 (%)": (mae_xgb_v1 - mae_xgb_v2) / mae_xgb_v1 * 100.0  # note positive means v2 is better
    }
]

comp_df = pd.DataFrame(models_data)
print(comp_df.to_string(index=False))

# -------------------------------------------------------------
# 8. PAIRED CLUSTER BOOTSTRAP (RESAMPLE BY STORE-PRODUCT SERIES)
# -------------------------------------------------------------
print("\n" + "=" * 90)
print("8. PAIRED CLUSTER BOOTSTRAP (1,000 RESAMPLES BY SERIES)")
print("Resampling unique (store_id, product_id) series clusters to account for autocorrelation...")
print("=" * 90)

# Build a dataframe containing predictions and errors per observation
test_split["series_id"] = test_split["store_id"].astype(str) + "_" + test_split["product_id"].astype(str)
unique_series = test_split["series_id"].unique()
n_series = len(unique_series)
print(f"Total unique series clusters in holdout: {n_series:,}")

# Precompute series-level absolute error sums and counts for fast bootstrapping
series_group = test_split.groupby("series_id")
series_n = series_group.size().values
series_sum_y = series_group["target_next_day"].sum().values

err_xgb_v1 = np.abs(y_test - pred_xgb_v1)
err_rf = np.abs(y_test - pred_rf)
err_roll7 = np.abs(y_test - pred_roll7)
err_t = np.abs(y_test - pred_t)
err_lag1 = np.abs(y_test - pred_lag1)
err_xgb_v2 = np.abs(y_test - pred_xgb_v2)

test_split["err_xgb_v1"] = err_xgb_v1
test_split["err_rf"] = err_rf
test_split["err_roll7"] = err_roll7
test_split["err_t"] = err_t
test_split["err_lag1"] = err_lag1
test_split["err_xgb_v2"] = err_xgb_v2

series_err_xgb_v1 = series_group["err_xgb_v1"].sum().values
series_err_rf = series_group["err_rf"].sum().values
series_err_roll7 = series_group["err_roll7"].sum().values
series_err_t = series_group["err_t"].sum().values
series_err_lag1 = series_group["err_lag1"].sum().values
series_err_xgb_v2 = series_group["err_xgb_v2"].sum().values

N_BOOTSTRAP = 1000
np.random.seed(42)

boot_diff_rf_xgb = []
boot_diff_roll7_xgb = []
boot_diff_t_xgb = []
boot_diff_lag1_xgb = []
boot_diff_v1_v2 = []

t0_boot = time.time()
for b in range(N_BOOTSTRAP):
    sample_idx = np.random.choice(n_series, size=n_series, replace=True)
    total_obs = np.sum(series_n[sample_idx])
    
    mae_b_xgb = np.sum(series_err_xgb_v1[sample_idx]) / total_obs
    mae_b_rf = np.sum(series_err_rf[sample_idx]) / total_obs
    mae_b_roll7 = np.sum(series_err_roll7[sample_idx]) / total_obs
    mae_b_t = np.sum(series_err_t[sample_idx]) / total_obs
    mae_b_lag1 = np.sum(series_err_lag1[sample_idx]) / total_obs
    mae_b_v2 = np.sum(series_err_xgb_v2[sample_idx]) / total_obs
    
    boot_diff_rf_xgb.append(mae_b_rf - mae_b_xgb)
    boot_diff_roll7_xgb.append(mae_b_roll7 - mae_b_xgb)
    boot_diff_t_xgb.append(mae_b_t - mae_b_xgb)
    boot_diff_lag1_xgb.append(mae_b_lag1 - mae_b_xgb)
    boot_diff_v1_v2.append(mae_b_xgb - mae_b_v2)

print(f"Bootstrap finished in {time.time() - t0_boot:.2f}s")

def get_ci(diffs):
    low = np.percentile(diffs, 2.5)
    mean_val = np.mean(diffs)
    high = np.percentile(diffs, 97.5)
    return low, mean_val, high

ci_rf_xgb = get_ci(boot_diff_rf_xgb)
ci_roll7_xgb = get_ci(boot_diff_roll7_xgb)
ci_t_xgb = get_ci(boot_diff_t_xgb)
ci_lag1_xgb = get_ci(boot_diff_lag1_xgb)
ci_v1_v2 = get_ci(boot_diff_v1_v2)

print("\n95% Paired Cluster Bootstrap Confidence Intervals for MAE Difference:")
print(f"1. RF vs XGBoost (MAE_RF - MAE_XGB):")
print(f"   Observed Diff: {mae_rf - mae_xgb_v1:+.4f} | 95% CI: [{ci_rf_xgb[0]:+.4f}, {ci_rf_xgb[2]:+.4f}]")
if ci_rf_xgb[0] <= 0 <= ci_rf_xgb[2]:
    print("   -> 0 IS in the 95% CI: The 0.0013 gap between RF and XGBoost is NOT statistically distinguishable from noise!")
else:
    print("   -> 0 is NOT in the 95% CI: The gap is statistically distinguishable from zero.")

print(f"2. Rolling_7 vs XGBoost (MAE_roll7 - MAE_XGB):")
print(f"   Observed Diff: {mae_roll7 - mae_xgb_v1:+.4f} | 95% CI: [{ci_roll7_xgb[0]:+.4f}, {ci_roll7_xgb[2]:+.4f}]")

print(f"3. True Prev-Day (t) vs XGBoost (MAE_t - MAE_XGB):")
print(f"   Observed Diff: {mae_t - mae_xgb_v1:+.4f} | 95% CI: [{ci_t_xgb[0]:+.4f}, {ci_t_xgb[2]:+.4f}]")

print(f"4. Stale Lag-1 vs XGBoost (MAE_lag1 - MAE_XGB):")
print(f"   Observed Diff: {mae_lag1 - mae_xgb_v1:+.4f} | 95% CI: [{ci_lag1_xgb[0]:+.4f}, {ci_lag1_xgb[2]:+.4f}]")

print(f"5. XGBoost v1 vs v2 (MAE_v1 - MAE_v2):")
print(f"   Observed Diff: {mae_xgb_v1 - mae_xgb_v2:+.4f} | 95% CI: [{ci_v1_v2[0]:+.4f}, {ci_v1_v2[2]:+.4f}]")

# -------------------------------------------------------------
# 9. RISK LAYER: BASE RATE & QUANTILE THRESHOLDS
# -------------------------------------------------------------
print("\n" + "=" * 90)
print("9. STOCKOUT RISK LAYER: BASE RATE & QUANTILE ALTERNATIVE")
print("=" * 90)

overall_stockout_rate = (test_split["stockout_at_t"] > 0).mean() * 100.0
overall_mean_stockout_hrs = test_split["stockout_at_t"].mean()
print(f"Holdout Overall Stockout Base Rate: {overall_stockout_rate:.2f}% (Mean stockout hours: {overall_mean_stockout_hrs:.2f}h)")

# Current fixed thresholds (<8, 8-23, >=24)
def assess_fixed(h):
    if h >= 24.0: return "HIGH"
    elif h >= 8.0: return "MEDIUM"
    else: return "LOW"

test_split["risk_fixed"] = test_split["stockout_rolling_7"].apply(assess_fixed)
summary_fixed = test_split.groupby("risk_fixed")["stockout_at_t"].agg(
    observations="count",
    mean_stockout_hrs="mean",
    median_stockout_hrs="median"
).round(2)
summary_fixed["actual_stockout_rate_%"] = test_split.groupby("risk_fixed")["stockout_at_t"].apply(
    lambda s: (s > 0).mean() * 100.0
).round(2)
summary_fixed["share_of_test_rows_%"] = (summary_fixed["observations"] / len(test_split) * 100.0).round(2)
summary_fixed["lift_over_base"] = (summary_fixed["actual_stockout_rate_%"] / overall_stockout_rate).round(2)
summary_fixed["overall_base_rate_%"] = round(overall_stockout_rate, 2)

print("\nFixed Threshold Summary (<8h, 8-23h, >=24h):")
print(summary_fixed.loc[["LOW", "MEDIUM", "HIGH"]])

# Quantile-based thresholds:
# Top 10% HIGH (>= 90th percentile)
# Next 20% MEDIUM (70th to 90th percentile)
# Bottom 70% LOW (< 70th percentile)
q70 = train_split["stockout_rolling_7"].quantile(0.70)
q90 = train_split["stockout_rolling_7"].quantile(0.90)
print(f"\nEmpirical Training Quantiles for stockout_rolling_7: 70th={q70:.1f}h, 90th={q90:.1f}h")

def assess_quantile(h):
    if h >= q90: return "HIGH"
    elif h >= q70: return "MEDIUM"
    else: return "LOW"

test_split["risk_quantile"] = test_split["stockout_rolling_7"].apply(assess_quantile)
summary_quantile = test_split.groupby("risk_quantile")["stockout_at_t"].agg(
    observations="count",
    mean_stockout_hrs="mean",
    median_stockout_hrs="median"
).round(2)
summary_quantile["actual_stockout_rate_%"] = test_split.groupby("risk_quantile")["stockout_at_t"].apply(
    lambda s: (s > 0).mean() * 100.0
).round(2)
summary_quantile["share_of_test_rows_%"] = (summary_quantile["observations"] / len(test_split) * 100.0).round(2)
summary_quantile["lift_over_base"] = (summary_quantile["actual_stockout_rate_%"] / overall_stockout_rate).round(2)
summary_quantile["overall_base_rate_%"] = round(overall_stockout_rate, 2)

print("\nQuantile-Based Configurable Alternative (LOW: <70th, MEDIUM: 70th-90th, HIGH: >=90th):")
print(summary_quantile.loc[["LOW", "MEDIUM", "HIGH"]])

# Export results dictionary for easy integration
results_payload = {
    "mean_y_test": mean_y_test,
    "metrics": comp_df.to_dict(orient="records"),
    "ci_rf_xgb": {"low": ci_rf_xgb[0], "mean": ci_rf_xgb[1], "high": ci_rf_xgb[2]},
    "ci_roll7_xgb": {"low": ci_roll7_xgb[0], "mean": ci_roll7_xgb[1], "high": ci_roll7_xgb[2]},
    "ci_t_xgb": {"low": ci_t_xgb[0], "mean": ci_t_xgb[1], "high": ci_t_xgb[2]},
    "ci_lag1_xgb": {"low": ci_lag1_xgb[0], "mean": ci_lag1_xgb[1], "high": ci_lag1_xgb[2]},
    "ci_v1_v2": {"low": ci_v1_v2[0], "mean": ci_v1_v2[1], "high": ci_v1_v2[2]},
    "overall_stockout_rate": overall_stockout_rate,
    "overall_mean_stockout_hrs": overall_mean_stockout_hrs,
    "q70": q70,
    "q90": q90,
    "v2_importance": v2_importance
}

with open(BASE_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=2)

print("\nAll experiment results written to experiment_results.json successfully!")
