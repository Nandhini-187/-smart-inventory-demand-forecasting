"""
Phase 4: Modelling & Training Pipeline
FreshRetailNet-50K Demand Forecasting & Inventory Management
"""

import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def compute_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, WAPE, Bias, R2, and conditional MAPE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sum_true = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / sum_true) if sum_true > 0 else 0.0
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))
    
    # Conditional MAPE on non-zero actuals
    nz_mask = y_true > 1e-4
    if np.sum(nz_mask) > 0:
        mape_nz = float(np.mean(np.abs(y_true[nz_mask] - y_pred[nz_mask]) / y_true[nz_mask])) * 100.0
    else:
        mape_nz = 0.0
        
    return {
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "bias": bias,
        "r2": r2,
        "mape_nonzero": mape_nz
    }

def classify_syntetos_boylan(df_train):
    """
    Computes ADI (Average Demand Interval) and CV^2 on training data per series.
    Quadrant categorization:
    - Smooth: ADI < 1.32 and CV^2 < 0.49
    - Erratic: ADI < 1.32 and CV^2 >= 0.49
    - Intermittent: ADI >= 1.32 and CV^2 < 0.49
    - Lumpy: ADI >= 1.32 and CV^2 >= 0.49
    """
    class_map = {}
    for (s_id, p_id), grp in df_train.groupby(['store_id', 'product_id'], observed=True):
        sales = grp['sale_amount'].to_numpy()
        n_periods = len(sales)
        nz_sales = sales[sales > 1e-4]
        n_nz = len(nz_sales)
        
        if n_nz == 0:
            adi = 999.0
            cv2 = 0.0
        else:
            adi = n_periods / n_nz
            mean_nz = np.mean(nz_sales)
            std_nz = np.std(nz_sales)
            cv2 = (std_nz / (mean_nz + 1e-6)) ** 2
            
        if adi < 1.32:
            vol_class = "smooth" if cv2 < 0.49 else "erratic"
        else:
            vol_class = "intermittent" if cv2 < 0.49 else "lumpy"
            
        class_map[(str(s_id), str(p_id))] = vol_class
    return class_map

def croston_sba(sales, alpha=0.1):
    """Syntetos-Boylan Approximation (SBA) of Croston's method for intermittent demand."""
    sales = np.asarray(sales)
    n = len(sales)
    if n == 0 or np.all(sales <= 1e-4):
        return 0.0
    
    # Initialize
    first_nz_idx = np.where(sales > 1e-4)[0]
    if len(first_nz_idx) == 0:
        return 0.0
        
    z = sales[first_nz_idx[0]] # non-zero demand level
    p = 1.0 # inter-arrival interval
    q = 1 # counter
    
    for t in range(first_nz_idx[0] + 1, n):
        if sales[t] > 1e-4:
            z = alpha * sales[t] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
            
    # SBA modification factor (1 - alpha / 2)
    sba_forecast = (1 - alpha / 2.0) * (z / max(p, 1e-4))
    return float(max(0.0, sba_forecast))

def train_and_evaluate_all():
    print(f"[{datetime.now()}] Loading processed feature datasets...")
    train_df = pd.read_parquet("data/processed/train_features.parquet")
    eval_df = pd.read_parquet("data/processed/eval_features.parquet")
    
    train_df['dt'] = pd.to_datetime(train_df['dt'])
    eval_df['dt'] = pd.to_datetime(eval_df['dt'])
    
    # Feature columns
    drop_cols = [
        'dt', 'sale_amount', 'latent_demand', 'target_sales', 'target_latent',
        'hours_sale', 'hours_stock_status', 'is_outlier',
        'same_day_stockout_hrs', 'same_day_stockout_ratio', 'same_day_in_stock_rate',
        'same_day_peak_hour', 'same_day_morning_share', 'same_day_evening_share',
        'first_stockout_hour'
    ]
    features = [c for c in train_df.columns if c not in drop_cols]
    
    # Ensure categorical dtypes for XGBoost enable_categorical=True
    cat_cols = ['city_id', 'store_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id', 'product_id', 'lifecycle_stage']
    for c in cat_cols:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype('category')
            eval_df[c] = eval_df[c].astype('category')
            
    print(f"Features count: {len(features)}")
    
    # Time-based validation split: last 14 days of train for validation holdout
    max_train_date = train_df['dt'].max()
    val_start_date = max_train_date - pd.Timedelta(days=13)
    
    train_fold = train_df[train_df['dt'] < val_start_date].copy()
    val_fold = train_df[train_df['dt'] >= val_start_date].copy()
    
    print(f"Train fold dates: {train_fold['dt'].min()} to {train_fold['dt'].max()} ({len(train_fold)} rows)")
    print(f"Val fold dates: {val_fold['dt'].min()} to {val_fold['dt'].max()} ({len(val_fold)} rows)")
    print(f"Eval split dates: {eval_df['dt'].min()} to {eval_df['dt'].max()} ({len(eval_df)} rows)")

    X_train, y_train_obs, y_train_latent = train_fold[features], train_fold['target_sales'].values, train_fold['target_latent'].values
    X_val, y_val_obs, y_val_latent = val_fold[features], val_fold['target_sales'].values, val_fold['target_latent'].values
    X_eval, y_eval_obs = eval_df[features], eval_df['target_sales'].values

    # Determine device (CUDA if available)
    try:
        test_model = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=2)
        test_model.fit(X_train.iloc[:10], y_train_obs[:10])
        device_choice = "cuda"
        print("Using GPU (device='cuda') for XGBoost acceleration.")
    except Exception as e:
        device_choice = "cpu"
        print(f"GPU unavailable for tree execution ({e}), falling back to CPU (tree_method='hist').")

    # =========================================================================
    # 4.4 BASELINES
    # =========================================================================
    print(f"\n[{datetime.now()}] Computing Baseline Models on Validation Fold...")
    # Naive lag1
    y_pred_naive = val_fold['sales_lag1'].to_numpy()
    # Seasonal Naive lag7
    y_pred_snaive = val_fold['sales_lag7'].to_numpy()
    # 7-day Moving Average
    y_pred_ma7 = val_fold['sales_roll7'].to_numpy()

    metrics_naive = compute_metrics(y_val_obs, y_pred_naive)
    metrics_snaive = compute_metrics(y_val_obs, y_pred_snaive)
    metrics_ma7 = compute_metrics(y_val_obs, y_pred_ma7)
    
    print(f"Naive (lag1):      MAE={metrics_naive['mae']:.4f}, RMSE={metrics_naive['rmse']:.4f}, R2={metrics_naive['r2']:.4f}")
    print(f"Seasonal (lag7):   MAE={metrics_snaive['mae']:.4f}, RMSE={metrics_snaive['rmse']:.4f}, R2={metrics_snaive['r2']:.4f}")
    print(f"7-day MA:          MAE={metrics_ma7['mae']:.4f}, RMSE={metrics_ma7['rmse']:.4f}, R2={metrics_ma7['r2']:.4f}")

    # =========================================================================
    # 4.1 OBJECTIVE COMPARISON (Tweedie vs Squared Error vs Poisson)
    # =========================================================================
    print(f"\n[{datetime.now()}] Comparing XGBoost Objectives (Tweedie vs SquaredError vs Poisson)...")
    objectives = {
        "reg:squarederror": {},
        "count:poisson": {"max_delta_step": 0.7},
        "reg:tweedie": {"tweedie_variance_power": 1.3}
    }
    obj_results = {}
    
    for obj_name, extra_params in objectives.items():
        model_obj = xgb.XGBRegressor(
            objective=obj_name,
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device_choice,
            enable_categorical=True,
            random_state=42,
            early_stopping_rounds=30,
            **extra_params
        )
        model_obj.fit(X_train, y_train_obs, eval_set=[(X_val, y_val_obs)], verbose=False)
        preds = np.clip(model_obj.predict(X_val), 0.0, None)
        met = compute_metrics(y_val_obs, preds)
        obj_results[obj_name] = met
        print(f"Objective '{obj_name}': MAE={met['mae']:.4f}, RMSE={met['rmse']:.4f}, R2={met['r2']:.4f}, Bias={met['bias']:.4f}")

    # =========================================================================
    # 4.1 OPTUNA HYPERPARAMETER TUNING
    # =========================================================================
    print(f"\n[{datetime.now()}] Running Optuna Hyperparameter Tuning for XGBoost Tweedie...")
    trial_budget = 10
    
    def objective(trial):
        params = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.6, step=0.1),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_estimators": 400,
            "tree_method": "hist",
            "device": device_choice,
            "enable_categorical": True,
            "random_state": 42,
            "early_stopping_rounds": 25
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train_obs, eval_set=[(X_val, y_val_obs)], verbose=False)
        p = np.clip(m.predict(X_val), 0.0, None)
        return mean_absolute_error(y_val_obs, p)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trial_budget)
    best_params = study.best_params
    print(f"Optuna Best Params (Budget={trial_budget} trials): {best_params} | Best Val MAE={study.best_value:.4f}")

    # Train final tuned point model on observed target
    final_point_params = {
        "objective": "reg:tweedie",
        "n_estimators": 500,
        "tree_method": "hist",
        "device": device_choice,
        "enable_categorical": True,
        "random_state": 42,
        "early_stopping_rounds": 30,
        **best_params
    }
    best_xgb_obs = xgb.XGBRegressor(**final_point_params)
    best_xgb_obs.fit(X_train, y_train_obs, eval_set=[(X_val, y_val_obs)], verbose=False)
    val_pred_xgb = np.clip(best_xgb_obs.predict(X_val), 0.0, None)
    metrics_xgb_obs = compute_metrics(y_val_obs, val_pred_xgb)
    print(f"Tuned XGBoost (Observed Target): MAE={metrics_xgb_obs['mae']:.4f}, RMSE={metrics_xgb_obs['rmse']:.4f}, R2={metrics_xgb_obs['r2']:.4f}")

    # =========================================================================
    # 3b TARGET COMPARISON: Observed vs Latent Demand
    # =========================================================================
    print(f"\n[{datetime.now()}] Phase 3b: Training XGBoost with Latent Demand Target...")
    best_xgb_latent = xgb.XGBRegressor(**final_point_params)
    best_xgb_latent.fit(X_train, y_train_latent, eval_set=[(X_val, y_val_latent)], verbose=False)
    val_pred_latent = np.clip(best_xgb_latent.predict(X_val), 0.0, None)
    
    # Evaluate latent model against observed sales on validation
    metrics_xgb_latent_on_obs = compute_metrics(y_val_obs, val_pred_latent)
    metrics_xgb_latent_on_latent = compute_metrics(y_val_latent, val_pred_latent)
    print(f"XGBoost Latent Model evaluated vs Observed Sales: MAE={metrics_xgb_latent_on_obs['mae']:.4f}, RMSE={metrics_xgb_latent_on_obs['rmse']:.4f}, Bias={metrics_xgb_latent_on_obs['bias']:.4f}")
    print(f"XGBoost Latent Model evaluated vs Latent Demand:  MAE={metrics_xgb_latent_on_latent['mae']:.4f}, RMSE={metrics_xgb_latent_on_latent['rmse']:.4f}")

    # =========================================================================
    # 4.2 QUANTILE FORECASTS (0.50, 0.90, 0.95)
    # =========================================================================
    print(f"\n[{datetime.now()}] Phase 4.2: Training XGBoost Quantile Models (0.50, 0.90, 0.95)...")
    quantiles = [0.50, 0.90, 0.95]
    quantile_models = {}
    quantile_val_preds = {}
    quantile_calibration = {}

    for q in quantiles:
        q_model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=350,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device_choice,
            enable_categorical=True,
            random_state=42,
            early_stopping_rounds=25
        )
        q_model.fit(X_train, y_train_latent, eval_set=[(X_val, y_val_latent)], verbose=False)
        q_preds = np.clip(q_model.predict(X_val), 0.0, None)
        quantile_models[q] = q_model
        quantile_val_preds[q] = q_preds
        
        # Calibration: fraction of actuals <= predicted quantile
        coverage = float(np.mean(y_val_obs <= q_preds))
        quantile_calibration[q] = {
            "nominal": q,
            "empirical_coverage": coverage,
            "pinball_loss": float(np.mean(np.maximum(q * (y_val_obs - q_preds), (1 - q) * (q_preds - y_val_obs))))
        }
        print(f"Quantile q={q:.2f}: Empirical Coverage={coverage*100:.2f}% (Nominal={q*100:.0f}%), Pinball={quantile_calibration[q]['pinball_loss']:.4f}")

    # =========================================================================
    # 4.3 VOLATILITY & LIFECYCLE ROUTING
    # =========================================================================
    print(f"\n[{datetime.now()}] Phase 4.3: Testing Volatility- and Lifecycle-Aware Routing...")
    vol_classes = classify_syntetos_boylan(train_fold)
    
    # Precompute Croston SBA and Holt forecasts on training fold history
    croston_preds = {}
    for (s_id, p_id), grp in train_fold.groupby(['store_id', 'product_id'], observed=True):
        croston_preds[(str(s_id), str(p_id))] = croston_sba(grp['sale_amount'].values)
        
    routed_preds = []
    val_vol_tags = []
    
    for idx, row in val_fold.iterrows():
        key = (str(row['store_id']), str(row['product_id']))
        vol = vol_classes.get(key, 'smooth')
        val_vol_tags.append(vol)
        stage = row['lifecycle_stage']
        xgb_p = val_pred_xgb[len(routed_preds)]
        
        if vol in ['intermittent', 'lumpy']:
            # Blend Croston SBA with XGBoost (0.4 Croston + 0.6 XGBoost)
            c_val = croston_preds.get(key, xgb_p)
            routed_p = 0.4 * c_val + 0.6 * xgb_p
        elif stage == 'end_of_life':
            # Trend damping blend
            ma_p = row['sales_roll7']
            routed_p = 0.5 * ma_p + 0.5 * xgb_p
        else:
            routed_p = xgb_p
            
        routed_preds.append(max(0.0, float(routed_p)))
        
    routed_preds = np.array(routed_preds)
    metrics_routed = compute_metrics(y_val_obs, routed_preds)
    print(f"Global XGBoost: MAE={metrics_xgb_obs['mae']:.4f}, RMSE={metrics_xgb_obs['rmse']:.4f}, R2={metrics_xgb_obs['r2']:.4f}")
    print(f"Routed Model:   MAE={metrics_routed['mae']:.4f}, RMSE={metrics_routed['rmse']:.4f}, R2={metrics_routed['r2']:.4f}")
    
    # Decision Gate: Keep router ONLY if it strictly beats Global XGBoost on validation
    router_beats_global = metrics_routed['mae'] < metrics_xgb_obs['mae']
    print(f"Router Decision Gate Passed: {router_beats_global} (Router MAE {metrics_routed['mae']:.4f} vs Global {metrics_xgb_obs['mae']:.4f})")

    # =========================================================================
    # OFFICIAL EVAL SPLIT PREDICTIONS
    # =========================================================================
    print(f"\n[{datetime.now()}] Generating Official Eval Split Predictions...")
    eval_pred_xgb = np.clip(best_xgb_obs.predict(X_eval), 0.0, None)
    eval_pred_latent = np.clip(best_xgb_latent.predict(X_eval), 0.0, None)
    
    eval_q_preds = {}
    for q in quantiles:
        eval_q_preds[q] = np.clip(quantile_models[q].predict(X_eval), 0.0, None)
        
    metrics_eval_xgb = compute_metrics(y_eval_obs, eval_pred_xgb)
    metrics_eval_latent = compute_metrics(y_eval_obs, eval_pred_latent)
    print(f"Eval Split Performance (XGB Point):  MAE={metrics_eval_xgb['mae']:.4f}, RMSE={metrics_eval_xgb['rmse']:.4f}, R2={metrics_eval_xgb['r2']:.4f}, WAPE={metrics_eval_xgb['wape']:.4f}")
    print(f"Eval Split Performance (Latent):     MAE={metrics_eval_latent['mae']:.4f}, RMSE={metrics_eval_latent['rmse']:.4f}, R2={metrics_eval_latent['r2']:.4f}, Bias={metrics_eval_latent['bias']:.4f}")

    # =========================================================================
    # SAVE MODELS & METRICS
    # =========================================================================
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    joblib.dump(best_xgb_obs, "models/model_point_xgb.pkl")
    joblib.dump(best_xgb_latent, "models/model_latent_demand.pkl")
    joblib.dump(quantile_models, "models/quantile_models.pkl")
    
    with open("models/features_list.json", "w") as f:
        json.dump(features, f, indent=2)
        
    training_summary = {
        "timestamp": str(datetime.now()),
        "optuna_budget": trial_budget,
        "best_hyperparams": best_params,
        "objectives_comparison": obj_results,
        "baselines_validation": {
            "naive": metrics_naive,
            "seasonal_naive": metrics_snaive,
            "moving_avg_7": metrics_ma7
        },
        "validation_models": {
            "global_xgboost_point": metrics_xgb_obs,
            "latent_target_model_on_obs": metrics_xgb_latent_on_obs,
            "latent_target_model_on_latent": metrics_xgb_latent_on_latent,
            "routed_model": metrics_routed,
            "router_adopted": bool(router_beats_global)
        },
        "quantile_calibration": quantile_calibration,
        "eval_split_metrics": {
            "point_model": metrics_eval_xgb,
            "latent_model": metrics_eval_latent
        }
    }
    
    with open("models/training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
        
    # Save evaluation predictions table for downstream inventory and anomaly modules
    eval_df['pred_point'] = eval_pred_xgb
    eval_df['pred_latent'] = eval_pred_latent
    eval_df['pred_q50'] = eval_q_preds[0.50]
    eval_df['pred_q90'] = eval_q_preds[0.90]
    eval_df['pred_q95'] = eval_q_preds[0.95]
    eval_df.to_parquet("data/processed/eval_predictions.parquet", index=False)
    print("Saved models, metadata, and eval predictions to models/ and data/processed/eval_predictions.parquet")

    return training_summary

if __name__ == "__main__":
    train_and_evaluate_all()
