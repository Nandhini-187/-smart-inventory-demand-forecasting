"""
Shared Inventory Logic Module for Retail Store Smart Inventory Management.
Provides core functions:
1. Contiguous date and minimum-history time series validation
2. Leakage-safe feature engineering contract
3. Non-negative demand inference
4. Stockout-history risk assessment and forecast-based stock coverage calculation
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    REQUIRED_COLUMNS,
    EXPECTED_FEATURES,
    TARGET_COLUMN,
    SAFETY_FACTOR,
    PLANNING_HORIZON_DAYS,
    MIN_REQUIRED_DAYS,
    RISK_THRESHOLD_MEDIUM,
    RISK_THRESHOLD_HIGH,
    RISK_QUANTILE_MEDIUM,
    RISK_QUANTILE_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
    RISK_LEVEL_HIGH,
    ACTION_LOW,
    ACTION_MEDIUM,
    ACTION_HIGH,
)


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate that required columns are present in input DataFrame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")


def validate_series_slice_length(df_series: pd.DataFrame, max_limit: int = 1000) -> None:
    """
    Enforce runtime row-count invariant: individual store-product series
    must contain fewer than max_limit rows (typically ~90 daily records).
    Prevents accidental full-dataset materialization.
    """
    if len(df_series) >= max_limit:
        raise AssertionError(
            f"Series length assertion failed: {len(df_series)} rows exceeds safety limit of {max_limit}. "
            f"Series must be sliced to an individual store-product key before materialization."
        )


def validate_series_continuity(df_series: pd.DataFrame) -> None:
    """
    Validate that an individual store-product series has sufficient history
    and contiguous daily observations without gaps.
    
    Because shift() moves by row rather than calendar date, gaps in dates
    would silently misalign 1-day and 7-day lag and rolling features.
    """
    validate_series_slice_length(df_series)
    if len(df_series) < MIN_REQUIRED_DAYS:
        raise ValueError(
            f"Insufficient historical data: series has {len(df_series)} records. "
            f"At least {MIN_REQUIRED_DAYS} contiguous daily records are required to compute 7-day lag and rolling features."
        )
        
    date_diffs = df_series["dt"].diff().dropna()
    gaps = date_diffs[date_diffs != pd.Timedelta(days=1)]
    if not gaps.empty:
        discontinuous_dates = df_series.loc[gaps.index, "dt"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"Date discontinuity detected: series contains missing calendar days near {discontinuous_dates[:3]}. "
            f"Lags and rolling windows require contiguous daily observations."
        )


def engineer_features(
    df: pd.DataFrame, 
    include_target: bool = False,
    enforce_continuity: bool = False
) -> pd.DataFrame:
    """
    Compute leakage-safe features for next-day demand forecasting.
    
    Exact Contract:
      Sort by [store_id, product_id, dt]; group by [store_id, product_id]:
        lag_1              = sale_amount.shift(1)  (sales on day t-1)
        lag_7              = sale_amount.shift(7)  (sales on day t-7: 8-day lag to target t+1, NOT weekly recurrence)
        rolling_7          = sale_amount.transform(lambda x: x.shift(1).rolling(7).mean()) (mean sales t-7..t-1)
        stockout_lag_1     = stock_hour6_22_cnt.shift(1) (stockout hours on day t-1)
        stockout_lag_7     = stock_hour6_22_cnt.shift(7) (stockout hours on day t-7)
        stockout_rolling_7 = stock_hour6_22_cnt.transform(lambda x: x.shift(1).rolling(7).sum()) (sum t-7..t-1)
        target_next_day    = sale_amount.shift(-1)  (target sales on day t+1; only when include_target=True)
        
    Horizon & Alignment Note:
      For a row dated day t, the target is day t+1.
      lag_7 is the sale on day t-7, which is (t+1) - (t-7) = 8 days before the target.
      Because 8 mod 7 = 1, lag_7 falls on a DIFFERENT weekday than target t+1.
      A same-weekday lag relative to target t+1 would be t-6 (sale_amount.shift(6)).
      Sales and stockouts on day t itself are excluded from this frozen feature contract.
      The shift(1) before every rolling window is mandatory to prevent leakage.
    """
    validate_input_dataframe(df)
    
    out = df.copy()
    out["dt"] = pd.to_datetime(out["dt"])
    out = out.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
    
    if enforce_continuity:
        for (st_id, pr_id), sub_df in out.groupby(["store_id", "product_id"], sort=False):
            validate_series_continuity(sub_df)
    
    grp = out.groupby(["store_id", "product_id"], sort=False)
    
    # 1. Demand lag features
    out["lag_1"] = grp["sale_amount"].shift(1)
    out["lag_7"] = grp["sale_amount"].shift(7)
    
    # 2. Demand rolling feature (7-day rolling mean demand level; shift(1) prevents leakage)
    out["rolling_7"] = grp["sale_amount"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    
    # 3. Stockout lag features
    out["stockout_lag_1"] = grp["stock_hour6_22_cnt"].shift(1)
    out["stockout_lag_7"] = grp["stock_hour6_22_cnt"].shift(7)
    
    # 4. Stockout rolling feature (7-day cumulative stockout hours; shift(1) prevents leakage)
    out["stockout_rolling_7"] = grp["stock_hour6_22_cnt"].transform(
        lambda x: x.shift(1).rolling(7).sum()
    )
    
    if include_target:
        out[TARGET_COLUMN] = grp["sale_amount"].shift(-1)
        
    return out


def predict_demand(booster: xgb.Booster, feature_vector: np.ndarray) -> float:
    """
    Perform point forecast on feature vector with non-negativity constraint.
    predicted_next_day_demand = max(model_prediction, 0)
    """
    if isinstance(feature_vector, (list, tuple)):
        feature_vector = np.array(feature_vector, dtype=np.float32)
        
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)
        
    dmatrix = xgb.DMatrix(feature_vector, feature_names=EXPECTED_FEATURES)
    raw_pred = booster.predict(dmatrix)[0]
    
    # Enforce non-negative demand constraint
    return float(max(raw_pred, 0.0))


def assess_stockout_risk(
    recent_stockout_hours: float,
    mode: str = "fixed"
) -> Tuple[str, str, str]:
    """
    Stockout-History Risk Assessment:
    Evaluated strictly from preceding 7-day cumulative stockout hours (stockout_rolling_7).
    Note: Risk reflects past stockout vulnerability, NOT the model demand forecast.
    
    Modes:
      - 'fixed' (default heuristic):
          LOW if < 8 h -> 'NO IMMEDIATE ACTION'
          MEDIUM if 8-23 h -> 'MONITOR / REPLENISH'
          HIGH if >= 24 h -> 'REPLENISH NOW'
      - 'quantile' (mitigates alert fatigue):
          LOW if < 29 h (bottom 70% of historical series)
          MEDIUM if 29-47 h (70th to 90th percentile)
          HIGH if >= 48 h (top 10% most vulnerable series)
      
    Returns: (risk_level, action_text, css_class)
    """
    hours = float(recent_stockout_hours)
    
    if mode == "quantile":
        thresh_high = RISK_QUANTILE_HIGH      # 48.0 hrs
        thresh_med = RISK_QUANTILE_MEDIUM     # 29.0 hrs
    else:
        thresh_high = RISK_THRESHOLD_HIGH     # 24.0 hrs
        thresh_med = RISK_THRESHOLD_MEDIUM    # 8.0 hrs

    if hours >= thresh_high:
        return RISK_LEVEL_HIGH, ACTION_HIGH, "risk-high"
    elif hours >= thresh_med:
        return RISK_LEVEL_MEDIUM, ACTION_MEDIUM, "risk-medium"
    else:
        return RISK_LEVEL_LOW, ACTION_LOW, "risk-low"


def compute_inventory_decision(
    predicted_next_day_demand: float,
    rolling_7: float,
    stockout_rolling_7: float,
    risk_mode: str = "fixed"
) -> Dict[str, Any]:
    """
    Computes forecast-based stock coverage and stockout-history risk.
    
    Formulas:
      predicted_next_day_demand = max(model_prediction, 0)
      predicted_7day_demand     = predicted_next_day_demand * 7
      safety_buffer             = rolling_7 * 0.50  (50% of preceding 7-day rolling mean demand level)
      recommended_stock_coverage = predicted_7day_demand + safety_buffer
      recent_stockout_hours     = stockout_rolling_7 (preceding 7-day stockout hours)
      
    Risk Modes:
      - 'fixed': <8h, 8-23h, >=24h
      - 'quantile': <29h, 29-47h, >=48h (top 10% HIGH)
    """
    next_day_demand = max(float(predicted_next_day_demand), 0.0)
    predicted_7day = next_day_demand * float(PLANNING_HORIZON_DAYS)
    
    mean_demand_level = float(rolling_7) if rolling_7 is not None and not np.isnan(rolling_7) else 0.0
    safety_buff = mean_demand_level * float(SAFETY_FACTOR)
    rec_coverage = predicted_7day + safety_buff
    
    stockout_hours = float(stockout_rolling_7) if stockout_rolling_7 is not None and not np.isnan(stockout_rolling_7) else 0.0
    risk_level, action_text, css_class = assess_stockout_risk(stockout_hours, mode=risk_mode)
    
    return {
        "predicted_next_day_demand": round(next_day_demand, 4),
        "predicted_7day_demand": round(predicted_7day, 4),
        "mean_demand_level": round(mean_demand_level, 4),
        "safety_factor": SAFETY_FACTOR,
        "safety_buffer": round(safety_buff, 4),
        "recommended_stock_coverage": round(rec_coverage, 4),
        "recent_stockout_hours": round(stockout_hours, 2),
        "risk_level": risk_level,
        "action_text": action_text,
        "css_class": css_class,
        "risk_mode": risk_mode
    }
