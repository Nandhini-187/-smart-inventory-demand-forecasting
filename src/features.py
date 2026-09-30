"""
Phase 3: Feature Engineering & Target Construction
FreshRetailNet-50K Demand Forecasting & Inventory Management
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

def compute_daily_building_blocks(df):
    """
    Computes same-day building blocks from raw columns (hourly arrays, stockouts).
    NOTE: These building blocks CANNOT be used directly by the model on day t;
    they MUST be lagged (shift(1)) before rolling or feeding into feature matrices.
    """
    # 1. Trading hours: 6 to 22 inclusive (17 trading hours)
    stockout_hours_list = []
    in_stock_rate_list = []
    peak_hour_list = []
    morning_share_list = []
    evening_share_list = []
    latent_demand_list = []

    hs_all = df['hours_sale'].values
    st_all = df['hours_stock_status'].values
    sales_all = df['sale_amount'].values

    for hs, st, sale in zip(hs_all, st_all, sales_all):
        hs_arr = np.asarray(hs, dtype=np.float32)
        st_arr = np.asarray(st, dtype=np.int32)
        trading_hs = hs_arr[6:23]
        trading_st = st_arr[6:23]

        stockout_hrs = int(np.sum(trading_st == 1))
        in_stock_hrs = 17 - stockout_hrs

        if in_stock_hrs > 0:
            # Sales during in-stock trading hours
            in_stock_sales = float(np.sum(trading_hs[trading_st == 0]))
            hourly_rate = float(in_stock_sales / in_stock_hrs)
        else:
            hourly_rate = 0.0

        # Latent uncensored demand = sale_amount + in_stock_hourly_rate * stockout_hours
        latent_d = float(sale + hourly_rate * stockout_hrs)

        # Peak hour across 24h
        peak_h = int(np.argmax(hs_arr))

        # Morning (6-12) vs Evening (17-22) sales share
        total_hs = max(float(np.sum(hs_arr)), 1e-5)
        morn_sales = float(np.sum(hs_arr[6:13]))
        eve_sales = float(np.sum(hs_arr[17:23]))

        stockout_hours_list.append(stockout_hrs)
        in_stock_rate_list.append(hourly_rate)
        peak_hour_list.append(peak_h)
        morning_share_list.append(morn_sales / total_hs)
        evening_share_list.append(eve_sales / total_hs)
        latent_demand_list.append(latent_d)

    df['same_day_stockout_hrs'] = np.array(stockout_hours_list, dtype=np.float32)
    df['same_day_stockout_ratio'] = df['same_day_stockout_hrs'] / 17.0
    df['same_day_in_stock_rate'] = np.array(in_stock_rate_list, dtype=np.float32)
    df['same_day_peak_hour'] = np.array(peak_hour_list, dtype=np.int16)
    df['same_day_morning_share'] = np.array(morning_share_list, dtype=np.float32)
    df['same_day_evening_share'] = np.array(evening_share_list, dtype=np.float32)
    df['latent_demand'] = np.array(latent_demand_list, dtype=np.float32)
    return df

def fit_transform_features(train_df, eval_df=None):
    """
    Constructs the 63 features across all feature groups with strict time-based shifting.
    Combines train and eval (if provided) to ensure seamless lag computation at the boundary,
    then separates them back.
    """
    print(f"[{datetime.now()}] Starting feature engineering...")
    
    # Track train length to split back later
    train_len = len(train_df)
    train_df = train_df.copy()
    train_df['is_eval_split'] = 0
    
    if eval_df is not None:
        eval_df = eval_df.copy()
        eval_df['is_eval_split'] = 1
        df = pd.concat([train_df, eval_df], ignore_index=True)
    else:
        df = train_df

    # Sort strictly by series and date
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.sort_values(by=['store_id', 'product_id', 'dt']).reset_index(drop=True)

    # 1. Compute same-day building blocks
    df = compute_daily_building_blocks(df)

    # Groupby object for lagged calculations
    grp = df.groupby(['store_id', 'product_id'], observed=True)

    # 2. Sales history (Shifted by 1 to prevent any leakage!)
    shifted_sales = grp['sale_amount'].shift(1)
    df['sales_lag1'] = shifted_sales
    df['sales_lag2'] = grp['sale_amount'].shift(2)
    df['sales_lag7'] = grp['sale_amount'].shift(7)
    df['sales_lag14'] = grp['sale_amount'].shift(14)

    # Rolling statistics on shifted_sales
    df['sales_roll3'] = grp['sales_lag1'].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df['sales_roll7'] = grp['sales_lag1'].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df['sales_roll14'] = grp['sales_lag1'].transform(lambda s: s.rolling(14, min_periods=1).mean())

    df['sales_roll7_std'] = grp['sales_lag1'].transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0.0)
    df['sales_roll7_max'] = grp['sales_lag1'].transform(lambda s: s.rolling(7, min_periods=1).max())
    df['sales_roll7_min'] = grp['sales_lag1'].transform(lambda s: s.rolling(7, min_periods=1).min())

    df['sales_trend'] = (df['sales_roll7'] / (df['sales_roll14'] + 1e-4)).clip(upper=10.0)
    df['zero_days_7'] = grp['sales_lag1'].transform(lambda s: (s == 0).rolling(7, min_periods=1).sum())

    # Days since last non-zero sale
    def calc_days_since_sale(s):
        last_idx = -999
        res = []
        for i, val in enumerate(s):
            if i == 0 or np.isnan(val):
                res.append(99)
            else:
                prev_val = s.iloc[i-1]
                if prev_val > 0:
                    last_idx = i - 1
                res.append(i - last_idx if last_idx >= 0 else 99)
        return pd.Series(res, index=s.index)

    df['days_since_sale'] = grp['sale_amount'].transform(calc_days_since_sale).clip(upper=90)
    df['sales_expanding_mean'] = grp['sales_lag1'].transform(lambda s: s.expanding().mean())

    # 3. Latent (uncensored) demand features (Strictly lagged)
    shifted_latent = grp['latent_demand'].shift(1)
    df['latent_lag1'] = shifted_latent
    df['latent_lag7'] = grp['latent_demand'].shift(7)
    df['latent_roll7'] = grp['latent_lag1'].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df['latent_roll14'] = grp['latent_lag1'].transform(lambda s: s.rolling(14, min_periods=1).mean())
    df['latent_roll7_std'] = grp['latent_lag1'].transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0.0)

    # 4. Stockout history features (Strictly lagged)
    shifted_st_ratio = grp['same_day_stockout_ratio'].shift(1)
    df['stockout_ratio_lag1'] = shifted_st_ratio
    df['stockout_ratio_lag7'] = grp['same_day_stockout_ratio'].shift(7)
    df['stockout_roll3'] = grp['stockout_ratio_lag1'].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df['stockout_roll7'] = grp['stockout_ratio_lag1'].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df['stockout_days_7'] = grp['stockout_ratio_lag1'].transform(lambda s: (s > 0).rolling(7, min_periods=1).sum())

    # Stockout streak (consecutive previous days with stockout > 0)
    def calc_stockout_streak(s):
        streak = 0
        res = []
        for val in s:
            if np.isnan(val) or val <= 0:
                streak = 0
            else:
                streak += 1
            res.append(streak)
        return pd.Series(res, index=s.index)

    df['stockout_streak_lag1'] = grp['stockout_ratio_lag1'].transform(calc_stockout_streak)
    df['stock_hour6_22_cnt_lag1'] = grp['stock_hour6_22_cnt'].shift(1)

    # First stockout hour lagged (first hour index where status was 1, or 24 if none)
    def get_first_st_hour(st):
        st_arr = np.asarray(st, dtype=np.int32)
        idx = np.where(st_arr[6:23] == 1)[0]
        return int(idx[0] + 6) if len(idx) > 0 else 24

    df['first_stockout_hour'] = [get_first_st_hour(st) for st in df['hours_stock_status']]
    df['first_stockout_hour_lag1'] = grp['first_stockout_hour'].shift(1)

    # 5. Hourly shape features (lagged)
    df['morning_sales_share_lag1'] = grp['same_day_morning_share'].shift(1)
    df['evening_sales_share_lag1'] = grp['same_day_evening_share'].shift(1)
    df['peak_hour_lag1'] = grp['same_day_peak_hour'].shift(1)
    df['in_stock_hourly_rate_lag1'] = grp['same_day_in_stock_rate'].shift(1)

    # 6. Price and promotions (Known in advance for target day)
    df['discount_lag1'] = grp['discount'].shift(1)
    df['discount_change'] = df['discount'] - df['discount_lag1'].fillna(df['discount'])
    df['discount_roll7'] = grp['discount'].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df['discount_vs_roll7'] = df['discount'] / (df['discount_roll7'] + 1e-4)
    df['is_discounted'] = (df['discount'] < 0.98).astype('int8')
    df['discount_depth'] = (1.0 - df['discount']).clip(lower=0.0)

    # Post-promo flag: discount was < 0.95 yesterday but returned to ~1.0 today
    df['post_promo_flag'] = ((df['discount_lag1'] < 0.95) & (df['discount'] >= 0.98)).astype('int8')

    # Calendar holiday distances
    holiday_dates = df[df['holiday_flag'] == 1]['dt'].drop_duplicates().sort_values()
    
    def calc_holiday_distance(dt_series, h_dates):
        to_next = []
        since_last = []
        for d in dt_series:
            future = h_dates[h_dates >= d]
            past = h_dates[h_dates <= d]
            to_next.append((future.iloc[0] - d).days if len(future) > 0 else 30)
            since_last.append((d - past.iloc[-1]).days if len(past) > 0 else 30)
        return pd.Series(to_next, index=dt_series.index), pd.Series(since_last, index=dt_series.index)

    df['days_to_next_holiday'], df['days_since_last_holiday'] = calc_holiday_distance(df['dt'], holiday_dates)
    df['days_to_next_holiday'] = df['days_to_next_holiday'].clip(upper=30)
    df['days_since_last_holiday'] = df['days_since_last_holiday'].clip(upper=30)

    # 7. Calendar features
    df['day_of_week'] = df['dt'].dt.dayofweek.astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    df['day_of_month'] = df['dt'].dt.day.astype('int8')

    # 8. Weather features (Stand-in for forecast + lags + deviations)
    if 'is_rainy' not in df.columns:
        df['is_rainy'] = (df['precpt'] > 0.1).astype('int8')
    weather_cols = ['precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    store_grp = df.groupby('store_id', observed=True)
    for col in weather_cols:
        df[f'{col}_lag1'] = store_grp[col].shift(1)
        df[f'{col}_change'] = df[col] - df[f'{col}_lag1'].fillna(df[col])
        store_mean = store_grp[col].transform('mean')
        df[f'{col}_deviation'] = df[col] - store_mean

    # 9. Hierarchy and Context features
    # Product in city lagged rolling sales
    city_prod_sales = df.groupby(['city_id', 'product_id', 'dt'], observed=True)['sale_amount'].transform('sum')
    df['city_product_sales_lag1'] = df.groupby(['city_id', 'product_id'], observed=True)['sale_amount'].shift(1)
    df['city_product_roll7'] = df.groupby(['city_id', 'product_id'], observed=True)['city_product_sales_lag1'].transform(lambda s: s.rolling(7, min_periods=1).mean())

    # Category in store lagged rolling sales
    cat_store_sales = df.groupby(['store_id', 'first_category_id', 'dt'], observed=True)['sale_amount'].transform('sum')
    df['cat_store_sales_lag1'] = df.groupby(['store_id', 'first_category_id'], observed=True)['sale_amount'].shift(1)
    df['cat_store_roll7'] = df.groupby(['store_id', 'first_category_id'], observed=True)['cat_store_sales_lag1'].transform(lambda s: s.rolling(7, min_periods=1).mean())

    # 10. Lifecycle features
    # days_since_first_sale
    def calc_days_since_first_sale(s):
        first_sale_idx = None
        res = []
        for i, val in enumerate(s):
            if val > 0 and first_sale_idx is None:
                first_sale_idx = i
            res.append(i - first_sale_idx if first_sale_idx is not None else 0)
        return pd.Series(res, index=s.index)

    df['days_since_first_sale'] = grp['sale_amount'].transform(calc_days_since_first_sale)

    # sales_slope_14: linear regression slope over past 14 days of shifted sales
    def calc_slope_14(s):
        x = np.arange(14)
        x_centered = x - np.mean(x)
        denom = np.sum(x_centered ** 2)
        res = []
        vals = s.to_numpy()
        for i in range(len(vals)):
            if i < 14:
                window = vals[:i+1]
                if len(window) < 3:
                    res.append(0.0)
                else:
                    xw = np.arange(len(window))
                    xw_c = xw - np.mean(xw)
                    res.append(float(np.sum(xw_c * window) / (np.sum(xw_c ** 2) + 1e-4)))
            else:
                window = vals[i-13:i+1]
                res.append(float(np.sum(x_centered * window) / denom))
        return pd.Series(res, index=s.index)

    df['sales_slope_14'] = grp['sales_lag1'].transform(calc_slope_14)

    # Lifecycle stage classification:
    # 'new': first non-zero sale occurs late in the window or < 21 days since first sale
    # 'end_of_life': sustained negative slope (< -0.05) with deep markdown (discount < 0.85) OR zero_days_7 >= 6 with prior sales
    # 'mature': otherwise
    is_new = (df['days_since_first_sale'] < 21)
    is_eol = ((df['sales_slope_14'] < -0.05) & (df['discount'] < 0.85)) | ((df['zero_days_7'] >= 6) & (df['days_since_first_sale'] >= 21))
    
    stage = np.where(is_new, 'new', np.where(is_eol, 'end_of_life', 'mature'))
    df['lifecycle_stage'] = pd.Categorical(stage, categories=['new', 'mature', 'end_of_life'])

    # Targets:
    # (a) observed sale_amount
    df['target_sales'] = df['sale_amount'].astype('float32')
    # (b) latent demand
    df['target_latent'] = df['latent_demand'].astype('float32')

    # Fill NA in lagged features resulting from shifts
    feature_cols = [
        # Sales history (13)
        'sales_lag1', 'sales_lag2', 'sales_lag7', 'sales_lag14',
        'sales_roll3', 'sales_roll7', 'sales_roll14',
        'sales_roll7_std', 'sales_roll7_max', 'sales_roll7_min',
        'sales_trend', 'zero_days_7', 'days_since_sale', 'sales_expanding_mean',
        # Latent demand (5)
        'latent_lag1', 'latent_lag7', 'latent_roll7', 'latent_roll14', 'latent_roll7_std',
        # Stockout history (8)
        'stockout_ratio_lag1', 'stockout_ratio_lag7', 'stockout_roll3', 'stockout_roll7',
        'stockout_days_7', 'stockout_streak_lag1', 'stock_hour6_22_cnt_lag1', 'first_stockout_hour_lag1',
        # Hourly shape (4)
        'morning_sales_share_lag1', 'evening_sales_share_lag1', 'peak_hour_lag1', 'in_stock_hourly_rate_lag1',
        # Price & Promotion (10)
        'discount', 'discount_lag1', 'discount_change', 'discount_vs_roll7',
        'is_discounted', 'discount_depth', 'post_promo_flag', 'activity_flag', 'holiday_flag',
        'days_to_next_holiday', 'days_since_last_holiday',
        # Calendar (3)
        'day_of_week', 'is_weekend', 'day_of_month',
        # Weather (13)
        'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level', 'is_rainy',
        'precpt_lag1', 'precpt_change', 'precpt_deviation',
        'avg_temperature_lag1', 'avg_temperature_change', 'avg_temperature_deviation',
        'avg_humidity_lag1', 'avg_humidity_change', 'avg_humidity_deviation',
        'avg_wind_level_lag1', 'avg_wind_level_change', 'avg_wind_level_deviation',
        # Hierarchy & Context (7)
        'city_id', 'store_id', 'management_group_id', 'first_category_id',
        'second_category_id', 'third_category_id', 'product_id',
        'city_product_sales_lag1', 'city_product_roll7', 'cat_store_sales_lag1', 'cat_store_roll7',
        # Lifecycle (4)
        'days_since_first_sale', 'sales_slope_14', 'lifecycle_stage'
    ]
    # Remove any duplicates in feature_cols list
    feature_cols = list(dict.fromkeys(feature_cols))

    # Filter feature cols to those present in df
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Total features defined: {len(feature_cols)}")

    # Impute numeric NaNs in lagged features with 0
    numeric_feature_cols = [c for c in feature_cols if c in df.columns and c not in ['city_id', 'store_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id', 'product_id', 'lifecycle_stage']]
    df[numeric_feature_cols] = df[numeric_feature_cols].fillna(0.0)

    # Separate back train and eval
    train_feat_df = df[df['is_eval_split'] == 0].drop(columns=['is_eval_split']).reset_index(drop=True)
    if eval_df is not None:
        eval_feat_df = df[df['is_eval_split'] == 1].drop(columns=['is_eval_split']).reset_index(drop=True)
    else:
        eval_feat_df = None

    print(f"[{datetime.now()}] Feature engineering complete! Train: {len(train_feat_df)} rows, Eval: {len(eval_feat_df) if eval_feat_df is not None else 0} rows")
    return train_feat_df, eval_feat_df, feature_cols

if __name__ == "__main__":
    train_clean = pd.read_parquet("data/processed/clean_train.parquet")
    eval_clean = pd.read_parquet("data/processed/clean_eval.parquet")
    train_feat, eval_feat, feature_cols = fit_transform_features(train_clean, eval_clean)
    
    # Save processed features (excluding bulky hourly lists to optimize storage/speed)
    drop_cols = ['hours_sale', 'hours_stock_status']
    train_feat = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
    eval_feat = eval_feat.drop(columns=[c for c in drop_cols if c in eval_feat.columns])
    
    train_feat.to_parquet("data/processed/train_features.parquet", index=False)
    eval_feat.to_parquet("data/processed/eval_features.parquet", index=False)
    print("Saved feature tables to data/processed/train_features.parquet and eval_features.parquet")
