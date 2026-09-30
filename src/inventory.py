"""
Phase 6: Inventory Optimization & Constrained Stock Allocation
FreshRetailNet-50K Demand Forecasting & Inventory Management
"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from datetime import datetime

def load_inventory_config(config_path="config/inventory.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_safety_stock_and_rop(
    df_pred,
    config,
    lead_time_mean=2.0,
    lead_time_std=0.5,
    z=1.645
):
    """
    Computes Dynamic Reorder Point (ROP) two ways:
    1. Parametric: z * sqrt(LT * sigma_d^2 + d^2 * sigma_LT^2) + d * LT
    2. Quantile-based: using XGBoost 0.95 quantile forecast scaled over lead time.
    """
    # Recent velocity based on latent demand forecast
    d_hat = np.maximum(df_pred['pred_latent'].to_numpy(dtype=np.float64), 0.0)
    sigma_d = np.maximum(df_pred['sales_roll7_std'].to_numpy(dtype=np.float64), 0.1)

    # 1. Parametric Safety Stock
    # Formula: z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
    var_demand_during_lt = lead_time_mean * (sigma_d ** 2) + (d_hat ** 2) * (lead_time_std ** 2)
    ss_parametric = z * np.sqrt(np.maximum(var_demand_during_lt, 0.0))
    rop_parametric = d_hat * lead_time_mean + ss_parametric

    # 2. Quantile-based Safety Stock
    # Uses 0.95 quantile demand forecast
    q95_forecast = np.maximum(df_pred['pred_q95'].to_numpy(dtype=np.float64), d_hat)
    # Lead-time scaling (square-root scaling of safety buffer over review+lead period)
    quantile_buffer = np.maximum(q95_forecast - d_hat, 0.0) * np.sqrt(lead_time_mean)
    ss_quantile = quantile_buffer
    rop_quantile = d_hat * lead_time_mean + ss_quantile

    return ss_parametric, rop_parametric, ss_quantile, rop_quantile

def generate_reorder_recommendations(
    predictions_path="data/processed/eval_predictions.parquet",
    config_path="config/inventory.yaml",
    output_path="outputs/reorder_recommendations.csv"
):
    print(f"[{datetime.now()}] Generating Reorder Recommendations...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    config = load_inventory_config(config_path)
    
    df = pd.read_parquet(predictions_path)
    # Pick the latest available day to generate actionable replenishment recommendations
    latest_dt = df['dt'].max()
    df_day = df[df['dt'] == latest_dt].copy().reset_index(drop=True)
    
    lt_mean = config['assumptions']['default_lead_time_mean_days']
    lt_std = config['assumptions']['default_lead_time_std_days']
    z_val = config['assumptions']['z_service_level']
    review_period = config['assumptions']['review_period_days']

    ss_param, rop_param, ss_q, rop_q = compute_safety_stock_and_rop(
        df_day, config, lead_time_mean=lt_mean, lead_time_std=lt_std, z=z_val
    )

    df_day['safety_stock_parametric'] = np.round(ss_param, 2)
    df_day['rop_parametric'] = np.round(rop_param, 2)
    df_day['safety_stock_quantile'] = np.round(ss_q, 2)
    df_day['rop_quantile'] = np.round(rop_q, 2)

    # Simulated On-Hand Stock (Clearly labelled as an assumption/simulation)
    # Base simulated stock on ~2 days of supply with random variance
    np.random.seed(42)
    daily_vel = np.maximum(df_day['pred_latent'].values, 0.5)
    simulated_on_hand = np.round(daily_vel * np.random.uniform(0.5, 3.0, size=len(df_day)), 1)
    df_day['stock_on_hand_simulated'] = simulated_on_hand

    # Order-Up-To Level (OUT): Demand over (Lead Time + Review Period) + Safety Stock
    out_level = daily_vel * (lt_mean + review_period) + ss_q
    df_day['order_up_to_level'] = np.round(out_level, 2)

    # Recommended Order Quantity: max(0, OUT - On Hand) if On Hand <= ROP
    reorder_needed = df_day['stock_on_hand_simulated'] <= df_day['rop_quantile']
    order_qty = np.where(reorder_needed, np.maximum(df_day['order_up_to_level'] - df_day['stock_on_hand_simulated'], 0.0), 0.0)
    df_day['recommended_order_qty'] = np.round(order_qty, 2)

    # Days of Cover: On Hand / Daily Velocity
    df_day['days_of_cover'] = np.round(df_day['stock_on_hand_simulated'] / daily_vel, 2)

    # Priority Rank: higher priority for lower days of cover and urgent stockouts
    df_day['urgency_score'] = np.where(df_day['days_of_cover'] < 1.0, 100.0, 10.0) / (df_day['days_of_cover'] + 0.1)
    recs = df_day.sort_values(by='urgency_score', ascending=False).reset_index(drop=True)

    keep_cols = [
        'store_id', 'product_id', 'dt', 'pred_latent', 'pred_q95',
        'stock_on_hand_simulated', 'rop_quantile', 'safety_stock_quantile',
        'rop_parametric', 'safety_stock_parametric',
        'order_up_to_level', 'recommended_order_qty', 'days_of_cover'
    ]
    recs[keep_cols].to_csv(output_path, index=False)
    print(f"Reorder recommendations written to {output_path} ({len(recs)} store-product series)")
    return recs[keep_cols]

def smart_multi_store_allocation(product_id=None, total_dc_pool=500.0):
    """
    Allocates central warehouse/DC stock pool to stores for a given product
    using Linear Programming (scipy.optimize.linprog) to minimize expected shortage.
    """
    df = pd.read_parquet("data/processed/eval_predictions.parquet")
    latest_dt = df['dt'].max()
    df_latest = df[df['dt'] == latest_dt]
    
    if product_id is None:
        product_id = df_latest['product_id'].iloc[0]
        
    sub = df_latest[df_latest['product_id'].astype(str) == str(product_id)].copy().reset_index(drop=True)
    n_stores = len(sub)
    if n_stores == 0:
        return None

    # Demand forecast per store
    demands = np.maximum(sub['pred_latent'].to_numpy(dtype=np.float64), 0.1)
    # Simulated current store stock
    stocks = np.maximum(demands * 0.5, 0.0)
    # Desired replenishment need: max(0, demand - stock)
    needs = np.maximum(demands - stocks, 0.0)
    
    total_need = np.sum(needs)
    
    # 1. Optimal LP Allocation
    # Minimize sum of shortages: s_i >= demand_i - (stock_i + x_i)
    # Decision variables: [x_1..x_n, s_1..s_n]
    # Minimize sum(s_i)
    # Subject to:
    #   sum(x_i) <= total_dc_pool
    #   s_i + x_i >= demand_i - stock_i
    #   x_i >= 0, s_i >= 0
    c = np.concatenate([np.zeros(n_stores), np.ones(n_stores)])
    
    # Inequality constraints: A_ub * var <= b_ub
    # Pool limit: sum(x_i) <= total_dc_pool
    A_pool = np.concatenate([np.ones((1, n_stores)), np.zeros((1, n_stores))], axis=1)
    b_pool = np.array([total_dc_pool])
    
    # Shortage constraints: -x_i - s_i <= -(demand_i - stock_i)
    A_shortage = np.zeros((n_stores, 2 * n_stores))
    for i in range(n_stores):
        A_shortage[i, i] = -1.0
        A_shortage[i, n_stores + i] = -1.0
    b_shortage = -(demands - stocks)
    
    A_ub = np.vstack([A_pool, A_shortage])
    b_ub = np.concatenate([b_pool, b_shortage])
    bounds = [(0, None)] * (2 * n_stores)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        alloc_lp = res.x[:n_stores]
    else:
        # Fallback to greedy proportional-to-need
        ratio = min(1.0, total_dc_pool / max(total_need, 1e-4))
        alloc_lp = needs * ratio

    # 2. Naive Equal Allocation Baseline
    alloc_naive = np.full(n_stores, total_dc_pool / max(n_stores, 1))

    # Evaluate fill rates: sum(min(demand, stock + alloc)) / sum(demand)
    fulfilled_lp = np.sum(np.minimum(demands, stocks + alloc_lp))
    fulfilled_naive = np.sum(np.minimum(demands, stocks + alloc_naive))
    tot_d = np.sum(demands)
    
    fill_rate_lp = fulfilled_lp / tot_d
    fill_rate_naive = fulfilled_naive / tot_d

    sub['allocated_units_optimal'] = np.round(alloc_lp, 2)
    sub['allocated_units_naive'] = np.round(alloc_naive, 2)
    sub['initial_stock_simulated'] = np.round(stocks, 2)
    sub['demand_forecast'] = np.round(demands, 2)
    
    out_table = sub[['store_id', 'product_id', 'demand_forecast', 'initial_stock_simulated', 'allocated_units_optimal', 'allocated_units_naive']]
    return {
        "product_id": str(product_id),
        "total_pool_units": float(total_dc_pool),
        "fill_rate_optimal_lp": float(fill_rate_lp),
        "fill_rate_naive_equal": float(fill_rate_naive),
        "fill_rate_improvement_pct": float((fill_rate_lp - fill_rate_naive) * 100),
        "allocation_table": out_table
    }

def backtest_inventory_policy(predictions_path="data/processed/eval_predictions.parquet"):
    """
    Backtests Dynamic ROP vs Static Safety Stock on the holdout period with simulated stock.
    Returns stockout rate and average inventory.
    """
    print(f"[{datetime.now()}] Running Holdout Inventory Policy Simulation...")
    df = pd.read_parquet(predictions_path)
    series_sample = df[['store_id', 'product_id']].drop_duplicates().iloc[:100]
    df_sim = df.merge(series_sample, on=['store_id', 'product_id'], how='inner').sort_values(by=['store_id', 'product_id', 'dt'])

    # Static policy: static safety stock = 1.5 * historical 7-day mean
    # Dynamic policy: dynamic ROP with latent demand + quantile safety stock
    dates = sorted(df_sim['dt'].unique())
    
    records_dyn = []
    records_stat = []
    
    for (s_id, p_id), grp in df_sim.groupby(['store_id', 'product_id']):
        stock_dyn = float(grp['pred_latent'].iloc[0] * 2.0)
        stock_stat = float(grp['sales_roll7'].iloc[0] * 2.0)
        
        for _, row in grp.iterrows():
            actual = float(row['sale_amount'])
            
            # Dynamic policy step
            so_dyn = 1 if stock_dyn < actual else 0
            served_dyn = min(stock_dyn, actual)
            stock_dyn = max(0.0, stock_dyn - served_dyn)
            # Reorder arrival (2-day simulated lead time)
            if stock_dyn <= row['pred_q95']:
                stock_dyn += float(row['pred_latent'] * 2.5) # replenishment
            records_dyn.append({"stockout": so_dyn, "eod_stock": stock_dyn})
            
            # Static policy step
            so_stat = 1 if stock_stat < actual else 0
            served_stat = min(stock_stat, actual)
            stock_stat = max(0.0, stock_stat - served_stat)
            if stock_stat <= row['sales_roll7']:
                stock_stat += float(row['sales_roll7'] * 2.5)
            records_stat.append({"stockout": so_stat, "eod_stock": stock_stat})

    df_dyn = pd.DataFrame(records_dyn)
    df_stat = pd.DataFrame(records_stat)
    
    res = {
        "dynamic_rop": {
            "stockout_rate": float(df_dyn['stockout'].mean()),
            "avg_inventory_holding": float(df_dyn['eod_stock'].mean()),
            "service_level": float(1.0 - df_dyn['stockout'].mean())
        },
        "static_rule": {
            "stockout_rate": float(df_stat['stockout'].mean()),
            "avg_inventory_holding": float(df_stat['eod_stock'].mean()),
            "service_level": float(1.0 - df_stat['stockout'].mean())
        }
    }
    print(f"Policy Backtest: Dynamic Service Level={res['dynamic_rop']['service_level']*100:.2f}%, Static Service Level={res['static_rule']['service_level']*100:.2f}%")
    return res

if __name__ == "__main__":
    generate_reorder_recommendations()
    alloc_res = smart_multi_store_allocation()
    print("Sample Smart Allocation:", alloc_res['fill_rate_optimal_lp'], "vs Naive:", alloc_res['fill_rate_naive_equal'])
    sim_res = backtest_inventory_policy()
