"""
Master Reproducibility Pipeline Runner
FreshRetailNet-50K Demand Forecasting & Inventory Management
Runs data cleaning, feature engineering, model training, evaluation, inventory control, and anomaly detection.
"""

import argparse
from datetime import datetime

from src.clean import clean_dataset
from src.features import fit_transform_features
from src.train import train_and_evaluate_all
from src.evaluate import run_evaluation
from src.inventory import generate_reorder_recommendations, smart_multi_store_allocation, backtest_inventory_policy
from src.anomaly import detect_anomalies, run_synthetic_validation
import pandas as pd

def run_pipeline(sample_series=1000):
    print("="*80)
    print(f"[{datetime.now()}] STARTING FRESHRETAILNET-50K END-TO-END PIPELINE")
    print("="*80)
    
    # 1. Data Cleaning
    print("\n>>> PHASE 2: DATA CLEANING & PREPROCESSING")
    clean_dataset(sample_series=sample_series)
    
    # 2. Feature Engineering
    print("\n>>> PHASE 3: FEATURE ENGINEERING (63+ FEATURES, NO LEAKAGE)")
    train_clean = pd.read_parquet("data/processed/clean_train.parquet")
    eval_clean = pd.read_parquet("data/processed/clean_eval.parquet")
    train_feat, eval_feat, feature_cols = fit_transform_features(train_clean, eval_clean)
    
    drop_cols = ['hours_sale', 'hours_stock_status']
    train_feat = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
    eval_feat = eval_feat.drop(columns=[c for c in drop_cols if c in eval_feat.columns])
    
    train_feat.to_parquet("data/processed/train_features.parquet", index=False)
    eval_feat.to_parquet("data/processed/eval_features.parquet", index=False)
    
    # 3. Model Training
    print("\n>>> PHASE 4: MODEL TRAINING & OPTUNA HYPERPARAMETER TUNING")
    train_and_evaluate_all()
    
    # 4. Evaluation & Reporting
    print("\n>>> PHASE 5: COMPREHENSIVE EVALUATION & SHAP REPORTING")
    run_evaluation()
    
    # 5. Inventory Optimization
    print("\n>>> PHASE 6: INVENTORY OPTIMIZATION, DYNAMIC ROP & LP ALLOCATION")
    generate_reorder_recommendations()
    smart_multi_store_allocation()
    backtest_inventory_policy()
    
    # 6. Anomaly Detection
    print("\n>>> PHASE 7: ANOMALY DETECTION & SYNTHETIC INJECTION BENCHMARK")
    detect_anomalies()
    run_synthetic_validation()
    
    print("\n" + "="*80)
    print(f"[{datetime.now()}] PIPELINE SUCCESSFULLY EXECUTED END-TO-END!")
    print("All artifacts generated in data/processed/, models/, reports/, and outputs/.")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_series", type=int, default=1000, help="Number of series to process (default: 1000)")
    args = parser.parse_args()
    run_pipeline(sample_series=args.sample_series)
