import pyarrow.parquet as pq
import pandas as pd
import numpy as np

def inspect_splits():
    print("=== INSPECTING RAW DATA SPLITS ===")
    eval_path = "data/raw/eval.parquet"
    train_path = "data/raw/train.parquet"
    
    eval_table = pq.read_table(eval_path)
    print(f"Eval rows: {eval_table.num_rows}, cols: {eval_table.num_columns}")
    eval_df = eval_table.select(["store_id", "product_id", "dt", "sale_amount"]).to_pandas()
    eval_df['dt'] = pd.to_datetime(eval_df['dt'])
    
    print(f"Eval date range: {eval_df['dt'].min()} to {eval_df['dt'].max()} ({eval_df['dt'].nunique()} unique dates)")
    print(f"Eval stores: {eval_df['store_id'].nunique()}, products: {eval_df['product_id'].nunique()}")
    eval_series = set(zip(eval_df['store_id'], eval_df['product_id']))
    print(f"Eval unique (store, product) series: {len(eval_series)}")
    
    train_meta = pq.read_metadata(train_path)
    print(f"Train total rows: {train_meta.num_rows}")
    
    # Read sample of train or full columns for store, product, dt
    train_df = pq.read_table(train_path, columns=["store_id", "product_id", "dt", "sale_amount"]).to_pandas()
    train_df['dt'] = pd.to_datetime(train_df['dt'])
    print(f"Train date range: {train_df['dt'].min()} to {train_df['dt'].max()} ({train_df['dt'].nunique()} unique dates)")
    print(f"Train stores: {train_df['store_id'].nunique()}, products: {train_df['product_id'].nunique()}")
    train_series = set(zip(train_df['store_id'], train_df['product_id']))
    print(f"Train unique (store, product) series: {len(train_series)}")
    
    overlap = eval_series.intersection(train_series)
    print(f"Series overlap between train and eval: {len(overlap)} / {len(eval_series)} ({len(overlap)/len(eval_series)*100:.2f}%)")
    
    # Check column names of train
    schema_cols = eval_table.column_names
    print(f"Columns ({len(schema_cols)}): {schema_cols}")

if __name__ == "__main__":
    inspect_splits()
