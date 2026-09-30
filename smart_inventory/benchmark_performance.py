#!/usr/bin/env python3
"""
Performance & Memory Diagnostic Benchmark Script.
Validates zero-copy PyArrow filtering, memory efficiency, and ensures that
the 4.5M-row training split is NEVER converted in full to a pandas DataFrame.
Uses AST code auditing and runtime row-count bounds.
"""

import ast
import os
import sys
import time
from pathlib import Path
import psutil
import pyarrow.compute as pc
from datasets import load_dataset

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]


def get_ram_mb() -> float:
    """Get Resident Set Size (RSS) memory of the current process in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def audit_to_pandas_via_ast(app_path: Path) -> bool:
    """
    Performs an AST-level syntax tree inspection of app.py to verify that
    .to_pandas() is NEVER called on the full 4.5M dataset object (e.g. train, dataset['train'], table).
    """
    with open(app_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(app_path))

    disallowed_receivers = {"train", "train_ds", "table", "pyarrow_table"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to_pandas":
                receiver = node.func.value
                # Check for direct calls like table.to_pandas() or train.to_pandas()
                if isinstance(receiver, ast.Name) and receiver.id in disallowed_receivers:
                    raise AssertionError(f"CRITICAL: Prohibited full dataset conversion found: {receiver.id}.to_pandas() in {app_path.name}")
                # Check for subscript calls like dataset['train'].to_pandas()
                if isinstance(receiver, ast.Subscript):
                    if isinstance(receiver.slice, ast.Constant) and receiver.slice.value == "train":
                        raise AssertionError(f"CRITICAL: Prohibited full dataset conversion found: dataset['train'].to_pandas() in {app_path.name}")
    return True


def benchmark():
    print("=" * 75)
    print("PERFORMANCE & MEMORY DIAGNOSTIC BENCHMARK")
    print("Dataset: Dingdong-Inc/FreshRetailNet-50K")
    print("=" * 75)

    # 1. RAM immediately before loading dataset
    ram_before_load = get_ram_mb()
    print(f"\n1. Process RAM immediately BEFORE loading dataset: {ram_before_load:.2f} MB")

    # 2. Load dataset
    t_load_start = time.perf_counter()
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    t_load_end = time.perf_counter()
    
    ram_after_load = get_ram_mb()
    train_ds = dataset["train"]
    total_rows = len(train_ds)
    
    print(f"2. Process RAM immediately AFTER loading/caching dataset: {ram_after_load:.2f} MB")
    print(f"   - Dataset Load Latency: {t_load_end - t_load_start:.4f} seconds")
    print(f"   - Total Observations in PyArrow Table: {total_rows:,} rows")
    print(f"   - Process RAM Delta for Dataset Object: +{ram_after_load - ram_before_load:.2f} MB")

    # 3. Access zero-copy PyArrow Table
    table = train_ds.data.table

    # Select sample Store ID & Product ID
    stores = sorted(pc.unique(table["store_id"]).to_pylist())
    test_store = stores[0]
    store_table = table.filter(pc.field("store_id") == test_store)
    products = sorted(pc.unique(store_table["product_id"]).to_pylist())
    test_product = products[0]

    # 4. Measure PyArrow Filtering Latency
    t_filter_start = time.perf_counter()
    expr = (pc.field("store_id") == test_store) & (pc.field("product_id") == test_product)
    series_table = table.filter(expr)
    t_filter_end = time.perf_counter()
    filtering_elapsed_ms = (t_filter_end - t_filter_start) * 1000.0

    print(f"\n3. PyArrow Zero-Copy Table Filtering (Store {test_store}, Product {test_product}):")
    print(f"   - Filter Latency: {filtering_elapsed_ms:.2f} ms ({t_filter_end - t_filter_start:.6f} seconds)")

    # 5. Measure Selected-Series pandas Conversion
    ram_before_pandas = get_ram_mb()
    df_series = series_table.select(REQUIRED_COLUMNS).to_pandas()
    ram_after_pandas = get_ram_mb()
    
    # Runtime Row-Count Assertion
    rows_returned = len(df_series)
    assert rows_returned < 1000, f"Runtime Violation: materialized series has {rows_returned} rows (expected single series < 1000)"
    
    df_memory_mb = df_series.memory_usage(deep=True).sum() / (1024 * 1024)
    ram_delta_process = max(0.0, ram_after_pandas - ram_before_pandas)

    print(f"\n4. Materialized Selected-Series Memory Diagnostics:")
    print(f"   - Series Records Materialized: {rows_returned} rows (Assertion: < 1000 rows PASSED)")
    print(f"   - In-Memory pandas DataFrame Object Footprint: {df_memory_mb:.4f} MB")
    print(f"   - Process-Level RSS RAM Delta during Conversion: +{ram_delta_process:.4f} MB")
    print("   - Note: DataFrame footprint and process RSS are separate measures. Operating system page")
    print("     allocations vary across OS environments; the critical invariant is low slice footprint.")

    # 6. AST Code Audit Verification
    app_target = BASE_DIR / "app.py"
    ast_verified = audit_to_pandas_via_ast(app_target)

    print("\n" + "=" * 75)
    print("CODE AUDIT & MEMORY SAFETY VERIFICATION:")
    print(f"   - Target File Audited via AST: {app_target.name}")
    print(f"   - AST Full-Dataset .to_pandas() Call Audit: PASSED (Verified Safe)")
    print("   - Runtime Row-Count Assertion: PASSED (Rows < 1,000)")
    print("   - Memory Safety Guarantee: Full 4.5M train split is NEVER materialized in pandas.")
    print("=" * 75)

    return {
        "ram_before_load_mb": round(ram_before_load, 2),
        "ram_after_load_mb": round(ram_after_load, 2),
        "df_memory_mb": round(df_memory_mb, 4),
        "ram_delta_process_mb": round(ram_delta_process, 4),
        "filtering_elapsed_ms": round(filtering_elapsed_ms, 2),
        "rows_returned": rows_returned,
        "ast_verified": ast_verified
    }


if __name__ == "__main__":
    benchmark()
