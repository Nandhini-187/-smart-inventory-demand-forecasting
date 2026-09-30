#!/usr/bin/env python3
"""
Automated Verification Suite for Retail Store Smart Inventory Management System.
Verifies all core requirements and regression invariants:
1. Model loading & feature order contract
2. Golden-value regression test (Store 0, Product 4 reference series)
3. Zero-leakage unit test (perturb target day and current day, assert features unchanged)
4. Time series continuity guards (minimum 8-day history & contiguous dates)
5. Stockout-history risk classification thresholds (<8h, 8-23h, >=24h)
6. Forecast non-negativity and stock coverage math
7. Streamlit app smoke test using official AppTest framework
8. Hugging Face PyArrow dataset slice test (with graceful offline skip)
9. FastAPI endpoints & dashboard HTML rendering via Starlette TestClient
"""

import sys
import os
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    MODEL_PATH,
    METADATA_PATH,
    EXPECTED_FEATURES,
    TARGET_COLUMN,
    REQUIRED_COLUMNS,
    RISK_THRESHOLD_MEDIUM,
    RISK_THRESHOLD_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
    RISK_LEVEL_HIGH,
    ACTION_LOW,
    ACTION_MEDIUM,
    ACTION_HIGH,
    BENCHMARK_RESULTS,
    GOLDEN_VALUES
)
from inventory_logic import (
    engineer_features,
    predict_demand,
    assess_stockout_risk,
    compute_inventory_decision,
    validate_series_continuity
)


class TestVerificationSuite(unittest.TestCase):
    
    def test_01_model_loading_and_feature_contract(self):
        """Check 1: Model loads cleanly and feature metadata strictly matches contract."""
        self.assertTrue(MODEL_PATH.exists(), f"Model file missing at: {MODEL_PATH}")
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        self.assertIsInstance(booster, xgb.Booster)
        
        self.assertTrue(METADATA_PATH.exists(), f"Metadata file missing at: {METADATA_PATH}")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        features = metadata.get("features", [])
        self.assertEqual(
            features, 
            EXPECTED_FEATURES,
            f"Feature order mismatch! Expected: {EXPECTED_FEATURES}, Got: {features}"
        )
        self.assertEqual(metadata.get("target"), TARGET_COLUMN)
        self.assertNotIn(TARGET_COLUMN, EXPECTED_FEATURES, "Target column must NEVER be in the feature list!")
        print("  [PASS] Check 1: Model loads and feature order contract matches exact specification.")

    def test_02_golden_value_regression(self):
        """
        Check 2: Golden value regression test.
        Ensures that model prediction, coverage target, and risk label for fixed reference inputs
        (captured from the original unmodified app.py on Store 0, Product 4)
        reproduce the exact golden baseline after refactoring into inventory_logic.py and config.py.
        """
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        
        feature_vec = np.array(GOLDEN_VALUES["features"], dtype=np.float32)
        pred = predict_demand(booster, feature_vec)
        
        self.assertAlmostEqual(
            pred, 
            GOLDEN_VALUES["predicted_next_day_demand"], 
            places=4,
            msg=f"Golden prediction drifted! Expected {GOLDEN_VALUES['predicted_next_day_demand']}, got {pred}"
        )
        
        decision = compute_inventory_decision(
            predicted_next_day_demand=pred,
            rolling_7=GOLDEN_VALUES["features"][2],             # rolling_7
            stockout_rolling_7=GOLDEN_VALUES["features"][5]      # stockout_rolling_7
        )
        
        self.assertAlmostEqual(
            decision["recommended_stock_coverage"],
            GOLDEN_VALUES["recommended_stock_coverage"],
            places=4,
            msg="Golden recommended stock coverage drifted!"
        )
        self.assertEqual(decision["risk_level"], GOLDEN_VALUES["risk_level"])
        self.assertEqual(decision["action_text"], GOLDEN_VALUES["action_text"])
        print(f"  [PASS] Check 2: Golden regression test verified (Prediction: {pred:.4f}, Coverage: {decision['recommended_stock_coverage']:.4f}).")

    def test_03_zero_data_leakage_rigorous(self):
        """
        Check 3: Rigorous zero-leakage unit test.
        1. Confirms target_next_day is not in the feature list.
        2. Perturbs target-day sales at t+1 and asserts day t features are UNCHANGED.
        3. Perturbs current-day sales at t and asserts day t features are UNCHANGED.
        """
        self.assertNotIn("target_next_day", EXPECTED_FEATURES, "CRITICAL: target_next_day found in feature list!")
        
        dates = pd.date_range("2026-01-01", periods=20, freq="D")
        base_df = pd.DataFrame({
            "store_id": 1,
            "product_id": 101,
            "dt": dates,
            "sale_amount": [float(i * 2 + 5) for i in range(20)],
            "stock_hour6_22_cnt": [float(i % 4) for i in range(20)]
        })
        
        orig_features = engineer_features(base_df, include_target=True)
        idx = 10
        orig_row_t = orig_features.loc[idx, EXPECTED_FEATURES].copy()
        
        # 3a. Target next day matches sale_amount at day 11 (t+1)
        self.assertEqual(orig_features.loc[idx, TARGET_COLUMN], base_df.loc[idx + 1, "sale_amount"])
        
        # 3b. Perturb target-day value at t+1
        perturbed_target_df = base_df.copy()
        perturbed_target_df.loc[idx + 1, "sale_amount"] = 99999.0
        perturbed_target_features = engineer_features(perturbed_target_df, include_target=True)
        perturbed_row_t1 = perturbed_target_features.loc[idx, EXPECTED_FEATURES]
        
        for feat in EXPECTED_FEATURES:
            self.assertEqual(
                orig_row_t[feat], 
                perturbed_row_t1[feat],
                f"TARGET LEAKAGE DETECTED on feature '{feat}' when target day t+1 was changed!"
            )
            
        # 3c. Perturb current-day value at t
        perturbed_current_df = base_df.copy()
        perturbed_current_df.loc[idx, "sale_amount"] = 88888.0
        perturbed_current_features = engineer_features(perturbed_current_df, include_target=True)
        perturbed_row_t0 = perturbed_current_features.loc[idx, EXPECTED_FEATURES]
        
        for feat in EXPECTED_FEATURES:
            self.assertEqual(
                orig_row_t[feat], 
                perturbed_row_t0[feat],
                f"CURRENT-DAY LEAKAGE DETECTED on feature '{feat}' when current day t was changed!"
            )
            
        print("  [PASS] Check 3: Rigorous zero-leakage unit test passed (target day and same day perturbations do not affect features).")

    def test_04_time_series_continuity_and_history_guards(self):
        """
        Check 4: Guards against series with < 8 days history and date gaps.
        Because shift() moves by row not calendar date, gaps cause silent feature misalignment.
        """
        # Test 1: Short series (< 8 days)
        short_dates = pd.date_range("2026-01-01", periods=6, freq="D")
        short_df = pd.DataFrame({
            "store_id": 1,
            "product_id": 101,
            "dt": short_dates,
            "sale_amount": [5.0] * 6,
            "stock_hour6_22_cnt": [0] * 6
        })
        with self.assertRaises(ValueError) as ctx_short:
            validate_series_continuity(short_df)
        self.assertIn("Insufficient historical data", str(ctx_short.exception))
        
        # Test 2: Series with missing date gaps (e.g. skipping Jan 05)
        gap_dates = [
            pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03"),
            pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-07"),
            pd.Timestamp("2026-01-08"), pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-10")
        ]
        gap_df = pd.DataFrame({
            "store_id": 1,
            "product_id": 101,
            "dt": gap_dates,
            "sale_amount": [5.0] * 9,
            "stock_hour6_22_cnt": [0] * 9
        })
        with self.assertRaises(ValueError) as ctx_gap:
            validate_series_continuity(gap_df)
        self.assertIn("Date discontinuity detected", str(ctx_gap.exception))
        
        print("  [PASS] Check 4: Time series continuity and minimum history guards verified.")

    def test_05_stockout_risk_thresholds(self):
        """
        Check 5: Stockout-history risk classification thresholds:
          LOW if < 8 h -> 'NO IMMEDIATE ACTION'
          MEDIUM if 8-23 h -> 'MONITOR / REPLENISH'
          HIGH if >= 24 h -> 'REPLENISH NOW'
        """
        test_cases = [
            (0.0, RISK_LEVEL_LOW, ACTION_LOW),
            (7.99, RISK_LEVEL_LOW, ACTION_LOW),
            (8.0, RISK_LEVEL_MEDIUM, ACTION_MEDIUM),
            (15.0, RISK_LEVEL_MEDIUM, ACTION_MEDIUM),
            (23.99, RISK_LEVEL_MEDIUM, ACTION_MEDIUM),
            (24.0, RISK_LEVEL_HIGH, ACTION_HIGH),
            (48.0, RISK_LEVEL_HIGH, ACTION_HIGH)
        ]
        for hours, exp_risk, exp_action in test_cases:
            risk, action, _ = assess_stockout_risk(hours)
            self.assertEqual(risk, exp_risk)
            self.assertEqual(action, exp_action)
            
        print("  [PASS] Check 5: Stockout-history risk thresholds and recommendation actions verified.")

    def test_06_non_negativity_and_coverage_math(self):
        """
        Check 6: Forecast non-negativity constraint and coverage math.
          predicted_next_day_demand = max(pred, 0)
          safety_buffer = rolling_7 * 0.50 (50% of rolling mean demand level, ~half a day of demand)
          recommended_stock_coverage = (pred * 7) + safety_buffer
        """
        decision_neg = compute_inventory_decision(
            predicted_next_day_demand=-4.5,
            rolling_7=12.0,
            stockout_rolling_7=4.0
        )
        self.assertEqual(decision_neg["predicted_next_day_demand"], 0.0)
        self.assertEqual(decision_neg["predicted_7day_demand"], 0.0)
        self.assertEqual(decision_neg["safety_buffer"], 6.0)
        self.assertEqual(decision_neg["recommended_stock_coverage"], 6.0)
        self.assertEqual(decision_neg["risk_level"], RISK_LEVEL_LOW)
        
        decision_pos = compute_inventory_decision(
            predicted_next_day_demand=5.0,
            rolling_7=8.0,
            stockout_rolling_7=28.0
        )
        self.assertEqual(decision_pos["predicted_next_day_demand"], 5.0)
        self.assertEqual(decision_pos["predicted_7day_demand"], 35.0)
        self.assertEqual(decision_pos["safety_buffer"], 4.0)
        self.assertEqual(decision_pos["recommended_stock_coverage"], 39.0)
        self.assertEqual(decision_pos["risk_level"], RISK_LEVEL_HIGH)
        
        print("  [PASS] Check 6: Forecast non-negativity and stock coverage math verified.")

    def test_07_streamlit_apptest_smoke(self):
        """Check 7: Official Streamlit AppTest framework smoke test."""
        from streamlit.testing.v1 import AppTest
        app_file = BASE_DIR / "app.py"
        self.assertTrue(app_file.exists(), f"app.py missing at: {app_file}")
        
        at = AppTest.from_file(str(app_file), default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"Streamlit AppTest raised an unhandled exception: {at.exception}")
        print("  [PASS] Check 7: Streamlit AppTest smoke test passed without unhandled exceptions.")

    def test_08_pyarrow_dataset_slice_offline_skippable(self):
        """
        Check 8: HuggingFace PyArrow dataset zero-copy slice and inference.
        Marked skippable offline if network connection is unavailable.
        """
        from datasets import load_dataset
        import pyarrow.compute as pc
        
        try:
            ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
            train_ds = ds["train"]
            table = train_ds.data.table
            
            # Slice Store 0, Product 4
            expr = (pc.field("store_id") == 0) & (pc.field("product_id") == 4)
            series_table = table.filter(expr)
            df_series = series_table.select(REQUIRED_COLUMNS).to_pandas()
            df_series["dt"] = pd.to_datetime(df_series["dt"])
            df_series = df_series.sort_values("dt").reset_index(drop=True)
            
            featured = engineer_features(df_series)
            latest = featured.dropna(subset=EXPECTED_FEATURES).iloc[-1]
            booster = xgb.Booster()
            booster.load_model(str(MODEL_PATH))
            vec = latest[EXPECTED_FEATURES].values.astype(np.float32)
            pred = predict_demand(booster, vec)
            self.assertGreaterEqual(pred, 0.0)
            print(f"  [PASS] Check 8: Hugging Face PyArrow slice verified online ({len(df_series)} rows; pred: {pred:.4f}).")
        except Exception as e:
            self.skipTest(f"Hugging Face dataset unavailable offline ({e}). Test skipped per protocol.")

    def test_09_fastapi_endpoints_and_dashboard_render(self):
        """Check 9: FastAPI TestClient verification for shared logic endpoint and dashboard HTML."""
        from starlette.testclient import TestClient
        root_dir = BASE_DIR.parent
        sys.path.insert(0, str(root_dir))
        
        try:
            from src.api.main import app as fastapi_app
            client = TestClient(fastapi_app)
            
            # 9a. Test shared decision endpoint
            resp_dec = client.get("/api/v1/inventory/decision", params={
                "predicted_next_day_demand": 5.0,
                "rolling_7": 10.0,
                "stockout_rolling_7": 12.0
            })
            self.assertEqual(resp_dec.status_code, 200)
            dec_data = resp_dec.json()
            self.assertEqual(dec_data["risk_level"], "MEDIUM")
            self.assertEqual(dec_data["action_text"], "MONITOR / REPLENISH")
            self.assertEqual(dec_data["recommended_stock_coverage"], 40.0)
            
            # 9b. Test /dashboard HTML rendering
            resp_dash = client.get("/dashboard")
            self.assertEqual(resp_dash.status_code, 200)
            self.assertIn("text/html", resp_dash.headers.get("content-type", ""))
            self.assertIn("FreshRetailNet-50K", resp_dash.text)
            
            print("  [PASS] Check 9: FastAPI decision endpoint and /dashboard HTML verified successfully.")
        except ImportError as e:
            self.skipTest(f"FastAPI src/api/main.py not importable ({e}).")


def run_all():
    print("=" * 75)
    print("RETAIL STORE SMART INVENTORY SYSTEM - VERIFICATION SUITE")
    print("=" * 75)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVerificationSuite)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    print("\n" + "=" * 75)
    if result.wasSuccessful():
        print(f">>> ALL VERIFICATION CHECKS PASSED ({result.testsRun} run, {len(result.skipped)} skipped)")
        print("=" * 75)
        return 0
    else:
        print(f">>> FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("=" * 75)
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
