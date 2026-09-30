# Inspection Report & Implementation Plan
**Project:** Retail Store Smart Inventory Management and Demand Forecasting  
**Author:** Senior ML Engineer  
**Date:** 2026-09-29  

---

## 1. Inspection Findings (Step 0)

We cloned the official repository (`https://github.com/S-BUVANESH/SMART_DEMAND_FORECASTING_AND_INVENTORY_MANAGEMENT_SYSTEM.git`) and examined its contents:
- `FreshRetailNet_Final_Reproducible_Colab.ipynb`: Complete reproducible notebook defining the 150-product subset, chronological 80/20 split (cutoff 2024-06-08), baseline, Random Forest, and XGBoost training, feature importances, and evaluation metrics.
- `xgboost_stock_aware.json`: Native XGBoost JSON booster artifact (~3.28 MB). Model loads cleanly into `xgb.Booster()`.
- `model_features.json`: Metadata defining the exact 6 features: `["lag_1", "lag_7", "rolling_7", "stockout_lag_1", "stockout_lag_7", "stockout_rolling_7"]`, target `target_next_day`, baseline MAE `0.5746`, validation MAE `0.4383`, RMSE `0.6829`, improvement `23.72%`.
- `generate_sample_data.py`: Synthetic generator creating 35 days across 3 stores and 5 products in `smart_inventory/data/retail_history.csv`.
- `smart_inventory/`: Directory containing `app.py`, `verify.py`, `benchmark_performance.py`, `requirements.txt`, `README.md`.

---

## 2. Gaps Identified Against Specification

| # | Component | Repo Status | Spec Requirement | Action Required |
|---|-----------|-------------|------------------|-----------------|
| 1 | **Verification Script (`verify.py`)** | Missing unit tests for data leakage, risk thresholds, non-negativity constraint, and app import. Also fails on Windows console with cp1252 encoding due to unicode characters (`✓`). | Must verify: model loads, feature order matches, no leakage unit test, risk thresholds, forecast >= 0, app imports. | Enhance `verify.py` with explicit unit test suites for all 6 requirements and UTF-8 console compatibility. |
| 2 | **Decision-Support Logic & Config** | Constants (`SAFETY_FACTOR = 0.50`, risk thresholds `8`, `24`) are hardcoded in `app.py`. No dedicated config or shared engine. | "put constants in a config file"; shared logic module for features & risk rules. | Create `config.py` and `inventory_logic.py` with exact formulas, thresholds, and risk action mappings. |
| 3 | **Synthetic Data Generator** | `generate_sample_data.py` lacks clear warning banner. | "Never generate synthetic data for the final system (generate_sample_data.py is only for offline demos and must be clearly labelled as such)." | Add prominent header warning stating it is solely for offline demo fallbacks. |
| 4 | **Deliverables & Root Setup** | Scripts (`app.py`, `verify.py`, `benchmark_performance.py`, `requirements.txt`) are located in `smart_inventory/`. | Deliverables specified at root or accessible directly via `python verify.py`, etc. | Ensure root contains all deliverable entry points and syncs with `smart_inventory/`. |
| 5 | **Honesty Rules & Documentation** | `README.md` and `app.py` have basic notes but need full adherence to all honesty rules. | Explicitly report measured results on 150-product dev subset; no 95% accuracy claims; no exact stock ledger/EOQ claims; no causal claims; inventory layer as decision support. | Update `README.md` and Streamlit app sidebar/About panel with complete honesty statements. |
| 6 | **Performance & Memory** | PyArrow zero-copy filtering implemented in `app.py` and tested in `benchmark_performance.py`. | Maintain Arrow-backed filtering, ensure delta is ~0.04 MB, no full `to_pandas()`. | Verify and validate via `benchmark_performance.py`. |

---

## 3. Step-by-Step Implementation Task List

- [ ] **Step 1: Environment & Dependency Setup**
  - Verify Python 3.13 environment with `streamlit`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `plotly`, `datasets`, `pyarrow`, `psutil`.
  - Update `requirements.txt`.
- [ ] **Step 2: Configuration & Shared Decision-Support Engine**
  - Create `config.py` containing `SAFETY_FACTOR = 0.50`, risk thresholds (`LOW < 8`, `MEDIUM 8-23`, `HIGH >= 24`), and action labels (`NO IMMEDIATE ACTION`, `MONITOR / REPLENISH`, `REPLENISH NOW`).
  - Create `inventory_logic.py` encapsulating:
    * Leakage-safe feature engineering (`engineer_features`).
    * Inference with non-negativity constraint (`predict_demand`).
    * Inventory risk assessment and 7-day recommended stock coverage computation.
- [ ] **Step 3: Label `generate_sample_data.py`**
  - Add explicit disclaimer header: strictly for offline fallback demos; real system uses official Hugging Face `Dingdong-Inc/FreshRetailNet-50K`.
- [ ] **Step 4: Update and Refine `verify.py`**
  - Implement full automated test suite:
    1. Model artifact loading (`xgboost_stock_aware.json`).
    2. Feature names and exact order contract (`model_features.json`).
    3. Unit test for zero data leakage (verifying `shift(1)` ensures target and current day are never leaked).
    4. Unit test for risk classification thresholds (<8, 8-23, >=24).
    5. Unit test for forecast non-negativity constraint (`max(pred, 0)`).
    6. Module import test (`app.py` and shared engine).
    7. Integration test on Hugging Face dataset slice.
  - Ensure Windows UTF-8 stdout encoding resilience.
- [ ] **Step 5: Verify and Polish Streamlit Application (`app.py`)**
  - Integrate shared `inventory_logic.py` and `config.py`.
  - Ensure PyArrow zero-copy compute expressions, caching via `st.cache_resource` / `st.cache_data`.
  - Ensure Single Next-Day Forecast point on Plotly chart (never 7 daily points).
  - Add comprehensive About / System Limitations panel meeting all Honesty Rules.
  - Support both official Hugging Face dataset (default) and optional CSV upload override.
- [ ] **Step 6: Update `README.md` and Deliverables**
  - Add project overview, architecture, feature contract, benchmark metrics, honesty disclaimers, and reproduction guide.
  - Ensure all deliverables exist at root and in `smart_inventory/`.
- [ ] **Step 7: Execute All Verification & Acceptance Checks**
  - Run `python verify.py` and confirm all tests pass.
  - Run `python benchmark_performance.py` and record memory/latency metrics.
  - Launch Streamlit application in background, interact via browser subagent, capture screenshot.
  - Summarize results vs reference metrics.
