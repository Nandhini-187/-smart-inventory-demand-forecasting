# Retail Store Smart Inventory Management and Demand Forecasting

**Academic Context:** B.Sc. Data Science Mini-Project  
**Author / Engineering Lead:** Senior ML Engineer  
**Dataset:** [Dingdong-Inc/FreshRetailNet-50K (Hugging Face)](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)  
**Primary Model Artifact:** Stock-Aware XGBoost Regressor (`xgboost_stock_aware.json`)  
**Experiment Model Artifact:** Stock-Aware XGBoost v2 (`xgboost_v2_features.json`)  
**Single Source of Truth:** `smart_inventory/`  

---

## 1. Executive Summary & Defensible Headline

This system implements an end-to-end retail demand forecasting pipeline and operational stock coverage decision-support layer. Built on the published **FreshRetailNet-50K** retail dataset (4.5 million training observations, 898 stores, 865 products, and 50,000 distinct time series spanning 2024-03-28 to 2024-06-25), the application combines sales history and past stockout hours to produce single-step demand point forecasts and recommended stock coverage targets.

> **Defensible Benchmark Headline:**  
> *"XGBoost beats a stale baseline by 23.72% and a same-day baseline by 14.92%, but beats a plain 7-day mean by only 1.91% and Random Forest by about 0.3%."*

---

## 2. Feature Contract & Information Structure

The production pipeline enforces a strict 6-feature leakage-safe contract:

```python
# Series sorted by [store_id, product_id, dt]; grouped by [store_id, product_id]:
lag_1              = sale_amount.shift(1)
lag_7              = sale_amount.shift(7)
rolling_7          = sale_amount.transform(lambda x: x.shift(1).rolling(7).mean())
stockout_lag_1     = stock_hour6_22_cnt.shift(1)
stockout_lag_7     = stock_hour6_22_cnt.shift(7)
stockout_rolling_7 = stock_hour6_22_cnt.transform(lambda x: x.shift(1).rolling(7).sum())
target_next_day    = sale_amount.shift(-1)  # Target only, NEVER in feature vector
```

### Critical Information Structure & Horizon Clarifications
1. **Target Date ($t+1$):** For an observation row dated day $t$, the single-step target is day $t+1$ (`target_next_day = sale_amount.shift(-1)`).
2. **Predictor History Horizon ($\le t-1$):** `lag_1` is the sale on day $t-1$. `rolling_7` is the mean sales over $t-7 \dots t-1$.
3. **`lag_7` is NOT a Weekly Recurrence Feature:**  
   Because the row is dated $t$ and the target is $t+1$, `lag_7` (`sale_amount.shift(7)`) represents the sale on day $t-7$.  
   The distance from feature to target is $(t+1) - (t-7) = 8$ calendar days. Because $8 \pmod 7 = 1$, `lag_7` falls on a **different day of the week** than the target.  
   - `lag_7` and `stockout_lag_7` are simply 8-day lagged demand/stockout points, **not** weekly seasonality features.
   - A true same-weekday feature relative to target $t+1$ is $(t+1) - 7 = t-6$, which corresponds to `sale_amount.shift(6)`.
4. **Information Gap (Day $t$ Omission):** The sales and stockout hours on the most recent day $t$ are **not** consumed by this frozen model. The model forecasts day $t+1$ using history through day $t-1$.
5. **Baseline Demand Level:** `rolling_7` represents the **preceding 7-day rolling mean demand level** (capturing baseline demand magnitude, not demand variability).

---

## 3. Comprehensive Benchmark & Multi-Baseline Audit

Metrics were rigorously evaluated on the deterministic 150-product development subset (`product_id < 150`, 214,560 holdout observations from 2024-06-09 to 2024-06-24, cutoff: `2024-06-08`, mean holdout daily sales: **1.1277 units**):

| Model / Baseline | Holdout MAE | Holdout RMSE | WAPE (%) | Normalized MAE (NMAE) | XGBoost Gain vs Baseline (%) | Description & Horizon Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Stale Lag-1 Baseline ($t-1 \rightarrow t+1$)** | **0.5746** | **0.9226** | **50.95%** | **0.5095** | **23.72%** | Historical persistence baseline; uses sales from $t-1$ to forecast $t+1$ (2-day stale) |
| **2. True Previous-Day Baseline ($t \rightarrow t+1$)** | **0.5152** | **0.8282** | **45.69%** | **0.4569** | **14.92%** | Realistic same-day persistence; uses actual sales at day $t$ to forecast next-day $t+1$ |
| **3. 7-Day Rolling Mean Baseline (`rolling_7`)** | **0.4469** | **0.7011** | **39.63%** | **0.3963** | **1.91%** | Direct moving average heuristic over preceding 7 days ($t-7 \dots t-1$) |
| **4. 8-Day Lag Persistence Baseline (`lag_7`)** | **0.5652** | **0.9090** | **50.12%** | **0.5012** | **22.45%** | 8-day lag persistence ($t-7 \rightarrow t+1$); falls on a different weekday than target |
| **4b. Weekday-Aligned 7-Day Lag (`shift(6)`)** | **0.5308** | **0.8470** | **47.07%** | **0.4707** | **17.43%** | Same-weekday persistence ($t-6 \rightarrow t+1$); exactly 7 days before target $t+1$ |
| **5. Random Forest (Stock-Aware, Frozen)** | **0.4396** | **0.6848** | **38.98%** | **0.3898** | **0.28%** | Live-trained benchmark (150 trees, max_depth=10, random_state=42) |
| **6. Deployed XGBoost Regressor (Frozen v1)** | **0.4383** | **0.6829** | **38.87%** | **0.3887** | **0.00%** | Deployed JSON model artifact (250 trees, max_depth=7, lr=0.05) |
| **7. XGBoost v2 Experiment (`xgboost_v2_features.json`)** | **0.4194** | **0.6594** | **37.19%** | **0.3719** | **+4.31%** | Experiment model adding sales at $t$, stockouts at $t$, rolling window through $t$, & `shift(6)` |

*Note: All percentage gains are computed using exact unrounded floating-point metrics (e.g., $(0.439568 - 0.438320) / 0.439568 = 0.28395\% \approx 0.28\%$; rounded inputs $0.0013 / 0.4396 = 0.30\%$).*

### 3.1 Paired Cluster Bootstrap Confidence Intervals (Resampling 13,410 Series)
To account for temporal autocorrelation within store-product time series, we conducted **1,000 cluster bootstrap resamples** across unique `(store_id, product_id)` series:
1. **Random Forest vs XGBoost ($\text{MAE}_{\text{RF}} - \text{MAE}_{\text{XGB}}$):**
   - Observed Difference: **+0.0012 MAE**
   - 95% Paired Cluster Bootstrap CI: **`[+0.0008, +0.0017]`**
   - *Interpretation:* While statistically distinguishable from zero due to the massive holdout size (214,560 rows), an MAE difference of ~0.0012 units represents practical parity between the two ensemble architectures.
2. **7-Day Rolling Mean vs XGBoost ($\text{MAE}_{\text{roll7}} - \text{MAE}_{\text{XGB}}$):**
   - Observed Difference: **+0.0085 MAE**
   - 95% Paired Cluster Bootstrap CI: **`[+0.0074, +0.0096]`**
   - *Interpretation:* XGBoost's actual advantage over a zero-parameter moving average is only ~0.0085 units (1.91% error reduction).
3. **True Previous-Day (Day $t$) vs XGBoost:**
   - Observed Difference: **+0.0768 MAE** | 95% CI: **`[+0.0746, +0.0789]`**
4. **Stale Lag-1 ($t-1$) vs XGBoost:**
   - Observed Difference: **+0.1362 MAE** | 95% CI: **`[+0.1329, +0.1395]`**

### 3.2 Relative Error Context (WAPE)
With average holdout sales of **1.1277 units/day**, an MAE of 0.4383 corresponds to a **WAPE of 38.87%** ($\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y} = \frac{\text{MAE}}{\bar{y}}$). The error magnitude appears small in absolute units only because typical daily transaction volumes per SKU are low.

---

## 4. The v2 Model Experiment (Separate Artifact)

To address the information gap where the frozen model ignores the most recent day ($t$), we executed a dedicated v2 experiment saved under **`smart_inventory/xgboost_v2_features.json`** (preserving the frozen model artifact `xgboost_stock_aware.json` untouched):

### v2 Feature Additions:
1. `sale_at_t`: Actual sales on day $t$ (1 day before target $t+1$)
2. `stockout_at_t`: Stockout hours on day $t$
3. `rolling_7_through_t`: 7-day rolling mean demand through day $t$ ($t-6 \dots t$)
4. `stockout_rolling_7_through_t`: 7-day cumulative stockout hours through day $t$
5. `lag_weekday_aligned`: `sale_amount.shift(6)` (sales on day $t-6$, exactly 7 days before target $t+1$)
6. `stockout_lag_weekday_aligned`: `stock_hour6_22_cnt.shift(6)` (stockout on day $t-6$)

### v2 Measured Results on Identical Holdout:
- **v2 Holdout MAE:** **0.4194** (vs 0.4383 for v1 $\rightarrow$ **+4.31% relative error reduction**)
- **v2 Holdout RMSE:** **0.6594** (vs 0.6829 for v1)
- **v2 Holdout WAPE:** **37.19%** (vs 38.87% for v1)
- **95% Bootstrap CI ($\text{MAE}_{\text{v1}} - \text{MAE}_{\text{v2}}$):** **`[+0.0179, +0.0199]`**
- **v2 Top Feature Importances:**
  1. `rolling_7_through_t`: **70.34%**
  2. `lag_weekday_aligned`: **15.68%** (true weekly recurrence becomes the 2nd most important driver)
  3. `sale_at_t`: **4.71%**

*Conclusion: Including day-$t$ sales and weekday-aligned lag improves measured accuracy by 4.31% on identical holdout data.*

---

## 5. Stockout-History Risk Layer & Quantile Alternatives

### 5.1 Definition & Validation
Risk classification is evaluated strictly from past cumulative stockout hours (`stockout_rolling_7` over $t-7 \dots t-1$), **independent of the demand forecast**.

- **Overall Holdout Stockout Base Rate:** **43.58%** of rows encounter an actual stockout on day $t$ (overall mean stockout hours: **3.29h**).

### 5.2 Fixed Thresholds vs Quantile-Based Alternative

#### Strategy A: Fixed Heuristic Thresholds (`<8h`, `8-23h`, `\ge 24h`)
| Stockout Risk Level | Observations | Share of Test Set (%) | Mean Stockout Hours | Actual Stockout Rate (%) | Lift over Base Rate | Action Text |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LOW** ($< 8$h) | 48,542 | 22.62% | 1.82 hrs | **28.54%** | **0.65x** | `NO IMMEDIATE ACTION` |
| **MEDIUM** ($8 \le \text{h} < 24$) | 80,829 | 37.67% | 2.64 hrs | **39.36%** | **0.90x** | `MONITOR / REPLENISH` |
| **HIGH** ($\ge 24$h) | 85,189 | 39.70% | 4.74 hrs | **56.15%** | **1.29x** | `REPLENISH NOW` |

*Alert Fatigue Note:* HIGH (39.7%) and MEDIUM (37.7%) combined flag **77.37%** of all holdout rows, providing poor operational separation and causing severe alert fatigue.

#### Strategy B: Quantile-Based Configurable Alternative (`<29h`, `29-47h`, `\ge 48h`)
To solve alert fatigue, thresholds were calibrated to empirical percentiles (70th percentile: 29.0h, 90th percentile: 48.0h):
| Stockout Risk Level | Observations | Share of Test Set (%) | Mean Stockout Hours | Actual Stockout Rate (%) | Lift over Base Rate | Operational Significance |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LOW** ($< 29$h, 70%) | 148,395 | 69.16% | 2.50 hrs | **36.63%** | **0.84x** | Routine inventory monitoring |
| **MEDIUM** ($29 \le \text{h} < 48$) | 42,149 | 19.64% | 4.28 hrs | **53.68%** | **1.23x** | Priority replenishment watchlist |
| **HIGH** ($\ge 48$h, Top 10%) | 24,016 | 11.19% | 6.42 hrs | **68.74%** | **1.58x** | Urgent action on top 10% most vulnerable series |

*Operational Lift:* Flags only **11.19%** of series as HIGH urgency, concentrating physical store intervention where actual stockout frequency reaches **68.74%** (a **1.58x lift** over the average base rate).

---

## 6. Forecast-Based Stock Coverage Calculation

The forecast-driven stock coverage target combines the point forecast with a safety buffer:
1. **Non-Negative Point Forecast:**
   $$\hat{y}_{\text{next}} = \max(\hat{y}_{\text{raw}}, 0)$$
2. **7-Day Planning Projection:**
   $$\text{demand}_{\text{7day}} = \hat{y}_{\text{next}} \times 7$$
3. **Safety Buffer (Mean Demand Baseline):**
   $$\text{safety\_buffer} = \text{rolling\_7} \times 0.50$$
4. **Recommended Stock Coverage Target:**
   $$\text{recommended\_stock\_coverage} = \text{demand}_{\text{7day}} + \text{safety\_buffer}$$

---

## 7. System Limitations & Academic Honesty Disclosures

In accordance with strict academic honesty principles, the following boundaries govern this project:
1. **Deterministic Development Subset Scope:** The model was trained and evaluated on `product_id < 150`. This represents the first 150 numeric product IDs in the dataset, **not** a random or category-stratified sample.
2. **Development Subset Metrics Only:** Reported metrics (MAE 0.4383, RMSE 0.6829, WAPE 38.87%) reflect this 150-product sub-cohort (214,560 holdout observations), not all 50,000 distinct time series across FreshRetailNet-50K.
3. **Not Production-Ready:** This software is an academic research prototype and decision-support tool. It is not enterprise-ready inventory replenishment software.
4. **No Exact Economic Order Quantity (EOQ) Claims:** The FreshRetailNet-50K dataset does not contain physical on-hand stock balances, order lead times, unit wholesale purchasing costs, or holding cost rates. Therefore, this system implements an operational decision-support heuristic, making no claims of exact EOQ optimization.
5. **No Causal Inferences:** Past stockout hours and lagged sales correlate with future demand, but feature importances and risk associations are purely predictive, not causal. High past stockouts do not causally induce future demand changes.
6. **Stock Coverage is a Demand-Planning Number, Not an Order Quantity:** Recommended stock coverage ($\text{demand}_{\text{7day}} + \text{safety\_buffer}$) is a gross demand-planning projection. Because on-hand inventory is not available in the dataset, it does not represent a net purchase order quantity (which would require subtracting current on-hand inventory).

---

## 8. Architecture, Performance & Defense-in-Depth

- **Single Source of Truth:** Core logic and configs reside in `smart_inventory/`.
- **Runtime Row-Count Assertions (Primary Defense):** In `app.py` data loading (`fetch_series_pandas`) and `inventory_logic.py`, strict runtime assertions enforce that sliced series contain fewer than 1,000 rows (`assert len(df_sub) < 1000`). Even if variable names are changed, full-dataset materialization is guaranteed to fail immediately at runtime.
- **AST Code Audit (Secondary Defense):** `benchmark_performance.py` scans the abstract syntax tree verifying that `.to_pandas()` is not called on full dataset objects (`train`, `train_ds`, `table`).
- **Memory Footprint:** Sliced series DataFrame consumes **0.0041 MB** in-memory object footprint, with an OS process-level RSS RAM delta of **+0.59 MB**.

---

## 9. Verification Suite: 9 Automated Checks

Run the automated verification suite:

```bash
cd smart_inventory
py verify.py
```

### Raw Terminal Output:
```text
Ran 9 tests in 22.769s

OK
===========================================================================
RETAIL STORE SMART INVENTORY SYSTEM - VERIFICATION SUITE
===========================================================================
  [PASS] Check 1: Model loads and feature order contract matches exact specification.
  [PASS] Check 2: Golden regression test verified (Prediction: 0.8284, Coverage: 6.2343).
  [PASS] Check 3: Rigorous zero-leakage unit test passed (target day and same day perturbations do not affect features).
  [PASS] Check 4: Time series continuity and minimum history guards verified.
  [PASS] Check 5: Stockout-history risk thresholds and recommendation actions verified.
  [PASS] Check 6: Forecast non-negativity and stock coverage math verified.
  [PASS] Check 7: Streamlit AppTest smoke test passed without unhandled exceptions.
  [PASS] Check 8: Hugging Face PyArrow slice verified online (90 rows; pred: 0.8284).
  [PASS] Check 9: FastAPI decision endpoint and /dashboard HTML verified successfully.

===========================================================================
>>> ALL VERIFICATION CHECKS PASSED (9 run, 0 skipped)
===========================================================================
```

### Full Description of All 9 Checks:
1. **`test_01_model_loading_and_feature_contract`**: Verifies `xgboost_stock_aware.json` loads into `xgb.Booster()`, and feature order in `model_features.json` strictly matches `EXPECTED_FEATURES` without target leakage.
2. **`test_02_golden_value_regression`**: Regresses Store 0, Product 4 reference inputs against frozen baseline values (Prediction: 0.8284, Coverage: 6.2343, Action: `NO IMMEDIATE ACTION`).
3. **`test_03_zero_data_leakage_rigorous`**: Perturbs target day sales ($t+1$) and same-day sales ($t$), asserting that day-$t$ feature values remain completely unchanged.
4. **`test_04_time_series_continuity_and_history_guards`**: Asserts that series with fewer than 8 days or containing calendar day gaps raise explicit `ValueError` exceptions.
5. **`test_05_stockout_risk_thresholds`**: Verifies boundary conditions for risk classification (<8h LOW, 8-23h MEDIUM, $\ge$24h HIGH).
6. **`test_06_non_negativity_and_coverage_math`**: Validates `max(pred, 0)` non-negativity clamping and recommended stock coverage calculation ($\text{pred}\times 7 + 0.50\times \text{rolling\_7}$).
7. **`test_07_streamlit_apptest_smoke`**: Executes the Streamlit application using the official `AppTest` framework to ensure zero unhandled UI exceptions.
8. **`test_08_pyarrow_dataset_slice_offline_skippable`**: Tests live PyArrow zero-copy slicing and inference from Hugging Face (marked skippable offline if network is unavailable).
9. **`test_09_fastapi_endpoints_and_dashboard_render`**: Uses Starlette `TestClient` to verify the `/api/v1/inventory/decision` endpoint and HTML dashboard rendering at `/dashboard`.

---

## 10. Execution Commands

```bash
# 1. Run 9-check verification suite
py verify.py

# 2. Run multi-baseline holdout audit & bootstrap analysis
py evaluate_holdout.py

# 3. Run memory benchmark & AST safety audit
py benchmark_performance.py

# 4. Launch interactive Streamlit dashboard
streamlit run app.py
```
