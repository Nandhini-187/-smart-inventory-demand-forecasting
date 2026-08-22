# Retail Store Smart Inventory Management and Demand Forecasting

A B.Sc. Data Science mini-project dashboard for stock-aware retail demand forecasting and inventory decision support.

## Project Overview

This application leverages a pre-trained **XGBoost Regressor** trained on the **FreshRetailNet-50K** retail dataset to generate next-day demand forecasts and assess stockout risks for store-product combinations.

## Dataset

* **Dataset Name:** FreshRetailNet-50K (Dingdong-Inc)
* **Target:** `target_next_day` (`sale_amount` of next day)
* **Prediction Horizon:** Next day

## Model & Feature Contract

* **Model Type:** Stock-Aware XGBoost Regressor (`xgboost_stock_aware.json`)
* **Required Features (in exact order):**
  1. `lag_1`: Previous day's sales amount
  2. `lag_7`: Sales amount 7 days earlier
  3. `rolling_7`: Preceding 7-day rolling mean sales (leakage-safe via `shift(1)`)
  4. `stockout_lag_1`: Previous day's stockout hours count (`stock_hour6_22_cnt`)
  5. `stockout_lag_7`: Stockout hours count 7 days earlier
  6. `stockout_rolling_7`: Preceding 7-day total stockout hours sum

## Model Validation Results

> **Note:** These metrics represent validated development-set results on a 150-product benchmark subset, not full-dataset performance.

| Model | MAE | RMSE |
| :--- | ---: | ---: |
| Naive Baseline | 0.5746 | 0.9226 |
| Random Forest (Stock-Aware) | 0.4396 | 0.6848 |
| XGBoost (Stock-Aware) | **0.4383** | **0.6829** |

* **XGBoost Improvement vs Baseline:** **23.72%**

## Important System Limitation

> **The system provides forecast-driven inventory decision support.** It does not calculate exact economically optimal reorder quantities because the source data does not provide complete inventory-control parameters such as physical stock balance, supplier lead time, ordering cost, and holding cost.

## Run Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Verification Script

```bash
python verify.py
```

### 3. Launch Streamlit Application

```bash
streamlit run app.py
```

## Directory Structure

```text
smart_inventory/
├── app.py                      # Main Streamlit dashboard application
├── verify.py                   # Automated verification & diagnostic script
├── xgboost_stock_aware.json    # Pre-trained XGBoost Regressor model artifact
├── model_features.json         # Model metadata & feature contract definition
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & limitations
└── data/
    └── retail_history.csv      # Sample retail sales & stockout history dataset
```
