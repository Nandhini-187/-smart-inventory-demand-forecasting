"""
Retail Store Smart Inventory Management and Demand Forecasting
Streamlit Dashboard Application
B.Sc. Data Science Mini-Project
Single Source of Truth: smart_inventory/
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# Ensure local imports work whether run from root or smart_inventory/
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import (
    MODEL_PATH,
    METADATA_PATH,
    SAMPLE_DATA_PATH,
    DATASET_REPO_ID,
    REQUIRED_COLUMNS,
    EXPECTED_FEATURES,
    TARGET_COLUMN,
    SAFETY_FACTOR,
    PLANNING_HORIZON_DAYS,
    MIN_REQUIRED_DAYS,
    BENCHMARK_RESULTS,
    FEATURE_IMPORTANCES_REF
)
from inventory_logic import (
    engineer_features,
    predict_demand,
    compute_inventory_decision,
    validate_input_dataframe,
    validate_series_continuity
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Smart Inventory & Demand Forecasting",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    .badge-dev {
        background-color: #38BDF8;
        color: #0F172A;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.45rem;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.1);
    }
    .risk-medium {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 2px solid #F59E0B;
        color: #92400E;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.45rem;
        box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #22C55E;
        color: #166534;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.45rem;
        box-shadow: 0 4px 6px -1px rgba(34, 197, 94, 0.1);
    }
    
    .honest-box {
        background-color: #F8FAFC;
        border-left: 4px solid #64748B;
        padding: 1.1rem 1.3rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_model(path: Path) -> xgb.Booster:
    """Load native XGBoost Booster."""
    if not path.exists():
        st.error(f"Model artifact not found at: {path}")
        st.stop()
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster

@st.cache_data
def load_metadata(path: Path) -> dict:
    """Load model contract metadata."""
    if not path.exists():
        st.error(f"Metadata file not found at: {path}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=True)
def load_hf_arrow_table():
    """
    Load official FreshRetailNet-50K Hugging Face dataset.
    Retains dataset in PyArrow table format without converting the 4.5M rows to pandas.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset(DATASET_REPO_ID)
        train_ds = ds["train"]
        return train_ds.data.table, "huggingface"
    except Exception as e:
        # Graceful fallback to local sample data if offline
        if SAMPLE_DATA_PATH.exists():
            import pyarrow as pa
            df_fallback = pd.read_csv(SAMPLE_DATA_PATH)
            table = pa.Table.from_pandas(df_fallback)
            return table, "local_fallback"
        else:
            raise RuntimeError(f"Unable to load Hugging Face dataset ({e}) and fallback CSV not found.")


# Initialize Model & Metadata
model = load_model(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)

# Verify Feature Contract
if metadata.get("features") != EXPECTED_FEATURES:
    st.error(f"Feature contract mismatch between app and metadata: {metadata.get('features')}")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR: NAVIGATION, METRICS, & HONESTY RULES
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏪 Retail Store AI")
    st.caption("Smart Inventory & Demand Forecasting")
    st.divider()
    
    st.markdown("#### 🤖 Model Specification")
    st.markdown(f"**Architecture:** {metadata.get('model_type', 'XGBoost Regressor')}")
    st.markdown(f"**Target:** `{metadata.get('target', TARGET_COLUMN)}`")
    st.markdown(f"**Horizon:** Next Day (Single-Step)")
    st.markdown(f"**Feature Count:** {len(EXPECTED_FEATURES)}")
    
    st.divider()
    st.markdown("#### 📊 Validated Multi-Baseline Metrics")
    st.caption("150-product development subset (214,560 holdout observations, cutoff 2024-06-08)")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Stale Lag-1 MAE", f"{BENCHMARK_RESULTS['stale_lag1_mae']:.4f}")
        st.metric("Prev-Day MAE", f"{BENCHMARK_RESULTS['true_prev_day_mae']:.4f}")
        st.metric("XGBoost MAE", f"{BENCHMARK_RESULTS['xgboost_mae']:.4f}")
    with col_sb2:
        st.metric("Rolling 7D MAE", f"{BENCHMARK_RESULTS['rolling_7_mae']:.4f}")
        st.metric("Rand Forest MAE", f"{BENCHMARK_RESULTS['rf_mae']:.4f}")
        st.metric("Holdout WAPE", f"{BENCHMARK_RESULTS['xgboost_wape']:.1f}%")
        
    st.info(
        "**Defensible Headline:** XGBoost beats a stale baseline by **23.72%** and a same-day baseline by **14.92%**, "
        "but beats a plain 7-day mean by only **1.91%** and Random Forest by about **0.3%**."
    )
    
    st.divider()
    st.markdown("#### ⚖️ Scope & Honesty Rules")
    st.markdown(
        "- **Subset Scope:** `product_id < 150` (first 150 numeric IDs, not a random sample).\n"
        "- **Evaluation:** Development subset metrics; not full 50,000 series.\n"
        "- **Readiness:** Academic research decision-support prototype, not production ERP.\n"
        "- **No Exact EOQ:** No unit costs, holding rates, or supplier lead times.\n"
        "- **No Causal Claims:** Past stockouts correlate with, but do not causally drive, demand.\n"
        "- **Coverage ≠ Order Qty:** Gross 7-day planning projection; dataset has no on-hand inventory."
    )


# ---------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="badge-dev">B.Sc. Data Science Mini-Project</div>
    <div class="hero-title">Retail Store Smart Inventory Management</div>
    <div class="hero-subtitle">
        Leakage-safe next-day demand forecasting paired with stockout-history risk assessment and forecast-based stock coverage.
        Built on Dingdong-Inc's FreshRetailNet-50K and XGBoost.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SECTION 1: DATA SOURCE SELECTION
# ---------------------------------------------------------
st.subheader("1. Data Source & Ingestion")

uploaded_file = st.file_uploader(
    "Optional: Upload custom retail history CSV (Leave empty to use official FreshRetailNet-50K)",
    type=["csv"],
    help="CSV must include columns: store_id, product_id, dt, sale_amount, stock_hour6_22_cnt"
)

is_custom_upload = False

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        validate_input_dataframe(user_df)
        user_df["dt"] = pd.to_datetime(user_df["dt"])
        is_custom_upload = True
        st.success(f"✅ Active Data Source: Custom CSV Upload ({len(user_df):,} records, {user_df['store_id'].nunique()} stores, {user_df['product_id'].nunique()} products)")
    except Exception as e:
        st.error(f"❌ Invalid CSV schema: {e}")
        st.stop()
else:
    # Use PyArrow cached HuggingFace dataset
    try:
        pyarrow_table, source_mode = load_hf_arrow_table()
        if source_mode == "huggingface":
            st.info("🌐 **Active Data Source:** Official `Dingdong-Inc/FreshRetailNet-50K` Hugging Face Release (4.5M train records, Zero-Copy Arrow-backed).")
        else:
            st.warning("⚠️ **Active Data Source:** Offline Fallback Demo CSV (Hugging Face connection unavailable).")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

# Summary metadata badges
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Total Train Records", "4,500,000" if not is_custom_upload else f"{len(user_df):,}")
with col_m2:
    st.metric("Eval Set Records", "350,000" if not is_custom_upload else "N/A")
with col_m3:
    st.metric("Stores Count", "898" if not is_custom_upload else f"{user_df['store_id'].nunique()}")
with col_m4:
    st.metric("Products Count", "865" if not is_custom_upload else f"{user_df['product_id'].nunique()}")
with col_m5:
    st.metric("Series Count", "50,000" if not is_custom_upload else f"{len(user_df.groupby(['store_id','product_id']))}")


# ---------------------------------------------------------
# SECTION 2: STORE & PRODUCT SELECTION (PyArrow zero-copy)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("2. Store & Product Series Selector")

col_sel1, col_sel2 = st.columns(2)

if is_custom_upload:
    stores_list = sorted(user_df["store_id"].unique())
    with col_sel1:
        sel_store = st.selectbox("Select Store ID", stores_list, key="sb_store_custom")
    prods_list = sorted(user_df[user_df["store_id"] == sel_store]["product_id"].unique())
    with col_sel2:
        sel_product = st.selectbox("Select Product ID", prods_list, key="sb_prod_custom")
        
    series_df = user_df[(user_df["store_id"] == sel_store) & (user_df["product_id"] == sel_product)].copy()
    series_df = series_df.sort_values("dt").reset_index(drop=True)
else:
    @st.cache_data
    def get_stores_pyarrow():
        return sorted(pc.unique(pyarrow_table["store_id"]).to_pylist())

    @st.cache_data
    def get_products_pyarrow(store_id: int):
        st_sub = pyarrow_table.filter(pc.field("store_id") == store_id)
        return sorted(pc.unique(st_sub["product_id"]).to_pylist())

    stores_list = get_stores_pyarrow()
    with col_sel1:
        sel_store = st.selectbox("Select Store ID", stores_list, index=0, key="sb_store_hf")
        
    prods_list = get_products_pyarrow(sel_store)
    with col_sel2:
        sel_product = st.selectbox("Select Product ID", prods_list, index=0, key="sb_prod_hf")

    # High-Performance PyArrow filtering: Convert ONLY the selected series (~30-90 rows) to pandas!
    @st.cache_data
    def fetch_series_pandas(store_id: int, product_id: int) -> pd.DataFrame:
        expr = (pc.field("store_id") == store_id) & (pc.field("product_id") == product_id)
        sub_table = pyarrow_table.filter(expr)
        df_sub = sub_table.select(REQUIRED_COLUMNS).to_pandas()
        df_sub["dt"] = pd.to_datetime(df_sub["dt"])
        df_sub = df_sub.sort_values("dt").reset_index(drop=True)
        # Runtime row-count assertion inside data loading pipeline
        assert len(df_sub) < 1000, f"Series row count {len(df_sub)} exceeds safety limit (< 1,000 rows expected per store-product series; prevents accidental full-dataset materialization)"
        return df_sub

    series_df = fetch_series_pandas(sel_store, sel_product)

# Time Series Continuity & Minimum History Validation (with row-count safety guard)
assert len(series_df) < 1000, f"Loaded series row count {len(series_df)} exceeds safety threshold (< 1,000 rows)."
try:
    validate_series_continuity(series_df)
except (ValueError, AssertionError) as err:
    st.warning(f"⚠️ Time Series Validation Warning: {err}")
    st.stop()

# Feature engineering on selected series
featured_df = engineer_features(series_df)
valid_feature_df = featured_df.dropna(subset=EXPECTED_FEATURES)

if valid_feature_df.empty:
    st.warning(f"⚠️ Insufficient records for Store {sel_store}, Product {sel_product} after rolling calculation.")
    st.stop()

latest_row = valid_feature_df.iloc[-1]
latest_date = latest_row["dt"]                                   # Day T (most recent date in historical slice)
feature_cutoff_date = latest_date - pd.Timedelta(days=1)          # Day T-1 (latest data consumed by frozen features)
forecast_date = latest_date + pd.Timedelta(days=1)                # Day T+1 (target single-step forecast date)

# Model Inference
feature_values = latest_row[EXPECTED_FEATURES].values.astype(np.float32)
pred_next_day = predict_demand(model, feature_values)


# ---------------------------------------------------------
# SECTION 3: FORECAST KPI CARDS (Explicit Date Horizons)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("3. Demand Forecast & Coverage KPIs")
st.caption(
    f"Historical Data Slice: **{series_df['dt'].min().strftime('%Y-%m-%d')}** to **{latest_date.strftime('%Y-%m-%d')} (Day T)** | "
    f"Forecast Target Date: **{forecast_date.strftime('%Y-%m-%d')} (Day T+1)** | "
    f"Model Input Horizon: **Data through Day T-1 ({feature_cutoff_date.strftime('%Y-%m-%d')})**"
)

# Preliminary decision computation for KPI cards
prelim_decision = compute_inventory_decision(
    predicted_next_day_demand=pred_next_day,
    rolling_7=latest_row["rolling_7"],
    stockout_rolling_7=latest_row["stockout_rolling_7"],
    risk_mode="fixed"
)

kpi_c1, kpi_c2, kpi_c3 = st.columns(3)

with kpi_c1:
    st.metric(
        label=f"Predicted Demand for Day T+1 ({forecast_date.strftime('%Y-%m-%d')})",
        value=f"{prelim_decision['predicted_next_day_demand']:.2f} units",
        help=f"Single-step point forecast for Day T+1 ({forecast_date.strftime('%Y-%m-%d')}) produced by the stock-aware XGBoost model using historical data strictly through Day T-1 ({feature_cutoff_date.strftime('%Y-%m-%d')}). Note: As defined by the frozen model contract, sales from the most recent day T ({latest_date.strftime('%Y-%m-%d')}) are omitted."
    )

with kpi_c2:
    st.metric(
        label="Projected 7-Day Demand Total",
        value=f"{prelim_decision['predicted_7day_demand']:.2f} units",
        delta="7 × Next-Day Prediction",
        delta_color="off",
        help="Direct multiplication projection of next-day demand across a 7-day planning cycle. (Not 7 individual multi-step forecasts)."
    )

with kpi_c3:
    st.metric(
        label="Recommended Stock Coverage Target",
        value=f"{prelim_decision['recommended_stock_coverage']:.2f} units",
        delta=f"+{prelim_decision['safety_buffer']:.2f} buffer (50% mean demand)",
        delta_color="normal",
        help="Coverage Target = Projected 7-Day Demand + Safety Buffer (50% of preceding 7-day rolling mean demand level)."
    )


# ---------------------------------------------------------
# SECTION 4 & 5: INVENTORY RISK & RECOMMENDED ACTION
# ---------------------------------------------------------
st.markdown("---")
col_risk, col_action = st.columns(2)

with col_risk:
    st.subheader("4. Stockout-History Risk Assessment")
    
    risk_strategy = st.radio(
        "Risk Threshold Strategy:",
        options=["Fixed Heuristic (<8h LOW, 8-23h MED, >=24h HIGH)", "Quantile-Based Configurable (<29h LOW, 29-47h MED, >=48h HIGH)"],
        index=0,
        horizontal=True,
        help="Fixed thresholds flag 77% of rows (causing alert fatigue). Quantile thresholds isolate the top 10% as HIGH with 1.58x lift over base."
    )
    active_risk_mode = "quantile" if "Quantile" in risk_strategy else "fixed"
    
    decision = compute_inventory_decision(
        predicted_next_day_demand=pred_next_day,
        rolling_7=latest_row["rolling_7"],
        stockout_rolling_7=latest_row["stockout_rolling_7"],
        risk_mode=active_risk_mode
    )
    
    st.markdown(f'<div class="{decision["css_class"]}">STOCKOUT RISK: {decision["risk_level"]}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    r_sub1, r_sub2, r_sub3 = st.columns(3)
    with r_sub1:
        st.metric("Past 7D Stockouts", f"{decision['recent_stockout_hours']:.1f} hrs")
    with r_sub2:
        st.metric("Holdout Base Rate", "43.58%")
    with r_sub3:
        lift_val = 1.29 if (decision["risk_level"] == "HIGH" and active_risk_mode == "fixed") else (
            1.58 if (decision["risk_level"] == "HIGH" and active_risk_mode == "quantile") else (
                0.90 if (decision["risk_level"] == "MEDIUM" and active_risk_mode == "fixed") else (
                    1.23 if (decision["risk_level"] == "MEDIUM" and active_risk_mode == "quantile") else 0.65
                )
            )
        )
        st.metric("Stockout Lift", f"{lift_val:.2f}x vs Base")
        
    st.caption("ℹ️ *Risk Classification is based solely on past 7-day stockout hours (stockout_rolling_7), NOT on the demand forecast.*")

with col_action:
    st.subheader("5. Recommended Action")
    st.markdown(f'<div class="{decision["css_class"]}">ACTION: {decision["action_text"]}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Series Key:** Store `{sel_store}` — Product `{sel_product}`")
    st.markdown(f"**Coverage Target:** `{decision['recommended_stock_coverage']:.2f}` units over next 7 days")
    st.markdown(f"**Selected Strategy:** `{'Quantile-Based (Top 10% Alert Guard)' if active_risk_mode == 'quantile' else 'Fixed Thresholds (Operational Heuristic)'}`")
    st.caption("ℹ️ *Replenishment action is prioritized by historical stockout pressure and sized by the next-day demand forecast.*")


# ---------------------------------------------------------
# SECTION 6: DEMAND TREND & FORECAST VISUALIZATION (Plotly)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("6. Historical Sales & Next-Day Forecast Visualization")

fig = go.Figure()

# 1. Historical Daily Sales
fig.add_trace(go.Scatter(
    x=series_df["dt"],
    y=series_df["sale_amount"],
    mode="lines+markers",
    name="Historical Daily Sales",
    line=dict(color="#2563EB", width=2.5),
    marker=dict(size=6, color="#1D4ED8"),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> %{y:.2f} units<extra></extra>"
))

# 2. 7-Day Rolling Mean Demand Level
fig.add_trace(go.Scatter(
    x=valid_feature_df["dt"],
    y=valid_feature_df["rolling_7"],
    mode="lines",
    name="7-Day Rolling Mean Demand (Shift 1)",
    line=dict(color="#10B981", width=1.8, dash="dash"),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Rolling 7D Mean:</b> %{y:.2f}<extra></extra>"
))

# 3. Dashed link between last historical day (Day T) and next-day forecast (Day T+1)
fig.add_trace(go.Scatter(
    x=[latest_date, forecast_date],
    y=[latest_row["sale_amount"], decision["predicted_next_day_demand"]],
    mode="lines",
    name="Forecast Step (T -> T+1)",
    line=dict(color="#EF4444", width=2, dash="dot"),
    showlegend=False,
    hoverinfo="skip"
))

# 4. SINGLE Next-Day Forecast Point (Distinct Star Marker)
fig.add_trace(go.Scatter(
    x=[forecast_date],
    y=[decision["predicted_next_day_demand"]],
    mode="markers",
    name=f"Forecast Point (T+1: {forecast_date.strftime('%Y-%m-%d')})",
    marker=dict(symbol="star", size=18, color="#EF4444", line=dict(color="#991B1B", width=2)),
    customdata=[feature_cutoff_date.strftime('%Y-%m-%d')],
    hovertemplate=(
        f"<b>Forecast Target Date (Day T+1):</b> %{{x|%Y-%m-%d}}<br>"
        f"<b>Predicted Demand:</b> %{{y:.2f}} units<br>"
        f"<i>Predictor Horizon: Features through Day T-1 (%{{customdata}})</i><br>"
        f"<i>Note: Most recent Day T ({latest_date.strftime('%Y-%m-%d')}) omitted by model</i><extra></extra>"
    )
))

fig.update_layout(
    title=f"Store {sel_store} | Product {sel_product}: Sales History (through Day T) & Forecast for Day T+1 ({forecast_date.strftime('%Y-%m-%d')})",
    xaxis_title="Date",
    yaxis_title="Sales / Demand (Units)",
    hovermode="x unified",
    template="plotly_white",
    height=440,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# SECTION 7: MODEL INPUT FEATURES TABLE
# ---------------------------------------------------------
st.markdown("---")
st.subheader("7. Model Input Features (Feature Contract)")
st.caption(f"Exact feature vector passed to XGBoost for inference on cutoff date {latest_date.strftime('%Y-%m-%d')} (Day T):")

descriptions = {
    "lag_1": "Sales on day t-1 (2 days before target t+1)",
    "lag_7": "Sales on day t-7 (8 days before target t+1; falls on a different weekday, NOT weekly recurrence)",
    "rolling_7": "Preceding 7-day rolling mean sales (t-7 to t-1; shift 1 applied)",
    "stockout_lag_1": "Stockout hours count on day t-1",
    "stockout_lag_7": "Stockout hours count on day t-7 (8 days before target t+1)",
    "stockout_rolling_7": "Preceding 7-day cumulative stockout hours (t-7 to t-1; shift 1 applied)"
}

feat_summary = []
for f_name in EXPECTED_FEATURES:
    val = float(latest_row[f_name])
    feat_summary.append({
        "Feature Contract Name": f_name,
        "Engineered Value": round(val, 4),
        "Historical Horizon": descriptions.get(f_name, "")
    })

st.dataframe(pd.DataFrame(feat_summary), hide_index=True, use_container_width=True)
st.caption(
    "ℹ️ **Horizon Clarification:** For a row dated day $t$, the target is day $t+1$. "
    "`lag_7` is the sale on day $t-7$, which is 8 days before the target ($8 \\pmod 7 = 1$, shifting weekday by 1). "
    "A same-weekday lag relative to target $t+1$ would be $t-6$ (`shift(6)`). "
    "Furthermore, sales and stockout hours on the most recent day $t$ are omitted by contract design in this frozen model."
)


# ---------------------------------------------------------
# SECTION 8: VALIDATION PERFORMANCE & MULTI-BASELINE AUDIT
# ---------------------------------------------------------
st.markdown("---")
st.subheader("8. Model Benchmark Performance & Multi-Baseline Audit")
st.caption("Validated on the 150-product development subset (214,560 holdout observations, cutoff 2024-06-08).")

col_vp1, col_vp2 = st.columns([1.2, 0.8])

with col_vp1:
    st.markdown("##### 📈 Multi-Baseline Comparison Table")
    perf_df = pd.DataFrame([
        {"Model / Baseline": "1. Stale Lag-1 Baseline (t-1 -> t+1)", "MAE": 0.5746, "RMSE": 0.9226, "WAPE": "50.95%", "Gain vs Stale": "0.00%", "XGB Gain": "23.72%"},
        {"Model / Baseline": "2. True Prev-Day Baseline (sale at t -> t+1)", "MAE": 0.5152, "RMSE": 0.8282, "WAPE": "45.69%", "Gain vs Stale": "10.34%", "XGB Gain": "14.92%"},
        {"Model / Baseline": "3. 7-Day Rolling Mean (rolling_7: t-7..t-1)", "MAE": 0.4469, "RMSE": 0.7011, "WAPE": "39.63%", "Gain vs Stale": "22.23%", "XGB Gain": "1.91%"},
        {"Model / Baseline": "4. 8-Day Lag Persistence (lag_7: t-7 -> t+1)", "MAE": 0.5652, "RMSE": 0.9090, "WAPE": "50.12%", "Gain vs Stale": "1.64%", "XGB Gain": "22.45%"},
        {"Model / Baseline": "4b. Weekday-Aligned 7-Day Lag (shift(6): t-6 -> t+1)", "MAE": 0.5308, "RMSE": 0.8470, "WAPE": "47.07%", "Gain vs Stale": "7.62%", "XGB Gain": "17.43%"},
        {"Model / Baseline": "5. Random Forest (Frozen Contract)", "MAE": 0.4396, "RMSE": 0.6848, "WAPE": "38.98%", "Gain vs Stale": "23.50%", "XGB Gain": "0.28%"},
        {"Model / Baseline": "6. Deployed XGBoost Regressor (Frozen v1)", "MAE": 0.4383, "RMSE": 0.6829, "WAPE": "38.87%", "Gain vs Stale": "23.72%", "XGB Gain": "0.00%"},
        {"Model / Baseline": "7. XGBoost v2 Experiment (with day t & shift(6))", "MAE": 0.4194, "RMSE": 0.6594, "WAPE": "37.19%", "Gain vs Stale": "27.01%", "XGB Gain": "+4.31% (v2 over v1)"}
    ])
    st.table(perf_df)
    st.markdown(
        "**Defensible Headline:** XGBoost beats a stale baseline by **23.72%** and a same-day baseline by **14.92%**, "
        "but beats a plain 7-day mean by only **1.91%** and Random Forest by about **0.3%** "
        "(RF vs XGBoost 95% Paired Cluster Bootstrap CI: `[+0.0008, +0.0017]`, showing practical parity)."
    )

with col_vp2:
    st.markdown("##### 🔬 XGBoost Feature Importance")
    fi_df = pd.DataFrame([
        {"Feature": k, "Importance (%)": v}
        for k, v in FEATURE_IMPORTANCES_REF.items()
    ])
    
    fig_fi = go.Figure(go.Bar(
        x=fi_df["Importance (%)"],
        y=fi_df["Feature"],
        orientation="h",
        marker=dict(color="#3B82F6"),
        text=[f"{v:.1f}%" for v in fi_df["Importance (%)"]],
        textposition="auto"
    ))
    fig_fi.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=240,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Importance (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("`lag_7` represents 8-day lag persistence, not weekly recurrence. `rolling_7` captures baseline demand level (79.14%).")


# ---------------------------------------------------------
# SECTION 9: ACADEMIC HONESTY, TRANSPARENCY & LIMITATIONS
# ---------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ Academic Honesty Disclosures, Scope & Limitations", expanded=True):
    st.markdown("""
<div class="honest-box">
    <strong>Mandatory Project Disclosures & Academic Honesty Rules:</strong>
    <ul>
        <li><strong>1. Development Subset Scope:</strong> The model was trained and evaluated on <code>product_id < 150</code>. This represents the first 150 numeric product IDs in the dataset, <em>not</em> a random or stratified sample of the catalog.</li>
        <li><strong>2. Results from Development Subset Only:</strong> All reported metrics (MAE 0.4383, RMSE 0.6829, WAPE 38.87%) reflect this 150-product sub-cohort (214,560 holdout observations), not all 50,000 distinct time series across FreshRetailNet-50K.</li>
        <li><strong>3. Not Production-Ready:</strong> This software is an academic research prototype and decision-support tool. It is not production-ready enterprise inventory replenishment software.</li>
        <li><strong>4. No Exact EOQ or Inventory Ledger Claims:</strong> The FreshRetailNet-50K dataset does not contain physical on-hand stock balances, order lead times, unit wholesale purchasing costs, or holding cost rates. Therefore, this system implements an operational decision-support heuristic, making no claims of exact Economic Order Quantity (EOQ) optimization.</li>
        <li><strong>5. No Causal Claims:</strong> Past stockout hours and lagged sales correlate with future demand, but feature importances and risk associations are purely predictive, not causal. High past stockouts do not causally induce future demand changes.</li>
        <li><strong>6. Coverage is a Demand-Planning Number, Not an Order Quantity:</strong> The recommended stock coverage (<code>predicted_7day + safety_buffer</code>) is a gross demand-planning projection. Because on-hand inventory is not available in the dataset, it does not represent a net purchase order quantity (which would require subtracting current on-hand stock).</li>
        <li><strong>7. lag_7 Alignment Clarification:</strong> For a row dated day <code>t</code>, target is <code>t+1</code> and <code>lag_7</code> is sale at <code>t-7</code>. This is an 8-day lag relative to the target, so it falls on a different day of the week. Weekday-aligned lag is <code>shift(6)</code> (day <code>t-6</code>), which yields 0.5308 MAE vs 0.5652 for <code>lag_7</code>.</li>
        <li><strong>8. Alert Fatigue & Risk Separation:</strong> Fixed risk thresholds (<8h LOW, 8-23h MED, >=24h HIGH) flag 77.4% of rows as MEDIUM or HIGH. Quantile-based thresholds (<29h LOW, 29-47h MED, >=48h HIGH) are provided as a configurable alternative that restricts HIGH alerts to the top ~11% of series with 1.58x lift over the 43.58% base rate.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
