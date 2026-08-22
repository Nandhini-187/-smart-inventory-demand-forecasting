import json
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from pathlib import Path
from datasets import load_dataset

# Set Streamlit page config
st.set_page_config(
    page_title="Retail Store Smart Inventory Management",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for visual excellence
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 1.2rem;
        text-align: center;
    }
    .risk-high {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .risk-medium {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        color: #92400E;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .risk-low {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        color: #166534;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 0.25rem;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_stock_aware.json"
METADATA_PATH = BASE_DIR / "model_features.json"

REQUIRED_COLUMNS = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]
EXPECTED_FEATURES = [
    "lag_1", "lag_7", "rolling_7",
    "stockout_lag_1", "stockout_lag_7", "stockout_rolling_7"
]

# ---------------------------------------------------------
# CACHED LOADERS & VALIDATION
# ---------------------------------------------------------
@st.cache_resource
def load_xgboost_model(model_file_path: Path):
    """Load native XGBoost Booster model with caching."""
    if not model_file_path.exists():
        st.error(f"❌ Model artifact not found at: `{model_file_path}`")
        st.stop()
    try:
        booster = xgb.Booster()
        booster.load_model(str(model_file_path))
        return booster
    except Exception as e:
        st.error(f"❌ Failed to load XGBoost model: {e}")
        st.stop()

@st.cache_data
def load_model_metadata(metadata_file_path: Path):
    """Load model metadata JSON with caching."""
    if not metadata_file_path.exists():
        st.error(f"❌ Metadata file not found at: `{metadata_file_path}`")
        st.stop()
    try:
        with open(metadata_file_path, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.error(f"❌ Failed to load metadata: {e}")
        st.stop()

@st.cache_resource
def load_hf_dataset():
    """Cache official HuggingFace FreshRetailNet-50K dataset."""
    try:
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        return dataset
    except Exception as e:
        st.error(f"❌ Failed to load FreshRetailNet-50K from Hugging Face: {e}")
        st.stop()

def verify_feature_contract(metadata: dict):
    """Verify feature names and order against model contract."""
    features = metadata.get("features", [])
    if features != EXPECTED_FEATURES:
        st.error(f"❌ Feature contract mismatch!\nExpected: `{EXPECTED_FEATURES}`\nGot: `{features}`")
        st.stop()

# Load model, metadata & dataset
metadata = load_model_metadata(METADATA_PATH)
verify_feature_contract(metadata)
booster_model = load_xgboost_model(MODEL_PATH)
hf_dataset = load_hf_dataset()

# Extract PyArrow table for zero-copy memory-efficient filtering
pyarrow_table = hf_dataset["train"].data.table

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("📦 Smart Inventory")
    st.caption("B.Sc. Data Science Mini-Project")
    st.divider()
    
    st.subheader("🤖 Model Specification")
    st.markdown(f"**Model:** {metadata.get('model_type', 'XGBoost Regressor')}")
    st.markdown(f"**Target:** `{metadata.get('target', 'target_next_day')}`")
    st.markdown(f"**Horizon:** {metadata.get('prediction_horizon', 'Next Day').title()}")
    st.divider()
    
    st.subheader("📊 Validation Metrics")
    st.caption("Validated Development Subset Results")
    st.metric("Baseline MAE", f"{metadata.get('baseline_mae', 0.5746):.4f}")
    st.metric("XGBoost MAE", f"{metadata.get('validation_mae', 0.4383):.4f}")
    st.metric("XGBoost RMSE", f"{metadata.get('validation_rmse', 0.6829):.4f}")
    st.success(f"**Improvement:** {metadata.get('improvement_vs_baseline_percent', 23.72):.2f}% vs Baseline")
    
    st.divider()
    st.info("ℹ️ **System Scope:** Forecast-driven inventory decision support. Does not calculate exact physical inventory balance or EOQ.")

# ---------------------------------------------------------
# MAIN HEADER & DISCLAIMER
# ---------------------------------------------------------
st.markdown('<div class="main-header">Retail Store Smart Inventory Management</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stock-Aware Demand Forecasting & Inventory Risk Assessment</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>Decision Support Notice:</strong> This application provides forecast-driven inventory decision support. 
    It does not calculate exact economically optimal reorder quantities because source data does not provide physical stock balance, 
    supplier lead time, ordering cost, or holding cost parameters.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 1 — DATA INPUT & SOURCE INDICATOR
# ---------------------------------------------------------
st.header("1. Data Input")

uploaded_file = st.file_uploader(
    "Optional CSV File Upload Override (Leave empty to use official FreshRetailNet-50K dataset)",
    type=["csv"],
    help="CSV must contain columns: store_id, product_id, dt, sale_amount, stock_hour6_22_cnt"
)

if uploaded_file is not None:
    st.success("✅ **Data Source:** User-uploaded CSV")
    try:
        user_df = pd.read_csv(uploaded_file)
        missing = [c for c in REQUIRED_COLUMNS if c not in user_df.columns]
        if missing:
            st.error(f"❌ Missing required column(s) in uploaded CSV: {', '.join(missing)}")
            st.stop()
        user_df["dt"] = pd.to_datetime(user_df["dt"])
        is_custom = True
    except Exception as e:
        st.error(f"❌ Invalid CSV file: {e}")
        st.stop()
else:
    st.info("🌐 **Data Source:** FreshRetailNet-50K — Dingdong-Inc / Hugging Face")
    is_custom = False

# Dataset Release Statistics Display
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Training Records", "4,500,000")
with col2:
    st.metric("Eval Records", "350,000")
with col3:
    st.metric("Stores Count", "898")
with col4:
    st.metric("Products Count", "865")
with col5:
    st.metric("Release Source", "Dingdong-Inc")

st.divider()

# ---------------------------------------------------------
# SECTION 2 — STORE & PRODUCT SELECTION
# ---------------------------------------------------------
st.header("2. Store & Product Selection")

col_store, col_prod = st.columns(2)

if is_custom:
    stores_available = sorted(user_df["store_id"].unique())
    with col_store:
        selected_store = st.selectbox("Select Store ID", stores_available)
    
    products_available = sorted(user_df[user_df["store_id"] == selected_store]["product_id"].unique())
    with col_prod:
        selected_product = st.selectbox("Select Product ID", products_available)
        
    series_df = user_df[
        (user_df["store_id"] == selected_store) & 
        (user_df["product_id"] == selected_product)
    ].copy().sort_values("dt").reset_index(drop=True)

else:
    # Memory-efficient PyArrow lookup for store and product dropdowns
    @st.cache_data
    def get_available_stores():
        return sorted(pc.unique(pyarrow_table["store_id"]).to_pylist())

    @st.cache_data
    def get_available_products_for_store(store_id_val):
        filtered_st = pyarrow_table.filter(pc.field("store_id") == store_id_val)
        return sorted(pc.unique(filtered_st["product_id"]).to_pylist())

    stores_available = get_available_stores()
    with col_store:
        selected_store = st.selectbox("Select Store ID", stores_available)

    products_available = get_available_products_for_store(selected_store)
    with col_prod:
        selected_product = st.selectbox("Select Product ID", products_available)

    # Materialize ONLY the specific store-product historical series (~30 to 90 rows)
    @st.cache_data
    def fetch_series_df(store_id_val, product_id_val):
        expr = (pc.field("store_id") == store_id_val) & (pc.field("product_id") == product_id_val)
        filtered = pyarrow_table.filter(expr)
        df_sub = filtered.select(REQUIRED_COLUMNS).to_pandas()
        df_sub["dt"] = pd.to_datetime(df_sub["dt"])
        df_sub = df_sub.sort_values("dt").reset_index(drop=True)
        return df_sub

    series_df = fetch_series_df(selected_store, selected_product)

# Feature Engineering (Leakage-Safe) on the selected series
def engineer_features_on_series(df_series: pd.DataFrame):
    df_series = df_series.copy()
    grp = df_series.groupby(["store_id", "product_id"], sort=False)
    
    df_series["lag_1"] = grp["sale_amount"].shift(1)
    df_series["lag_7"] = grp["sale_amount"].shift(7)
    df_series["rolling_7"] = grp["sale_amount"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    
    df_series["stockout_lag_1"] = grp["stock_hour6_22_cnt"].shift(1)
    df_series["stockout_lag_7"] = grp["stock_hour6_22_cnt"].shift(7)
    df_series["stockout_rolling_7"] = grp["stock_hour6_22_cnt"].transform(
        lambda x: x.shift(1).rolling(7).sum()
    )
    return df_series

series_featured = engineer_features_on_series(series_df)
valid_rows = series_featured.dropna(subset=EXPECTED_FEATURES)

if valid_rows.empty:
    st.warning(f"⚠️ Insufficient historical records for Store {selected_store}, Product {selected_product} to compute 7-day lag/rolling features (at least 8 daily records required).")
    st.stop()

latest_row = valid_rows.iloc[-1]
latest_date = latest_row["dt"]

# Extract model feature values
feature_values = latest_row[EXPECTED_FEATURES].values.astype(np.float32)
dmatrix = xgb.DMatrix(feature_values.reshape(1, -1), feature_names=EXPECTED_FEATURES)

# Predict next-day demand
pred_raw = booster_model.predict(dmatrix)
predicted_next_day_demand = float(np.clip(pred_raw[0], 0, None))

# 7-day projection & safety buffer
predicted_7day_demand = predicted_next_day_demand * 7.0
demand_variability = float(latest_row["rolling_7"])
safety_buffer = demand_variability * 0.50
recommended_stock_coverage = predicted_7day_demand + safety_buffer

# Recent stockout hours
recent_stockout_hours = float(latest_row["stockout_rolling_7"])

# Risk classification
if recent_stockout_hours >= 24.0:
    risk_level = "HIGH"
    risk_class = "risk-high"
    action_text = "REPLENISH NOW"
elif recent_stockout_hours >= 8.0:
    risk_level = "MEDIUM"
    risk_class = "risk-medium"
    action_text = "MONITOR / REPLENISH"
else:
    risk_level = "LOW"
    risk_class = "risk-low"
    action_text = "NO IMMEDIATE ACTION"

st.divider()

# ---------------------------------------------------------
# SECTION 3 — FORECAST KPI CARDS
# ---------------------------------------------------------
st.header("3. Demand Forecast")
st.caption(f"Historical date range: **{series_df['dt'].min().strftime('%Y-%m-%d')}** to **{latest_date.strftime('%Y-%m-%d')}** | Forecast date: **{(latest_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}**")

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="Predicted Next-Day Demand",
        value=f"{predicted_next_day_demand:.2f} units",
        help="Model prediction for the target next-day demand."
    )

with kpi2:
    st.metric(
        label="Projected 7-Day Demand Total",
        value=f"{predicted_7day_demand:.2f} units",
        delta="7 × Next-Day Prediction",
        delta_color="off",
        help="Multiplication projection of next-day demand across a 7-day period. (Not 7 individual daily forecasts)."
    )

with kpi3:
    st.metric(
        label="Recommended Stock Coverage",
        value=f"{recommended_stock_coverage:.2f} units",
        delta=f"Includes {safety_buffer:.2f} safety buffer",
        delta_color="normal",
        help="Projected 7-Day Demand + 50% Demand Variability Safety Buffer."
    )

st.divider()

# ---------------------------------------------------------
# SECTION 4 & 5 — INVENTORY RISK & RECOMMENDED ACTION
# ---------------------------------------------------------
col_risk, col_action = st.columns(2)

with col_risk:
    st.header("4. Inventory Stockout Risk")
    st.markdown(f'<div class="{risk_class}">RISK LEVEL: {risk_level}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.metric("Recent Stockout Hours (7D)", f"{recent_stockout_hours:.1f} hrs")
    with r_col2:
        st.metric("7-Day Demand Variability", f"{demand_variability:.2f} units")
        
    st.caption("ℹ️ *Risk thresholds are configurable project assumptions based on recent stockout history (HIGH >=24h, MEDIUM 8–23h, LOW <8h).*")

with col_action:
    st.header("5. Recommended Action")
    st.markdown(f'<div class="{risk_class}">RECOMMENDATION: {action_text}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Store:** `{selected_store}` | **Product:** `{selected_product}`")
    st.markdown(f"**Coverage Target:** `{recommended_stock_coverage:.2f} units`")
    st.caption("ℹ️ *Replenishment priority derived from demand forecast and historical stockout pressure.*")

st.divider()

# ---------------------------------------------------------
# SECTION 6 — DEMAND TREND CHART
# ---------------------------------------------------------
st.header("6. Demand Trend & Forecast Visualization")

next_day_date = latest_date + pd.Timedelta(days=1)

fig = go.Figure()

# Historical daily sales
fig.add_trace(go.Scatter(
    x=series_df["dt"],
    y=series_df["sale_amount"],
    mode="lines+markers",
    name="Historical Daily Sales",
    line=dict(color="#2563EB", width=2.5),
    marker=dict(size=6, color="#1D4ED8"),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Sales:</b> %{y:.2f} units<extra></extra>"
))

# Dashed line connecting last historical point to next-day forecast
fig.add_trace(go.Scatter(
    x=[latest_date, next_day_date],
    y=[latest_row["sale_amount"], predicted_next_day_demand],
    mode="lines",
    name="Forecast Projection Link",
    line=dict(color="#EF4444", width=2, dash="dot"),
    showlegend=False,
    hoverinfo="skip"
))

# Single Next-Day Forecast Point
fig.add_trace(go.Scatter(
    x=[next_day_date],
    y=[predicted_next_day_demand],
    mode="markers",
    name="Next-Day Demand Forecast",
    marker=dict(symbol="star", size=16, color="#EF4444", line=dict(color="#991B1B", width=1.5)),
    hovertemplate="<b>Forecast Date:</b> %{x|%Y-%m-%d}<br><b>Predicted Demand:</b> %{y:.2f} units<extra></extra>"
))

fig.update_layout(
    title=f"Historical Daily Demand & Next-Day Forecast (Store {selected_store}, Product {selected_product})",
    xaxis_title="Date",
    yaxis_title="Demand / Sales Amount (Units)",
    hovermode="x unified",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# SECTION 7 — MODEL INPUT FEATURES
# ---------------------------------------------------------
st.header("7. Model Feature Inputs")
st.caption(f"Input vector passed to XGBoost Regressor for prediction date {latest_date.strftime('%Y-%m-%d')}:")

feature_df = pd.DataFrame([
    {
        "Feature Name": feat,
        "Value": round(float(latest_row[feat]), 4),
        "Description": {
            "lag_1": "Previous day's sale_amount",
            "lag_7": "Sale amount 7 days earlier",
            "rolling_7": "Mean sale_amount over previous 7 days (shift 1)",
            "stockout_lag_1": "Previous day's stockout hours count",
            "stockout_lag_7": "Stockout hours count 7 days earlier",
            "stockout_rolling_7": "Sum of stockout hours over previous 7 days"
        }[feat]
    }
    for feat in EXPECTED_FEATURES
])

st.dataframe(feature_df, hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# SECTION 8 — MODEL PERFORMANCE
# ---------------------------------------------------------
st.header("8. Model Validation Performance")
st.caption("Results evaluated on the validated 150-product development subset (Chronological 80/20 split).")

metrics_data = pd.DataFrame({
    "Model": ["Naive Baseline", "Random Forest (Stock-Aware)", "XGBoost (Stock-Aware)"],
    "MAE (Mean Absolute Error)": [0.5746, 0.4396, 0.4383],
    "RMSE (Root Mean Squared Error)": [0.9226, 0.6848, 0.6829]
})

st.table(metrics_data)

st.success("🎯 **XGBoost Improvement vs Naive Baseline:** **23.72%** (MAE reduced from 0.5746 to 0.4383)")

st.caption("ℹ️ *Note: Validation metrics are from the development subset workflow and describe overall model accuracy across the test period.*")
