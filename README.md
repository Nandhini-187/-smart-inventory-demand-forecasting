# Retail Store Smart Inventory Management and Demand Forecasting

**Academic Context:** B.Sc. Data Science Mini-Project  
**Single Source of Truth:** All application code, tests, and configurations are located in [`smart_inventory/`](file:///c:/Users/Abirami/Desktop/New%20folder%20(2)/smart_inventory).

---

## Core Benchmark Finding
> **Defensible Headline:**  
> *"XGBoost beats a stale baseline by 23.72% and a same-day baseline by 14.92%, but beats a plain 7-day mean by only 1.91% and Random Forest by about 0.3%."*

---

## Quick Start

```bash
cd smart_inventory
pip install -r requirements.txt
py verify.py
streamlit run app.py
```

### Thin Wrappers at Root
For convenience, root entry points delegate directly to `smart_inventory/`:
- `py verify.py` $\rightarrow$ delegates to `smart_inventory/verify.py` (Runs 9 automated checks)
- `py evaluate_holdout.py` $\rightarrow$ delegates to `smart_inventory/evaluate_holdout.py` (Multi-baseline audit, paired bootstrap CIs, v2 experiment)
- `py benchmark_performance.py` $\rightarrow$ delegates to `smart_inventory/benchmark_performance.py` (Memory footprint & AST audit)
- `streamlit run app.py` $\rightarrow$ delegates to `smart_inventory/app.py`

Please see [`smart_inventory/README.md`](file:///c:/Users/Abirami/Desktop/New%20folder%20(2)/smart_inventory/README.md) for full architecture, multi-baseline audit tables, paired cluster bootstrap intervals, risk quantile alternatives, and full academic honesty disclosures.

