import pandas as pd
import numpy as np
from pathlib import Path

# Target file path
data_dir = Path(__file__).resolve().parent / "smart_inventory" / "data"
data_dir.mkdir(parents=True, exist_ok=True)
csv_path = data_dir / "retail_history.csv"

np.random.seed(42)

# Date range: 35 days from 2026-07-01 to 2026-08-04
dates = pd.date_range(start="2026-07-01", periods=35, freq="D")
stores = [1, 2, 3]
products = [101, 102, 103, 104, 105]

rows = []
for store in stores:
    for prod in products:
        # Base demand depending on product and store
        base_demand = 5.0 + (prod % 10) * 1.5 + (store * 2.0)
        for i, dt in enumerate(dates):
            # Day of week variation (weekends slightly higher)
            dow_mult = 1.25 if dt.weekday() >= 5 else 0.95
            
            # Stockout occurrence (some days have stockouts)
            stockout_prob = 0.20
            has_stockout = np.random.rand() < stockout_prob
            stockout_cnt = int(np.random.randint(2, 16)) if has_stockout else 0
            
            # Reduce sales if stockout occurred
            stockout_impact = max(0.2, 1.0 - (stockout_cnt / 16.0) * 0.5)
            
            # Calculate sales amount
            sales = max(0.0, np.round(np.random.normal(base_demand, 1.5) * dow_mult * stockout_impact, 2))
            
            rows.append({
                "store_id": store,
                "product_id": prod,
                "dt": dt.strftime("%Y-%m-%d"),
                "sale_amount": sales,
                "stock_hour6_22_cnt": stockout_cnt
            })

df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)
print(f"Generated sample CSV with {len(df)} rows at: {csv_path}")
