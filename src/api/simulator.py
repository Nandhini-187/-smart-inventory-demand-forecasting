"""
Real-time Transaction Event Simulator
Streams synthetic POS and WMS events into the FastAPI ledger.
"""

import time
import random
import requests
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def run_simulation_stream(num_events=20, interval_sec=0.5):
    print(f"[{datetime.now()}] Starting Transaction Simulator to {API_BASE_URL}...")
    
    stores = ["store_101", "store_102", "store_103", "store_104", "store_105"]
    products = ["prod_501", "prod_502", "prod_503", "prod_504"]
    
    for i in range(num_events):
        s = random.choice(stores)
        p = random.choice(products)
        event_type = random.choices(["pos", "ecommerce", "wms"], weights=[0.5, 0.3, 0.2])[0]
        
        try:
            if event_type == "pos":
                qty = round(random.uniform(1.0, 5.0), 1)
                res = requests.post(f"{API_BASE_URL}/ingest/pos", json={
                    "store_id": s,
                    "product_id": p,
                    "units_sold": qty,
                    "transaction_id": f"tx_{random.randint(10000, 99999)}"
                }, timeout=3)
            elif event_type == "ecommerce":
                qty = round(random.uniform(1.0, 3.0), 1)
                res = requests.post(f"{API_BASE_URL}/ingest/ecommerce", json={
                    "store_id": s,
                    "product_id": p,
                    "units_ordered": qty,
                    "order_id": f"ord_{random.randint(10000, 99999)}",
                    "channel": random.choice(["app", "web"])
                }, timeout=3)
            else: # wms replenishment
                qty = round(random.uniform(20.0, 50.0), 1)
                res = requests.post(f"{API_BASE_URL}/ingest/wms", json={
                    "store_id": s,
                    "product_id": p,
                    "units_received": qty,
                    "shipment_id": f"ship_{random.randint(1000, 9999)}"
                }, timeout=3)
                
            if res.status_code == 200:
                data = res.json()
                print(f"[Event {i+1}/{num_events}] {data['source']} -> Store: {s}, Prod: {p}, Delta: {data['delta_qty']}, Current Stock: {data['current_on_hand']}")
            else:
                print(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            print(f"Connection failed: {e}")
            
        time.sleep(interval_sec)

if __name__ == "__main__":
    run_simulation_stream()
