"""
Adapter Interfaces and Mock Connectors for POS, E-commerce, and WMS.
NOTE: These mock connectors simulate ingestion from retail POS terminals,
e-commerce channels, and warehouse management systems. In production, real
connectors would authenticate and connect to Kafka, SAP, or REST feeds.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from src.api.ledger import record_transaction

class BaseInventoryAdapter(ABC):
    @abstractmethod
    def ingest_event(self, payload: dict) -> dict:
        pass

class MockPOSAdapter(BaseInventoryAdapter):
    """Adapter for in-store Point of Sale transaction feeds."""
    def ingest_event(self, payload: dict) -> dict:
        store_id = str(payload['store_id'])
        product_id = str(payload['product_id'])
        qty = float(payload['units_sold'])
        # POS sale decrements store inventory
        new_stock = record_transaction(
            source="POS",
            store_id=store_id,
            product_id=product_id,
            delta_qty=-qty,
            notes=f"POS Sale: tx={payload.get('transaction_id', 'mock_tx')}"
        )
        return {
            "status": "success",
            "source": "POS",
            "store_id": store_id,
            "product_id": product_id,
            "delta_qty": -qty,
            "current_on_hand": new_stock,
            "timestamp": datetime.now().isoformat()
        }

class MockEcommerceAdapter(BaseInventoryAdapter):
    """Adapter for digital order feeds (app, web)."""
    def ingest_event(self, payload: dict) -> dict:
        store_id = str(payload['store_id'])
        product_id = str(payload['product_id'])
        qty = float(payload['units_ordered'])
        # Ecommerce fulfillment decrements store/dark-store inventory
        new_stock = record_transaction(
            source="ECOMMERCE",
            store_id=store_id,
            product_id=product_id,
            delta_qty=-qty,
            notes=f"E-commerce Order: id={payload.get('order_id', 'mock_order')} channel={payload.get('channel', 'app')}"
        )
        return {
            "status": "success",
            "source": "ECOMMERCE",
            "store_id": store_id,
            "product_id": product_id,
            "delta_qty": -qty,
            "current_on_hand": new_stock,
            "timestamp": datetime.now().isoformat()
        }

class MockWMSAdapter(BaseInventoryAdapter):
    """Adapter for Warehouse Management System stock movements."""
    def ingest_event(self, payload: dict) -> dict:
        store_id = str(payload['store_id'])
        product_id = str(payload['product_id'])
        qty = float(payload['units_received'])
        # Inbound shipment increments store inventory (or decrements if shrink adjustment)
        new_stock = record_transaction(
            source="WMS",
            store_id=store_id,
            product_id=product_id,
            delta_qty=qty,
            notes=f"WMS Receipt: shipment={payload.get('shipment_id', 'mock_shipment')}"
        )
        return {
            "status": "success",
            "source": "WMS",
            "store_id": store_id,
            "product_id": product_id,
            "delta_qty": qty,
            "current_on_hand": new_stock,
            "timestamp": datetime.now().isoformat()
        }
