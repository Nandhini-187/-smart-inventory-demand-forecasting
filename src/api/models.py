"""
Pydantic Schemas for Inventory Ledger & Serving API
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class POSIngestRequest(BaseModel):
    store_id: str
    product_id: str
    units_sold: float = Field(..., gt=0, description="Observed point of sale units sold")
    transaction_id: Optional[str] = None
    timestamp: Optional[datetime] = None

class EcommerceIngestRequest(BaseModel):
    store_id: str
    product_id: str
    units_ordered: float = Field(..., gt=0, description="E-commerce order units")
    order_id: Optional[str] = None
    channel: str = Field(default="app", description="Online channel: app, web, mini-program")

class WMSIngestRequest(BaseModel):
    store_id: str
    product_id: str
    units_received: float = Field(..., description="Replenishment units received (positive) or shrink/damage (negative)")
    shipment_id: Optional[str] = None
    carrier: Optional[str] = None

class AllocationRequest(BaseModel):
    product_id: str
    pool_units: float = Field(default=500.0, gt=0, description="Central warehouse available pool units")
    min_store_units: float = Field(default=0.0, ge=0)

class IngestResponse(BaseModel):
    status: str
    source: str
    store_id: str
    product_id: str
    delta_qty: float
    current_on_hand: float
    timestamp: str
