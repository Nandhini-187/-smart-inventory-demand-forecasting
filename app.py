"""
Thin wrapper entry point for Streamlit application.
Single source of truth is maintained in smart_inventory/app.py.
"""

import sys
import runpy
from pathlib import Path

target_file = Path(__file__).resolve().parent / "smart_inventory" / "app.py"
runpy.run_path(str(target_file), run_name="__main__")
