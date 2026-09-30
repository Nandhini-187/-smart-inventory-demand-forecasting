#!/usr/bin/env python3
"""
Thin wrapper entry point for holdout evaluation suite.
Single source of truth is maintained in smart_inventory/evaluate_holdout.py.
"""

import os
import sys
import subprocess

script_path = os.path.join(os.path.dirname(__file__), "smart_inventory", "evaluate_holdout.py")
sys.exit(subprocess.call([sys.executable, script_path] + sys.argv[1:]))
