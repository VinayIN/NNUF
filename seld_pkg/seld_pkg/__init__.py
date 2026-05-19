"""SELD package."""

import sys
from pathlib import Path

BASELINE_DIR = f"{Path(__file__).resolve().parents[1]}/DCASE2025_seld_baseline"
if f"{BASELINE_DIR}" not in sys.path:
    sys.path.insert(0, f"{BASELINE_DIR}")
