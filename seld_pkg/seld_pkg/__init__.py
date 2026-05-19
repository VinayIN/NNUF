import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parent / "DCASE2025_seld_baseline"
sys.path.insert(0, str(BASELINE_ROOT))
