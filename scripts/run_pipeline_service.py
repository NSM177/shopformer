#!/usr/bin/env python3
"""Script chay FastAPI Pipeline Service (port 8001)."""
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shoplift_iot.pipeline_service.app import main

if __name__ == "__main__":
    main()

