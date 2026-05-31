#!/usr/bin/env python3
"""
Run Locally — Fetches real-time data and runs the full pipeline.
Execute this on your machine where yfinance has internet access.

Usage:
    pip install -r requirements.txt
    python run_local.py
"""
import subprocess, sys

# Auto-install deps
for pkg in ["yfinance","duckdb","pyarrow","xgboost","shap","scikit-learn","statsmodels","pandas-datareader"]:
    try: __import__(pkg.replace("-","_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--quiet","--break-system-packages"],
                              stderr=subprocess.DEVNULL)

from main import main
main()
