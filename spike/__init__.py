"""CPU spike prediction — inference package.

Provides feature engineering, model classes, prediction, and a FastAPI service
for XGBoost-based CPU spike forecasting.

Quickstart
----------
    from spike.predict import predict

    import pandas as pd
    window = pd.read_csv("cpu_window.csv")   # cluster_agg format, last 120 min
    result = predict(window, model_dir="models/spike/")
"""
__version__ = "0.1.0"
