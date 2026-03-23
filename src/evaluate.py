import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "MAPE": round(mape, 3), "R2": round(r2, 4)}

def compare_models(results: dict):
    """results = {'ARIMA': (y_true, y_pred), 'LSTM': ..., 'Hybrid': ...}"""
    import pandas as pd
    rows = []
    for name, (yt, yp) in results.items():
        row = {"Model": name}
        row.update(metrics(yt, yp))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")
