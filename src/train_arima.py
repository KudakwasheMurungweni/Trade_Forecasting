import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pmdarima import auto_arima

def train_arima(train_series, test_len, seasonal=True, m=12):
    model = auto_arima(
        train_series,
        seasonal=seasonal, m=m,
        stepwise=True, suppress_warnings=True,
        error_action="ignore", max_order=10,
    )
    forecast = model.predict(n_periods=test_len)
    return model, forecast

def run_arima(train_df, test_df, target_col="trade_value_mn_usd"):
    model, preds = train_arima(train_df[target_col].values, len(test_df))
    return model, preds, test_df[target_col].values
