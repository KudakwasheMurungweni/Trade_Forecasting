import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean(df):
    df = df.dropna(subset=["trade_value_mn_usd"])
    df = df.ffill()
    return df

def get_feature_cols():
    return [
        "lag_1", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_std_3", "growth_rate_mom",
        "exchange_rate_usd_zwl", "inflation_rate_yoy_pct",
        "gdp_proxy_bn_usd", "commodity_price_index", "fuel_price_usd_litre",
        "num_partners", "top_partner_share", "trade_concentration_hhi",
        "regional_trade_share_sadc",
        "month", "quarter",
        "covid_dummy", "currency_crisis", "drought_indicator",
    ]

def get_target_col():
    return "trade_value_mn_usd"

def scale(df, feature_cols, target_col, scaler_X=None, scaler_y=None, fit=True):
    X = df[feature_cols].values
    y = df[[target_col]].values
    if fit:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
    else:
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)
    return X_scaled, y_scaled.ravel(), scaler_X, scaler_y

def train_test_split_temporal(df, test_months=24):
    split = len(df) - test_months
    return df.iloc[:split].copy(), df.iloc[split:].copy()
