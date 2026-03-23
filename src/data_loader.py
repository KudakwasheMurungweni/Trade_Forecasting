import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_final_dataset(trade_type="import"):
    df = pd.read_csv(DATA_DIR / "raw" / "final_dataset.csv", parse_dates=["date"])
    return df[df["trade_type"] == trade_type].copy().sort_values("date").reset_index(drop=True)

def load_trade_data():
    return pd.read_csv(DATA_DIR / "raw" / "trade_data.csv")

def load_macro():
    return pd.read_csv(DATA_DIR / "raw" / "macro_data.csv")

def load_partners():
    return pd.read_csv(DATA_DIR / "raw" / "trade_partners.csv")
