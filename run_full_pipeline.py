"""
run_full_pipeline.py
====================================================
End-to-end script: trains ARIMA, LSTM, and Hybrid
for both imports and exports, then saves every output
required for Phase IV review.

Run from project root:
    python run_full_pipeline.py

Outputs written to:
    outputs/metrics/
    outputs/plots/
    outputs/models/
    outputs/shap/
    outputs/tables/
"""

import os, sys, time, warnings, joblib
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Folder setup ────────────────────────────────────────────────────────────
DIRS = [
    "outputs/metrics", "outputs/plots", "outputs/models",
    "outputs/shap",    "outputs/tables",
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# ── Styling ──────────────────────────────────────────────────────────────────
DARK   = "#0f1117"
PANEL  = "#1c1f28"
BORDER = "#2a2d38"
TEAL   = "#1d9e75"
AMBER  = "#ef9f27"
PURPLE = "#7f77dd"
MUTED  = "#9a9890"
TEXT   = "#e8e6e0"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER, "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans", "font.size": 10,
    "lines.linewidth": 1.8, "figure.dpi": 130,
})

MODEL_COLORS = {"ARIMA": "#b4b2a9", "LSTM": PURPLE, "Hybrid": TEAL}
TT_COLORS    = {"import": TEAL, "export": AMBER}

# ── Helpers ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae,3), "RMSE": round(rmse,3),
            "MAPE": round(mape,3), "R2": round(r2,4)}

def log(msg, indent=0):
    prefix = "  " * indent
    print(f"{prefix}{msg}")

def section(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

# ── Data loading & preprocessing ─────────────────────────────────────────────
FEAT_COLS = [
    "lag_1","lag_3","lag_6","lag_12",
    "rolling_mean_3","rolling_std_3","growth_rate_mom",
    "exchange_rate_usd_zwl","inflation_rate_yoy_pct",
    "gdp_proxy_bn_usd","commodity_price_index","fuel_price_usd_litre",
    "num_partners","top_partner_share","trade_concentration_hhi",
    "regional_trade_share_sadc",
    "month","quarter",
    "covid_dummy","currency_crisis","drought_indicator",
]
TARGET = "trade_value_mn_usd"
# Indices of macro+structural features (exogenous branch for hybrid)
EXOG_IDX = list(range(7, len(FEAT_COLS)))

def load_and_prep(trade_type, test_months=24):
    df = pd.read_csv("data/raw/final_dataset.csv", parse_dates=["date"])
    sub = df[df["trade_type"] == trade_type].copy()
    sub = sub.sort_values("date").reset_index(drop=True)
    sub = sub.dropna(subset=[TARGET]).ffill()
    feat_cols = [c for c in FEAT_COLS if c in sub.columns]

    split = len(sub) - test_months
    train, test = sub.iloc[:split].copy(), sub.iloc[split:].copy()

    sx = MinMaxScaler(); sy = MinMaxScaler()
    X_tr = sx.fit_transform(train[feat_cols].values)
    y_tr = sy.fit_transform(train[[TARGET]].values).ravel()
    X_te = sx.transform(test[feat_cols].values)
    y_te_raw = test[TARGET].values

    return dict(
        train=train, test=test,
        X_tr=X_tr, y_tr=y_tr,
        X_te=X_te, y_te_raw=y_te_raw,
        sx=sx, sy=sy, feat_cols=feat_cols,
        trade_type=trade_type,
    )

# ── Sequence builders ─────────────────────────────────────────────────────────
def make_sequences(X, y, window=12):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def make_hybrid_inputs(X, window=12):
    seq, exog = [], []
    for i in range(window, len(X)):
        seq.append(X[i-window:i])
        exog.append(X[i, EXOG_IDX])
    return np.array(seq), np.array(exog)

# ── ARIMA ────────────────────────────────────────────────────────────────────
def run_arima(p):
    from pmdarima import auto_arima
    log("Fitting auto_ARIMA (seasonal, m=12)...", 1)
    t0 = time.time()
    model = auto_arima(
        p["train"][TARGET].values,
        seasonal=True, m=12, stepwise=True,
        suppress_warnings=True, error_action="ignore",
        max_order=8, information_criterion="aic",
    )
    preds = model.predict(n_periods=len(p["test"]))
    elapsed = time.time() - t0
    y_true = p["test"][TARGET].values
    m = metrics(y_true, preds)
    log(f"Order: {model.order} × {model.seasonal_order} | "
        f"MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.2f}% | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed

# ── LSTM ─────────────────────────────────────────────────────────────────────
def run_lstm(p, window=12, units=64, epochs=100, batch=16):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    Xs_tr, ys_tr = make_sequences(p["X_tr"], p["y_tr"], window)
    Xs_te, _     = make_sequences(p["X_te"], np.zeros(len(p["X_te"])), window)

    model = Sequential([
        LSTM(units, input_shape=(window, p["X_tr"].shape[1])),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    log(f"Training LSTM ({units} units, window={window})...", 1)
    t0 = time.time()
    hist = model.fit(
        Xs_tr, ys_tr,
        validation_split=0.1,
        epochs=epochs, batch_size=batch,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
        verbose=0,
    )
    elapsed = time.time() - t0
    preds_s = model.predict(Xs_te, verbose=0).ravel()
    preds   = p["sy"].inverse_transform(preds_s.reshape(-1,1)).ravel()
    y_true  = p["y_te_raw"][window:]
    m = metrics(y_true, preds)
    log(f"Epochs run: {len(hist.history['loss'])} | "
        f"MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.2f}% | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed, hist.history

# ── Hybrid ────────────────────────────────────────────────────────────────────
def run_hybrid(p, window=12, lstm_units=64, dense_units=32, epochs=100, batch=16):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Concatenate)
    from tensorflow.keras.callbacks import EarlyStopping

    seq_tr, exog_tr = make_hybrid_inputs(p["X_tr"], window)
    seq_te, exog_te = make_hybrid_inputs(p["X_te"], window)
    ys_tr = p["y_tr"][window:]

    # Branch 1 — temporal
    seq_in = Input(shape=(window, p["X_tr"].shape[1]), name="seq")
    x = LSTM(lstm_units)(seq_in)
    x = Dropout(0.2)(x)

    # Branch 2 — macro + structural (exogenous)
    exog_in = Input(shape=(exog_tr.shape[1],), name="exog")
    e = Dense(dense_units, activation="relu")(exog_in)
    e = Dropout(0.1)(e)

    # Fusion
    merged = Concatenate()([x, e])
    out = Dense(32, activation="relu")(merged)
    out = Dense(1, name="output")(out)

    model = Model(inputs=[seq_in, exog_in], outputs=out)
    model.compile(optimizer="adam", loss="mse")

    log(f"Training Hybrid (LSTM={lstm_units} + Dense={dense_units}, window={window})...", 1)
    t0 = time.time()
    hist = model.fit(
        [seq_tr, exog_tr], ys_tr,
        validation_split=0.1,
        epochs=epochs, batch_size=batch,
        callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
        verbose=0,
    )
    elapsed = time.time() - t0
    preds_s = model.predict([seq_te, exog_te], verbose=0).ravel()
    preds   = p["sy"].inverse_transform(preds_s.reshape(-1,1)).ravel()
    y_true  = p["y_te_raw"][window:]
    m = metrics(y_true, preds)
    log(f"Epochs run: {len(hist.history['loss'])} | "
        f"MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.2f}% | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed, hist.history

# ── Plot: actual vs forecast ─────────────────────────────────────────────────
def plot_forecast(results_dict, test_dates, trade_type, window=12):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(f"Actual vs Forecast — {trade_type.upper()}S",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.98)
    fig.patch.set_facecolor(DARK)

    for ax, (model_name, (y_true, y_pred, m)) in zip(axes, results_dict.items()):
        offset = window if model_name != "ARIMA" else 0
        dates  = test_dates[offset:]
        n = min(len(dates), len(y_true), len(y_pred))

        ax.plot(dates[:n], y_true[:n], color=TEXT,    lw=1.8, label="Actual")
        ax.plot(dates[:n], y_pred[:n], color=MODEL_COLORS[model_name],
                lw=1.8, ls="--", label=model_name)

        # Shade error band
        ax.fill_between(dates[:n], y_true[:n], y_pred[:n],
                        alpha=0.12, color=MODEL_COLORS[model_name])

        ax.set_title(
            f"{model_name}  ·  MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  "
            f"MAPE={m['MAPE']:.2f}%  R²={m['R2']:.3f}",
            color=MUTED, fontsize=9, pad=5,
        )
        ax.legend(fontsize=8, framealpha=0.15)
        ax.set_ylabel("USD million", color=MUTED, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"outputs/plots/{trade_type}_forecast_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"Saved: {path}", 2)

# ── Plot: residuals ───────────────────────────────────────────────────────────
def plot_residuals(results_dict, test_dates, trade_type, window=12):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Residuals — {trade_type.upper()}S",
                 color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)

    for ax, (model_name, (y_true, y_pred, m)) in zip(axes, results_dict.items()):
        offset = window if model_name != "ARIMA" else 0
        n = min(len(y_true), len(y_pred))
        resid = y_true[:n] - y_pred[:n]

        ax.bar(range(n), resid,
               color=[MODEL_COLORS[model_name] if r >= 0 else "#d85a30" for r in resid],
               alpha=0.75, width=0.8)
        ax.axhline(0, color=TEXT, lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"{model_name}", color=MUTED, fontsize=9)
        ax.set_xlabel("Test period", color=MUTED, fontsize=8)
        ax.set_ylabel("Residual (USD mn)", color=MUTED, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate mean absolute error
        ax.text(0.97, 0.93, f"MAE={m['MAE']:.1f}",
                transform=ax.transAxes, ha="right",
                color=MODEL_COLORS[model_name], fontsize=8)

    plt.tight_layout()
    path = f"outputs/plots/{trade_type}_residuals.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"Saved: {path}", 2)

# ── Plot: training loss ───────────────────────────────────────────────────────
def plot_loss(histories, trade_type):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Training Loss — {trade_type.upper()}S",
                 color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)

    for ax, (model_name, history) in zip(axes, histories.items()):
        ax.plot(history["loss"],     color=MODEL_COLORS[model_name], lw=1.8, label="Train")
        if "val_loss" in history:
            ax.plot(history["val_loss"], color=AMBER, lw=1.5, ls="--", label="Val")
        ax.set_title(f"{model_name}", color=MUTED, fontsize=9)
        ax.set_xlabel("Epoch", color=MUTED, fontsize=8)
        ax.set_ylabel("MSE Loss", color=MUTED, fontsize=8)
        ax.legend(fontsize=8, framealpha=0.15)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"outputs/plots/{trade_type}_training_loss.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"Saved: {path}", 2)

# ── Plot: model comparison bar chart ─────────────────────────────────────────
def plot_comparison_bars(all_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison — Imports & Exports",
                 color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)

    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        labels, import_vals, export_vals = [], [], []
        for model_name in ["ARIMA", "LSTM", "Hybrid"]:
            if model_name in all_metrics["import"] and model_name in all_metrics["export"]:
                labels.append(model_name)
                import_vals.append(all_metrics["import"][model_name][metric])
                export_vals.append(all_metrics["export"][model_name][metric])

        x = np.arange(len(labels))
        w = 0.35
        bars1 = ax.bar(x - w/2, import_vals, w, label="Imports",
                       color=TEAL, alpha=0.85, edgecolor=DARK)
        bars2 = ax.bar(x + w/2, export_vals, w, label="Exports",
                       color=AMBER, alpha=0.85, edgecolor=DARK)

        ax.set_xticks(x); ax.set_xticklabels(labels, color=MUTED)
        ax.set_title(metric, color=TEXT, fontsize=10, fontweight="bold")
        ax.set_ylabel("USD million" if metric != "MAPE" else "%", color=MUTED, fontsize=8)
        ax.legend(fontsize=8, framealpha=0.15)
        ax.grid(True, alpha=0.3, axis="y")

        # Value labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + h*0.02,
                    f"{h:.1f}", ha="center", va="bottom",
                    color=MUTED, fontsize=7)

    plt.tight_layout()
    path = "outputs/plots/model_comparison_bars.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"Saved: {path}", 2)

# ── Plot: SHAP-style feature importance (Hybrid) ──────────────────────────────
def plot_feature_importance_hybrid(model, feat_cols, X_te, sy, trade_type, window=12, n_perturb=50):
    """
    Permutation-based feature importance for the hybrid model.
    Perturbs each feature column and measures RMSE increase.
    Fast alternative to full SHAP KernelExplainer for large windows.
    """
    log("Computing permutation feature importance...", 2)

    seq_te, exog_te = make_hybrid_inputs(X_te, window)
    base_preds = sy.inverse_transform(
        model.predict([seq_te, exog_te], verbose=0).reshape(-1,1)
    ).ravel()

    importances = {}
    for i, feat in enumerate(feat_cols):
        X_perturbed = X_te.copy()
        np.random.shuffle(X_perturbed[:, i])
        seq_p, exog_p = make_hybrid_inputs(X_perturbed, window)
        preds_p = sy.inverse_transform(
            model.predict([seq_p, exog_p], verbose=0).reshape(-1,1)
        ).ravel()
        # Importance = mean absolute change in prediction
        importances[feat] = np.mean(np.abs(preds_p - base_preds))

    imp_df = pd.Series(importances).sort_values(ascending=True)

    # Color by feature category
    cat_colors = {
        "lag": PURPLE, "rolling": "#afa9ec",
        "exchange": AMBER, "inflation": AMBER, "gdp": AMBER,
        "commodity": AMBER, "fuel": AMBER, "interest": AMBER,
        "num_partners": TEAL, "top_partner": TEAL,
        "trade_conc": TEAL, "regional": TEAL,
        "month": "#5f5e5a", "quarter": "#5f5e5a",
        "covid": "#d85a30", "currency_crisis": "#d85a30",
        "drought": "#d85a30", "policy": "#d85a30",
    }
    def get_color(name):
        for key, c in cat_colors.items():
            if key in name: return c
        return MUTED

    colors = [get_color(f) for f in imp_df.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK)
    ax.barh(imp_df.index, imp_df.values, color=colors, edgecolor=DARK, height=0.7)
    ax.set_xlabel("Mean absolute prediction shift (USD mn)", color=MUTED, fontsize=9)
    ax.set_title(
        f"Hybrid Model — Permutation Feature Importance ({trade_type.capitalize()}s)",
        color=TEXT, fontsize=11, fontweight="bold", pad=12,
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PURPLE, label="Temporal (lags/rolling)"),
        Patch(facecolor=AMBER,  label="Macroeconomic"),
        Patch(facecolor=TEAL,   label="Structural (partners)"),
        Patch(facecolor="#d85a30", label="Shock indicators"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, framealpha=0.2,
              loc="lower right")

    plt.tight_layout()
    path = f"outputs/shap/{trade_type}_feature_importance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"Saved: {path}", 2)

    # Save as CSV too
    imp_df.sort_values(ascending=False).to_csv(
        f"outputs/shap/{trade_type}_feature_importance.csv", header=["importance"]
    )
    return imp_df

# ── Plot: shock period analysis ───────────────────────────────────────────────
def plot_shock_analysis(results_dict, test_df, trade_type, window=12):
    """Highlight COVID and currency crisis periods in the best model's forecast."""
    best = "Hybrid" if "Hybrid" in results_dict else list(results_dict.keys())[-1]
    y_true, y_pred, m = results_dict[best]
    offset = window if best != "ARIMA" else 0
    dates  = test_df["date"].values[offset:]
    covid  = test_df["covid_dummy"].values[offset:]
    crisis = test_df["currency_crisis"].values[offset:]
    n = min(len(dates), len(y_true), len(y_pred))

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(DARK)

    ax.plot(range(n), y_true[:n], color=TEXT,   lw=2,   label="Actual")
    ax.plot(range(n), y_pred[:n], color=TEAL,   lw=1.8, ls="--", label=f"{best} forecast")

    # Shade shock periods
    for i in range(n):
        if covid[i] == 1:
            ax.axvspan(i-0.5, i+0.5, alpha=0.18, color="#d85a30", linewidth=0)
        if crisis[i] == 1:
            ax.axvspan(i-0.5, i+0.5, alpha=0.12, color=AMBER, linewidth=0)

    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0],[0], color=TEXT,  lw=2,   label="Actual"),
        plt.Line2D([0],[0], color=TEAL,  lw=1.8, ls="--", label=f"{best} forecast"),
        Patch(facecolor="#d85a30", alpha=0.4, label="COVID period"),
        Patch(facecolor=AMBER,     alpha=0.3, label="Currency crisis"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, framealpha=0.15)
    ax.set_title(f"Shock Period Analysis — {trade_type.capitalize()}s ({best})",
                 color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Test month", color=MUTED)
    ax.set_ylabel("USD million", color=MUTED)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"outputs/plots/{trade_type}_shock_analysis.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"Saved: {path}", 2)

# ── Summary table ─────────────────────────────────────────────────────────────
def save_tables(all_metrics, all_times):
    rows = []
    for tt in ["import", "export"]:
        for model_name, m in all_metrics[tt].items():
            rows.append({
                "Trade Type": tt.capitalize(),
                "Model": model_name,
                "MAE": m["MAE"], "RMSE": m["RMSE"],
                "MAPE (%)": m["MAPE"], "R²": m["R2"],
                "Train Time (s)": round(all_times[tt].get(model_name, 0), 1),
            })
    df = pd.DataFrame(rows)
    df.to_csv("outputs/tables/model_comparison.csv", index=False)

    # Best model per trade type
    summary = []
    for tt in ["import", "export"]:
        best = min(all_metrics[tt], key=lambda k: all_metrics[tt][k]["RMSE"])
        bm = all_metrics[tt][best]
        summary.append({
            "Trade Type": tt.capitalize(), "Best Model": best,
            "MAE": bm["MAE"], "RMSE": bm["RMSE"],
            "MAPE (%)": bm["MAPE"], "R²": bm["R2"],
        })
    pd.DataFrame(summary).to_csv("outputs/tables/best_model_summary.csv", index=False)

    log("Saved: outputs/tables/model_comparison.csv", 2)
    log("Saved: outputs/tables/best_model_summary.csv", 2)
    return df

# ── Result analysis narrative ─────────────────────────────────────────────────
def print_analysis(all_metrics, imp_dfs):
    section("RESULT ANALYSIS — Chapter 4 Talking Points")

    for tt in ["import", "export"]:
        m = all_metrics[tt]
        models = list(m.keys())
        best   = min(models, key=lambda k: m[k]["RMSE"])
        worst  = max(models, key=lambda k: m[k]["RMSE"])

        log(f"\n{tt.upper()}S:", 0)
        log(f"Best model  : {best}   (RMSE={m[best]['RMSE']:.1f})", 1)
        log(f"Worst model : {worst}  (RMSE={m[worst]['RMSE']:.1f})", 1)

        if "ARIMA" in m and "Hybrid" in m:
            imp = ((m["ARIMA"]["RMSE"] - m["Hybrid"]["RMSE"]) / m["ARIMA"]["RMSE"]) * 100
            sign = "improvement" if imp > 0 else "regression"
            log(f"Hybrid vs ARIMA RMSE: {abs(imp):.1f}% {sign}", 1)

        if "LSTM" in m and "Hybrid" in m:
            imp = ((m["LSTM"]["RMSE"] - m["Hybrid"]["RMSE"]) / m["LSTM"]["RMSE"]) * 100
            sign = "improvement" if imp > 0 else "regression"
            log(f"Hybrid vs LSTM  RMSE: {abs(imp):.1f}% {sign}", 1)

        if tt in imp_dfs:
            top3 = imp_dfs[tt].sort_values(ascending=False).head(3)
            log(f"Top 3 importance features: {', '.join(top3.index.tolist())}", 1)

    # Cross-type comparison
    log("\nIMPORTS vs EXPORTS:", 0)
    for model_name in ["ARIMA","LSTM","Hybrid"]:
        if model_name in all_metrics["import"] and model_name in all_metrics["export"]:
            imp_r = all_metrics["import"][model_name]["RMSE"]
            exp_r = all_metrics["export"][model_name]["RMSE"]
            easier = "imports" if imp_r < exp_r else "exports"
            log(f"{model_name}: imports RMSE={imp_r:.1f} | exports RMSE={exp_r:.1f} → {easier} forecast more accurately", 1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    total_start = time.time()
    all_metrics = {"import": {}, "export": {}}
    all_times   = {"import": {}, "export": {}}
    imp_dfs     = {}

    for trade_type in ["import", "export"]:
        section(f"TRADE TYPE: {trade_type.upper()}S")
        p = load_and_prep(trade_type, test_months=24)
        results   = {}   # model_name → (y_true, y_pred, metrics_dict)
        histories = {}   # lstm, hybrid → history dict
        test_dates = p["test"]["date"].values

        # ── Step 1: ARIMA ───────────────────────────────────────────────────
        log("[ 1 / 3 ]  ARIMA", 0)
        try:
            arima_model, arima_preds, arima_true, arima_m, arima_t = run_arima(p)
            results["ARIMA"] = (arima_true, arima_preds, arima_m)
            all_metrics[trade_type]["ARIMA"] = arima_m
            all_times[trade_type]["ARIMA"]   = arima_t
            joblib.dump(arima_model, f"outputs/models/{trade_type}_arima.pkl")
            log(f"Saved: outputs/models/{trade_type}_arima.pkl", 2)
        except Exception as e:
            log(f"ARIMA failed: {e}", 2)

        # ── Step 2: LSTM ────────────────────────────────────────────────────
        log("[ 2 / 3 ]  LSTM", 0)
        try:
            lstm_model, lstm_preds, lstm_true, lstm_m, lstm_t, lstm_h = run_lstm(p)
            results["LSTM"] = (lstm_true, lstm_preds, lstm_m)
            all_metrics[trade_type]["LSTM"] = lstm_m
            all_times[trade_type]["LSTM"]   = lstm_t
            histories["LSTM"] = lstm_h
            lstm_model.save(f"outputs/models/{trade_type}_lstm.keras")
            log(f"Saved: outputs/models/{trade_type}_lstm.keras", 2)
        except Exception as e:
            log(f"LSTM failed: {e}", 2)

        # ── Step 3: Hybrid ──────────────────────────────────────────────────
        log("[ 3 / 3 ]  HYBRID MODEL", 0)
        try:
            hyb_model, hyb_preds, hyb_true, hyb_m, hyb_t, hyb_h = run_hybrid(p)
            results["Hybrid"] = (hyb_true, hyb_preds, hyb_m)
            all_metrics[trade_type]["Hybrid"] = hyb_m
            all_times[trade_type]["Hybrid"]   = hyb_t
            histories["Hybrid"] = hyb_h
            hyb_model.save(f"outputs/models/{trade_type}_hybrid.keras")
            log(f"Saved: outputs/models/{trade_type}_hybrid.keras", 2)
        except Exception as e:
            log(f"Hybrid failed: {e}", 2)

        # ── Save scalers ────────────────────────────────────────────────────
        joblib.dump({"sx": p["sx"], "sy": p["sy"]},
                    f"outputs/models/{trade_type}_scalers.pkl")

        # ── Plots ───────────────────────────────────────────────────────────
        log("Generating plots...", 0)
        if results:
            plot_forecast(results, test_dates, trade_type)
            plot_residuals(results, test_dates, trade_type)
            plot_shock_analysis(results, p["test"], trade_type)
        if histories:
            plot_loss(histories, trade_type)

        # ── Feature importance (Hybrid only) ────────────────────────────────
        if "Hybrid" in results:
            log("Feature importance...", 0)
            imp_df = plot_feature_importance_hybrid(
                hyb_model, p["feat_cols"], p["X_te"], p["sy"], trade_type
            )
            imp_dfs[trade_type] = imp_df

        # ── Per-model metrics CSV ────────────────────────────────────────────
        rows = [{"Model": k, **v[2]} for k, v in results.items()]
        pd.DataFrame(rows).to_csv(
            f"outputs/metrics/{trade_type}_metrics.csv", index=False
        )
        log(f"Saved: outputs/metrics/{trade_type}_metrics.csv", 2)

    # ── Cross-model summary tables ───────────────────────────────────────────
    section("SAVING SUMMARY TABLES")
    cmp_df = save_tables(all_metrics, all_times)
    print(cmp_df.to_string(index=False))

    # ── Comparison bar chart ─────────────────────────────────────────────────
    plot_comparison_bars(all_metrics)

    # ── Analysis narrative ───────────────────────────────────────────────────
    print_analysis(all_metrics, imp_dfs)

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    section(f"PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    log("All outputs saved to outputs/", 0)
    log("", 0)
    log("FILES WRITTEN:", 0)
    for d in DIRS:
        files = os.listdir(d)
        if files:
            log(f"  {d}/  ({len(files)} files)", 0)
            for f in sorted(files):
                log(f"    {f}", 0)

if __name__ == "__main__":
    main()