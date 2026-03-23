"""
run_full_pipeline_v2.py
====================================================
FIXED version — addresses 3 root causes of poor LSTM/Hybrid performance:

  Fix 1 — Log-transform extreme columns (exchange_rate 1→35921, inflation)
           Raw values caused MinMaxScaler to compress all variation near zero.

  Fix 2 — Fill 27 NaN values before scaling
           NaNs in training sequences caused LSTM to learn noise, not patterns.

  Fix 3 — Reduce window from 12 → 6, add stacked LSTM layers,
           use Huber loss + ReduceLROnPlateau for more stable training.

Run from project root:
    python run_full_pipeline_v2.py

All outputs overwrite the previous run in outputs/
"""

import os, sys, time, warnings, joblib
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Folder setup ─────────────────────────────────────────────────────────────
for d in ["outputs/metrics","outputs/plots","outputs/models",
          "outputs/shap","outputs/tables"]:
    os.makedirs(d, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
DARK=  "#0f1117"; PANEL= "#1c1f28"; BORDER="#2a2d38"
TEAL=  "#1d9e75"; AMBER= "#ef9f27"; PURPLE="#7f77dd"
MUTED= "#9a9890"; TEXT=  "#e8e6e0"
MODEL_COLORS = {"ARIMA": "#b4b2a9", "LSTM": PURPLE, "Hybrid": TEAL}

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER, "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans", "font.size": 10,
    "lines.linewidth": 1.8, "figure.dpi": 130,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true-y_pred)/(np.abs(y_true)+1e-8)))*100
    r2   = r2_score(y_true, y_pred)
    return {"MAE":round(mae,3),"RMSE":round(rmse,3),"MAPE":round(mape,3),"R2":round(r2,4)}

def log(msg, indent=0): print("  "*indent + msg)
def section(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}")

# ── Feature config ────────────────────────────────────────────────────────────
# Replace raw exchange_rate and inflation with their log versions
FEAT_COLS = [
    "lag_1","lag_3","lag_6","lag_12",
    "rolling_mean_3","rolling_std_3","growth_rate_mom",
    "exchange_rate_log",          # ← FIX 1: was exchange_rate_usd_zwl
    "inflation_log",              # ← FIX 1: was inflation_rate_yoy_pct
    "gdp_proxy_bn_usd","commodity_price_index","fuel_price_usd_litre",
    "num_partners","top_partner_share","trade_concentration_hhi",
    "regional_trade_share_sadc",
    "month","quarter",
    "covid_dummy","currency_crisis","drought_indicator",
]
EXOG_IDX = list(range(7, len(FEAT_COLS)))   # macro + structural cols
TARGET   = "trade_value_mn_usd"
WINDOW   = 6   # ← FIX 3: was 12

# ── Data loading ──────────────────────────────────────────────────────────────
def load_and_prep(trade_type, test_months=24):
    df = pd.read_csv("data/raw/final_dataset.csv", parse_dates=["date"])
    sub = df[df["trade_type"]==trade_type].copy().sort_values("date").reset_index(drop=True)
    sub = sub.dropna(subset=[TARGET])

    # FIX 1 — log-transform extreme columns
    sub["exchange_rate_log"] = np.log1p(sub["exchange_rate_usd_zwl"])
    sub["inflation_log"]     = np.log1p(sub["inflation_rate_yoy_pct"].clip(lower=0))

    feat_cols = [c for c in FEAT_COLS if c in sub.columns]

    # FIX 2 — fill NaNs before scaling
    sub[feat_cols] = sub[feat_cols].ffill().bfill().fillna(0)

    split = len(sub) - test_months
    train, test = sub.iloc[:split].copy(), sub.iloc[split:].copy()

    sx = MinMaxScaler(); sy = MinMaxScaler()
    X_tr = sx.fit_transform(train[feat_cols].values)
    y_tr = sy.fit_transform(train[[TARGET]].values).ravel()
    X_te = sx.transform(test[feat_cols].values)
    y_te_raw = test[TARGET].values

    log(f"Prepared {trade_type}: train={len(train)-WINDOW} seqs, "
        f"test={len(test)-WINDOW} seqs, features={len(feat_cols)}", 1)

    return dict(train=train, test=test,
                X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te_raw=y_te_raw,
                sx=sx, sy=sy, feat_cols=feat_cols, trade_type=trade_type)

# ── Sequence builders ─────────────────────────────────────────────────────────
def make_sequences(X, y, window=WINDOW):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)

def make_hybrid_inputs(X, window=WINDOW):
    seq, exog = [], []
    for i in range(window, len(X)):
        seq.append(X[i-window:i])
        exog.append(X[i, EXOG_IDX])
    return np.array(seq), np.array(exog)

# ── ARIMA ─────────────────────────────────────────────────────────────────────
def run_arima(p):
    from pmdarima import auto_arima
    log("Fitting auto_ARIMA (seasonal, m=12)...", 1)
    t0 = time.time()
    model = auto_arima(
        p["train"][TARGET].values,
        seasonal=True, m=12, stepwise=True,
        suppress_warnings=True, error_action="ignore", max_order=8,
    )
    preds = model.predict(n_periods=len(p["test"]))
    elapsed = time.time()-t0
    y_true = p["test"][TARGET].values
    m = metrics(y_true, preds)
    log(f"Order {model.order}×{model.seasonal_order} | "
        f"MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.2f}% | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed

# ── LSTM (stacked, Huber loss, LR scheduler) ──────────────────────────────────
def run_lstm(p, window=WINDOW, epochs=200, batch=8):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    Xs_tr, ys_tr = make_sequences(p["X_tr"], p["y_tr"], window)
    Xs_te, _     = make_sequences(p["X_te"], np.zeros(len(p["X_te"])), window)
    n_feat = p["X_tr"].shape[1]

    # FIX 3 — stacked LSTM + Huber loss + LR scheduler
    model = Sequential([
        LSTM(64, input_shape=(window, n_feat), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(0.001), loss="huber")

    log(f"Training stacked LSTM (window={window}, Huber loss)...", 1)
    t0 = time.time()
    hist = model.fit(
        Xs_tr, ys_tr,
        validation_split=0.15,
        epochs=epochs, batch_size=batch,
        callbacks=[
            EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, verbose=0),
        ],
        verbose=0,
    )
    elapsed = time.time()-t0
    preds_s = model.predict(Xs_te, verbose=0).ravel()
    preds   = p["sy"].inverse_transform(preds_s.reshape(-1,1)).ravel()
    y_true  = p["y_te_raw"][window:]
    m = metrics(y_true, preds)
    log(f"Epochs run: {len(hist.history['loss'])} | "
        f"MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.2f}% | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed, hist.history

# ── Hybrid (stacked LSTM + macro/structural dense branch) ────────────────────
def run_hybrid(p, window=WINDOW, epochs=200, batch=8):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    seq_tr, exog_tr = make_hybrid_inputs(p["X_tr"], window)
    seq_te, exog_te = make_hybrid_inputs(p["X_te"], window)
    ys_tr = p["y_tr"][window:]
    n_feat = p["X_tr"].shape[1]

    # Temporal branch — stacked LSTM
    seq_in = Input(shape=(window, n_feat), name="seq")
    x = LSTM(64, return_sequences=True)(seq_in)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)

    # Exogenous branch — macro + structural features
    exog_in = Input(shape=(exog_tr.shape[1],), name="exog")
    e = Dense(32, activation="relu")(exog_in)
    e = BatchNormalization()(e)
    e = Dropout(0.1)(e)
    e = Dense(16, activation="relu")(e)

    # Fusion
    merged = Concatenate()([x, e])
    out = Dense(32, activation="relu")(merged)
    out = Dropout(0.1)(out)
    out = Dense(1, name="output")(out)

    model = Model(inputs=[seq_in, exog_in], outputs=out)
    model.compile(optimizer=Adam(0.001), loss="huber")

    log(f"Training Hybrid (stacked LSTM + exog branch, window={window}, Huber)...", 1)
    t0 = time.time()
    hist = model.fit(
        [seq_tr, exog_tr], ys_tr,
        validation_split=0.15,
        epochs=epochs, batch_size=batch,
        callbacks=[
            EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, verbose=0),
        ],
        verbose=0,
    )
    elapsed = time.time()-t0
    preds_s = model.predict([seq_te, exog_te], verbose=0).ravel()
    preds   = p["sy"].inverse_transform(preds_s.reshape(-1,1)).ravel()
    y_true  = p["y_te_raw"][window:]
    m = metrics(y_true, preds)
    log(f"Epochs run: {len(hist.history['loss'])} | "
        f"MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.2f}% | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed, hist.history

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_forecast(results, test_dates, trade_type):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle(f"Actual vs Forecast — {trade_type.upper()}S (v2 fixed)",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.98)
    fig.patch.set_facecolor(DARK)
    for ax, (name, (yt, yp, m)) in zip(axes, results.items()):
        offset = WINDOW if name != "ARIMA" else 0
        dates  = test_dates[offset:]
        n = min(len(dates), len(yt), len(yp))
        ax.plot(dates[:n], yt[:n], color=TEXT, lw=1.8, label="Actual")
        ax.plot(dates[:n], yp[:n], color=MODEL_COLORS[name], lw=1.8, ls="--", label=name)
        ax.fill_between(dates[:n], yt[:n], yp[:n], alpha=0.12, color=MODEL_COLORS[name])
        ax.set_title(
            f"{name}  ·  MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  "
            f"MAPE={m['MAPE']:.2f}%  R²={m['R2']:.3f}",
            color=MUTED, fontsize=9, pad=5,
        )
        ax.legend(fontsize=8, framealpha=0.15)
        ax.set_ylabel("USD million", color=MUTED, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(rect=[0,0,1,0.97])
    path = f"outputs/plots/{trade_type}_forecast_comparison.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    log(f"Saved: {path}", 2)

def plot_residuals(results, test_dates, trade_type):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Residuals — {trade_type.upper()}S", color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)
    for ax, (name, (yt, yp, m)) in zip(axes, results.items()):
        n = min(len(yt), len(yp))
        resid = yt[:n]-yp[:n]
        ax.bar(range(n), resid,
               color=[MODEL_COLORS[name] if r>=0 else "#d85a30" for r in resid],
               alpha=0.75, width=0.8)
        ax.axhline(0, color=TEXT, lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"{name}", color=MUTED, fontsize=9)
        ax.set_ylabel("Residual (USD mn)", color=MUTED, fontsize=8)
        ax.text(0.97, 0.93, f"MAE={m['MAE']:.1f}",
                transform=ax.transAxes, ha="right", color=MODEL_COLORS[name], fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"outputs/plots/{trade_type}_residuals.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    log(f"Saved: {path}", 2)

def plot_loss(histories, trade_type):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Training Loss — {trade_type.upper()}S", color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h["loss"], color=MODEL_COLORS[name], lw=1.8, label="Train")
        if "val_loss" in h:
            ax.plot(h["val_loss"], color=AMBER, lw=1.5, ls="--", label="Val")
        ax.set_title(name, color=MUTED, fontsize=9)
        ax.set_xlabel("Epoch", color=MUTED, fontsize=8)
        ax.set_ylabel("Huber Loss", color=MUTED, fontsize=8)
        ax.legend(fontsize=8, framealpha=0.15)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"outputs/plots/{trade_type}_training_loss.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    log(f"Saved: {path}", 2)

def plot_shock_analysis(results, test_df, trade_type):
    best = "Hybrid" if "Hybrid" in results else list(results.keys())[-1]
    yt, yp, m = results[best]
    offset = WINDOW if best != "ARIMA" else 0
    dates  = test_df["date"].values[offset:]
    covid  = test_df["covid_dummy"].values[offset:]
    crisis = test_df["currency_crisis"].values[offset:]
    n = min(len(dates), len(yt), len(yp))
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(DARK)
    ax.plot(range(n), yt[:n], color=TEXT, lw=2, label="Actual")
    ax.plot(range(n), yp[:n], color=TEAL, lw=1.8, ls="--", label=f"{best} forecast")
    for i in range(n):
        if covid[i]:   ax.axvspan(i-.5, i+.5, alpha=0.18, color="#d85a30", linewidth=0)
        if crisis[i]:  ax.axvspan(i-.5, i+.5, alpha=0.12, color=AMBER,     linewidth=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        plt.Line2D([0],[0],color=TEXT, lw=2, label="Actual"),
        plt.Line2D([0],[0],color=TEAL, lw=1.8, ls="--", label=f"{best} forecast"),
        Patch(facecolor="#d85a30", alpha=0.4, label="COVID"),
        Patch(facecolor=AMBER,     alpha=0.3, label="Currency crisis"),
    ], fontsize=8, framealpha=0.15)
    ax.set_title(f"Shock Periods — {trade_type.capitalize()}s ({best})", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Test month", color=MUTED)
    ax.set_ylabel("USD million", color=MUTED)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"outputs/plots/{trade_type}_shock_analysis.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    log(f"Saved: {path}", 2)

def plot_comparison_bars(all_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Comparison — Imports & Exports (v2)",
                 color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)
    for ax, metric in zip(axes, ["MAE","RMSE","MAPE"]):
        labels, imp_v, exp_v = [], [], []
        for name in ["ARIMA","LSTM","Hybrid"]:
            if name in all_metrics["import"] and name in all_metrics["export"]:
                labels.append(name)
                imp_v.append(all_metrics["import"][name][metric])
                exp_v.append(all_metrics["export"][name][metric])
        x = np.arange(len(labels)); w = 0.35
        b1 = ax.bar(x-w/2, imp_v, w, label="Imports", color=TEAL, alpha=0.85, edgecolor=DARK)
        b2 = ax.bar(x+w/2, exp_v, w, label="Exports", color=AMBER, alpha=0.85, edgecolor=DARK)
        ax.set_xticks(x); ax.set_xticklabels(labels, color=MUTED)
        ax.set_title(metric, color=TEXT, fontsize=10, fontweight="bold")
        ax.set_ylabel("USD mn" if metric != "MAPE" else "%", color=MUTED, fontsize=8)
        ax.legend(fontsize=8, framealpha=0.15)
        ax.grid(True, alpha=0.3, axis="y")
        for bar in list(b1)+list(b2):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+h*0.02,
                    f"{h:.1f}", ha="center", va="bottom", color=MUTED, fontsize=7)
    plt.tight_layout()
    path = "outputs/plots/model_comparison_bars.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    log(f"Saved: {path}", 2)

def plot_feature_importance(model, feat_cols, X_te, sy, trade_type):
    log("Computing permutation feature importance...", 2)
    seq_te, exog_te = make_hybrid_inputs(X_te, WINDOW)
    base = sy.inverse_transform(
        model.predict([seq_te, exog_te], verbose=0).reshape(-1,1)
    ).ravel()
    importances = {}
    for i, feat in enumerate(feat_cols):
        X_p = X_te.copy()
        np.random.shuffle(X_p[:,i])
        sq, ex = make_hybrid_inputs(X_p, WINDOW)
        p_pred = sy.inverse_transform(
            model.predict([sq, ex], verbose=0).reshape(-1,1)
        ).ravel()
        importances[feat] = np.mean(np.abs(p_pred - base))
    imp_s = pd.Series(importances).sort_values(ascending=True)
    cat_colors = {
        "lag":PURPLE,"rolling":PURPLE,"growth":PURPLE,
        "exchange":AMBER,"inflation":AMBER,"gdp":AMBER,
        "commodity":AMBER,"fuel":AMBER,
        "num_partners":TEAL,"top_partner":TEAL,"trade_conc":TEAL,"regional":TEAL,
        "month":"#5f5e5a","quarter":"#5f5e5a",
        "covid":"#d85a30","currency":"#d85a30","drought":"#d85a30","policy":"#d85a30",
    }
    def gc(name):
        for k,c in cat_colors.items():
            if k in name: return c
        return MUTED
    colors = [gc(f) for f in imp_s.index]
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK)
    ax.barh(imp_s.index, imp_s.values, color=colors, edgecolor=DARK, height=0.7)
    ax.set_xlabel("Mean absolute prediction shift (USD mn)", color=MUTED, fontsize=9)
    ax.set_title(f"Hybrid — Feature Importance ({trade_type.capitalize()}s)",
                 color=TEXT, fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, axis="x")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=PURPLE, label="Temporal (lags)"),
        Patch(facecolor=AMBER,  label="Macroeconomic"),
        Patch(facecolor=TEAL,   label="Structural (partners)"),
        Patch(facecolor="#d85a30", label="Shock indicators"),
    ], fontsize=8, framealpha=0.2, loc="lower right")
    plt.tight_layout()
    path = f"outputs/shap/{trade_type}_feature_importance.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    imp_s.sort_values(ascending=False).to_csv(
        f"outputs/shap/{trade_type}_feature_importance.csv", header=["importance"])
    log(f"Saved: {path}", 2)
    return imp_s

def save_tables(all_metrics, all_times):
    rows = []
    for tt in ["import","export"]:
        for name, m in all_metrics[tt].items():
            rows.append({"Trade Type":tt.capitalize(),"Model":name,
                         "MAE":m["MAE"],"RMSE":m["RMSE"],
                         "MAPE (%)":m["MAPE"],"R²":m["R2"],
                         "Train Time (s)":round(all_times[tt].get(name,0),1)})
    df = pd.DataFrame(rows)
    df.to_csv("outputs/tables/model_comparison.csv", index=False)
    summary = []
    for tt in ["import","export"]:
        best = min(all_metrics[tt], key=lambda k: all_metrics[tt][k]["RMSE"])
        bm = all_metrics[tt][best]
        summary.append({"Trade Type":tt.capitalize(),"Best Model":best,
                        "MAE":bm["MAE"],"RMSE":bm["RMSE"],
                        "MAPE (%)":bm["MAPE"],"R²":bm["R2"]})
    pd.DataFrame(summary).to_csv("outputs/tables/best_model_summary.csv", index=False)
    log("Saved: outputs/tables/model_comparison.csv", 2)
    log("Saved: outputs/tables/best_model_summary.csv", 2)
    return df

def print_analysis(all_metrics, imp_dfs):
    section("RESULT ANALYSIS — Chapter 4 Talking Points")
    for tt in ["import","export"]:
        m = all_metrics[tt]
        models = list(m.keys())
        best  = min(models, key=lambda k: m[k]["RMSE"])
        worst = max(models, key=lambda k: m[k]["RMSE"])
        log(f"\n{tt.upper()}S:", 0)
        log(f"Best model  : {best}  (RMSE={m[best]['RMSE']:.1f})", 1)
        log(f"Worst model : {worst} (RMSE={m[worst]['RMSE']:.1f})", 1)
        if "ARIMA" in m and "Hybrid" in m:
            imp = ((m["ARIMA"]["RMSE"]-m["Hybrid"]["RMSE"])/m["ARIMA"]["RMSE"])*100
            sign = "improvement" if imp>0 else "regression"
            log(f"Hybrid vs ARIMA: {abs(imp):.1f}% {sign} in RMSE", 1)
        if "LSTM" in m and "Hybrid" in m:
            imp = ((m["LSTM"]["RMSE"]-m["Hybrid"]["RMSE"])/m["LSTM"]["RMSE"])*100
            sign = "improvement" if imp>0 else "regression"
            log(f"Hybrid vs LSTM : {abs(imp):.1f}% {sign} in RMSE", 1)
        if tt in imp_dfs:
            top3 = imp_dfs[tt].sort_values(ascending=False).head(3)
            log(f"Top 3 drivers  : {', '.join(top3.index.tolist())}", 1)
    log("\nIMPORTS vs EXPORTS:", 0)
    for name in ["ARIMA","LSTM","Hybrid"]:
        if name in all_metrics["import"] and name in all_metrics["export"]:
            ir = all_metrics["import"][name]["RMSE"]
            er = all_metrics["export"][name]["RMSE"]
            log(f"{name}: imports RMSE={ir:.1f} | exports RMSE={er:.1f} → "
                f"{'imports' if ir<er else 'exports'} forecast more accurately", 1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    total = time.time()
    all_metrics = {"import":{}, "export":{}}
    all_times   = {"import":{}, "export":{}}
    imp_dfs     = {}

    for trade_type in ["import","export"]:
        section(f"TRADE TYPE: {trade_type.upper()}S")
        p = load_and_prep(trade_type)
        results   = {}
        histories = {}
        test_dates = p["test"]["date"].values

        log("[ 1 / 3 ]  ARIMA", 0)
        try:
            am, ap, at, am_m, at_t = run_arima(p)
            results["ARIMA"] = (at, ap, am_m)
            all_metrics[trade_type]["ARIMA"] = am_m
            all_times[trade_type]["ARIMA"]   = at_t
            joblib.dump(am, f"outputs/models/{trade_type}_arima.pkl")
            log(f"Saved: outputs/models/{trade_type}_arima.pkl", 2)
        except Exception as e:
            log(f"ARIMA failed: {e}", 2)

        log("[ 2 / 3 ]  LSTM", 0)
        try:
            lm, lp, lt, lm_m, lt_t, lh = run_lstm(p)
            results["LSTM"] = (lt, lp, lm_m)
            all_metrics[trade_type]["LSTM"] = lm_m
            all_times[trade_type]["LSTM"]   = lt_t
            histories["LSTM"] = lh
            lm.save(f"outputs/models/{trade_type}_lstm.keras")
            log(f"Saved: outputs/models/{trade_type}_lstm.keras", 2)
        except Exception as e:
            log(f"LSTM failed: {e}", 2)

        log("[ 3 / 3 ]  HYBRID", 0)
        try:
            hm, hp, ht, hm_m, ht_t, hh = run_hybrid(p)
            results["Hybrid"] = (ht, hp, hm_m)
            all_metrics[trade_type]["Hybrid"] = hm_m
            all_times[trade_type]["Hybrid"]   = ht_t
            histories["Hybrid"] = hh
            hm.save(f"outputs/models/{trade_type}_hybrid.keras")
            log(f"Saved: outputs/models/{trade_type}_hybrid.keras", 2)
        except Exception as e:
            log(f"Hybrid failed: {e}", 2)

        joblib.dump({"sx":p["sx"],"sy":p["sy"]},
                    f"outputs/models/{trade_type}_scalers.pkl")

        log("Generating plots...", 0)
        if results:
            plot_forecast(results, test_dates, trade_type)
            plot_residuals(results, test_dates, trade_type)
            plot_shock_analysis(results, p["test"], trade_type)
        if histories:
            plot_loss(histories, trade_type)

        if "Hybrid" in results:
            log("Feature importance (Hybrid)...", 0)
            imp_dfs[trade_type] = plot_feature_importance(
                hm, p["feat_cols"], p["X_te"], p["sy"], trade_type)

        rows = [{"Model":k, **v[2]} for k,v in results.items()]
        pd.DataFrame(rows).to_csv(f"outputs/metrics/{trade_type}_metrics.csv", index=False)
        log(f"Saved: outputs/metrics/{trade_type}_metrics.csv", 2)

    section("SAVING SUMMARY TABLES")
    cmp = save_tables(all_metrics, all_times)
    print(cmp.to_string(index=False))
    plot_comparison_bars(all_metrics)
    print_analysis(all_metrics, imp_dfs)

    section(f"COMPLETE  ({(time.time()-total)/60:.1f} min total)")
    log("All outputs saved to outputs/", 0)

if __name__ == "__main__":
    main()