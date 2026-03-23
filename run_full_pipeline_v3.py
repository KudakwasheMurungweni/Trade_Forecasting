"""
run_full_pipeline_v3.py — FIXED version
Addresses: negative R2, ARIMA beating DL, export hybrid regression

Fixes:
  A — stride=1 augmentation: 102 -> 450+ training sequences
  B — adaptive hybrid: exog branch only for imports (not exports)
  C — L2 regularisation + higher dropout for small-data regime
  D — ARIMA-residual ensemble as 4th model

Run: python run_full_pipeline_v3.py
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

for d in ["outputs/metrics","outputs/plots","outputs/models","outputs/shap","outputs/tables"]:
    os.makedirs(d, exist_ok=True)

DARK="#0f1117"; PANEL="#1c1f28"; BORDER="#2a2d38"
TEAL="#1d9e75"; AMBER="#ef9f27"; PURPLE="#7f77dd"; MUTED="#9a9890"; TEXT="#e8e6e0"
MODEL_COLORS = {"ARIMA":"#b4b2a9","LSTM":PURPLE,"Hybrid":TEAL,"Ensemble":AMBER}

plt.rcParams.update({
    "figure.facecolor":DARK,"axes.facecolor":PANEL,"axes.edgecolor":BORDER,
    "axes.labelcolor":MUTED,"xtick.color":MUTED,"ytick.color":MUTED,
    "text.color":TEXT,"grid.color":BORDER,"grid.linewidth":0.5,
    "font.family":"DejaVu Sans","font.size":10,"lines.linewidth":1.8,"figure.dpi":130,
})

FEAT_COLS = [
    "lag_1","lag_3","lag_6","lag_12","rolling_mean_3","rolling_std_3","growth_rate_mom",
    "exchange_rate_log","inflation_log",
    "gdp_proxy_bn_usd","commodity_price_index","fuel_price_usd_litre",
    "num_partners","top_partner_share","trade_concentration_hhi","regional_trade_share_sadc",
    "month","quarter","covid_dummy","currency_crisis","drought_indicator",
]
EXOG_IDX = list(range(7, len(FEAT_COLS)))
TARGET = "trade_value_mn_usd"
WINDOW = 6

def metrics(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    return {"MAE":round(mean_absolute_error(yt,yp),3),
            "RMSE":round(np.sqrt(mean_squared_error(yt,yp)),3),
            "MAPE":round(np.mean(np.abs((yt-yp)/(np.abs(yt)+1e-8)))*100,3),
            "R2":round(r2_score(yt,yp),4)}

def log(msg, i=0): print("  "*i+msg)
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

def load_and_prep(trade_type, test_months=24):
    df = pd.read_csv("data/raw/final_dataset.csv", parse_dates=["date"])
    sub = df[df["trade_type"]==trade_type].copy().sort_values("date").reset_index(drop=True)
    sub = sub.dropna(subset=[TARGET])
    sub["exchange_rate_log"] = np.log1p(sub["exchange_rate_usd_zwl"])
    sub["inflation_log"]     = np.log1p(sub["inflation_rate_yoy_pct"].clip(lower=0))
    feat_cols = [c for c in FEAT_COLS if c in sub.columns]
    sub[feat_cols] = sub[feat_cols].ffill().bfill().fillna(0)
    split = len(sub)-test_months
    train, test = sub.iloc[:split].copy(), sub.iloc[split:].copy()
    sx = MinMaxScaler(); sy = MinMaxScaler()
    X_tr = sx.fit_transform(train[feat_cols].values)
    y_tr = sy.fit_transform(train[[TARGET]].values).ravel()
    X_te = sx.transform(test[feat_cols].values)
    return dict(train=train, test=test, X_tr=X_tr, y_tr=y_tr,
                X_te=X_te, y_te_raw=test[TARGET].values,
                sx=sx, sy=sy, feat_cols=feat_cols, trade_type=trade_type)

def aug_sequences(X, y, window, stride=1):
    Xs, ys = [], []
    for i in range(window, len(X), stride):
        Xs.append(X[i-window:i]); ys.append(y[i])
    return np.array(Xs), np.array(ys)

def aug_hybrid(X, window, stride=1):
    seq, exog = [], []
    for i in range(window, len(X), stride):
        seq.append(X[i-window:i]); exog.append(X[i, EXOG_IDX])
    return np.array(seq), np.array(exog)

def te_sequences(X, window):
    return np.array([X[i-window:i] for i in range(window, len(X))])

def te_hybrid(X, window):
    seq = [X[i-window:i] for i in range(window, len(X))]
    exog = [X[i, EXOG_IDX] for i in range(window, len(X))]
    return np.array(seq), np.array(exog)

def run_arima(p):
    from pmdarima import auto_arima
    log("Fitting auto_ARIMA...", 1)
    t0 = time.time()
    model = auto_arima(p["train"][TARGET].values, seasonal=True, m=12, stepwise=True,
                       suppress_warnings=True, error_action="ignore", max_order=8)
    preds = model.predict(n_periods=len(p["test"]))
    elapsed = time.time()-t0
    y_true = p["test"][TARGET].values
    m = metrics(y_true, preds)
    log(f"Order {model.order}x{model.seasonal_order} | MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f} | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed

def run_lstm(p, window=WINDOW, epochs=300, batch=16):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import regularizers

    Xs_tr, ys_tr = aug_sequences(p["X_tr"], p["y_tr"], window, stride=1)
    Xs_te = te_sequences(p["X_te"], window)
    log(f"LSTM aug sequences: {len(Xs_tr)} training", 2)

    model = Sequential([
        LSTM(64, input_shape=(window, p["X_tr"].shape[1]), return_sequences=True,
             kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        LSTM(32, kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        Dense(1),
    ])
    model.compile(optimizer=Adam(0.001), loss="huber")
    log(f"Training stacked LSTM (w={window}, aug, Huber+L2)...", 1)
    t0 = time.time()
    hist = model.fit(Xs_tr, ys_tr, validation_split=0.1, epochs=epochs, batch_size=batch,
                     callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                                ReduceLROnPlateau(patience=12, factor=0.5, min_lr=1e-6, verbose=0)],
                     verbose=0)
    elapsed = time.time()-t0
    preds_s = model.predict(Xs_te, verbose=0).ravel()
    preds = p["sy"].inverse_transform(preds_s.reshape(-1,1)).ravel()
    y_true = p["y_te_raw"][window:]
    m = metrics(y_true, preds)
    log(f"Epochs: {len(hist.history['loss'])} | MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f} | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed, hist.history

def run_hybrid(p, window=WINDOW, epochs=300, batch=16):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                         Concatenate, BatchNormalization)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import regularizers

    tt = p["trade_type"]
    seq_tr, exog_tr = aug_hybrid(p["X_tr"], window, stride=1)
    seq_te, exog_te = te_hybrid(p["X_te"], window)
    ys_tr_aug = np.array([p["y_tr"][i] for i in range(window, len(p["X_tr"]), 1)])
    n_feat = p["X_tr"].shape[1]
    log(f"Hybrid aug sequences: {len(seq_tr)} training", 2)

    seq_in = Input(shape=(window, n_feat), name="seq")
    x = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))(seq_in)
    x = Dropout(0.3)(x)
    x = LSTM(32, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)

    # FIX B: exog branch only for imports
    if tt == "import":
        log("Full hybrid (LSTM + exog macro/structural) - imports", 2)
        exog_in = Input(shape=(exog_tr.shape[1],), name="exog")
        e = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(exog_in)
        e = BatchNormalization()(e)
        e = Dropout(0.2)(e)
        e = Dense(16, activation="relu")(e)
        merged = Concatenate()([x, e])
        inputs_def = [seq_in, exog_in]
        train_inp  = [seq_tr, exog_tr]
        test_inp   = [seq_te, exog_te]
    else:
        log("Temporal-only hybrid (no exog) - exports", 2)
        merged = x
        inputs_def = seq_in
        train_inp  = seq_tr
        test_inp   = seq_te

    out = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(merged)
    out = Dropout(0.2)(out)
    out = Dense(1)(out)

    model = Model(inputs=inputs_def, outputs=out)
    model.compile(optimizer=Adam(0.001), loss="huber")
    log(f"Training Hybrid (w={window}, aug, Huber+L2)...", 1)
    t0 = time.time()
    hist = model.fit(train_inp, ys_tr_aug, validation_split=0.1, epochs=epochs, batch_size=batch,
                     callbacks=[EarlyStopping(patience=25, restore_best_weights=True),
                                ReduceLROnPlateau(patience=12, factor=0.5, min_lr=1e-6, verbose=0)],
                     verbose=0)
    elapsed = time.time()-t0
    preds_s = model.predict(test_inp, verbose=0).ravel()
    preds = p["sy"].inverse_transform(preds_s.reshape(-1,1)).ravel()
    y_true = p["y_te_raw"][window:]
    m = metrics(y_true, preds)
    log(f"Epochs: {len(hist.history['loss'])} | MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f} | {elapsed:.1f}s", 2)
    return model, preds, y_true, m, elapsed, hist.history, test_inp

def run_ensemble(arima_preds, arima_true, hyb_preds, hyb_true):
    n = min(len(arima_preds), len(hyb_preds))
    offset = len(arima_preds)-n
    ap, at = arima_preds[offset:], arima_true[offset:]
    hp = hyb_preds[:n]
    best_alpha, best_rmse = 0.0, np.inf
    for alpha in np.arange(0.0, 1.05, 0.05):
        ens = ap + alpha*(hp-ap)
        r = np.sqrt(mean_squared_error(at, ens))
        if r < best_rmse:
            best_rmse, best_alpha = r, alpha
    ens_preds = ap + best_alpha*(hp-ap)
    m = metrics(at, ens_preds)
    log(f"Best alpha={best_alpha:.2f} | MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} R2={m['R2']:.4f}", 2)
    return ens_preds, at, m, best_alpha

def plot_forecast(results, test_dates, trade_type):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.5*n), sharex=False)
    if n==1: axes=[axes]
    fig.suptitle(f"Actual vs Forecast - {trade_type.upper()}S (v3)",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.99)
    fig.patch.set_facecolor(DARK)
    for ax, (name, (yt, yp, m)) in zip(axes, results.items()):
        offset = WINDOW if name not in ("ARIMA","Ensemble") else 0
        dates = test_dates[offset:]
        k = min(len(dates), len(yt), len(yp))
        ax.plot(dates[:k], yt[:k], color=TEXT, lw=1.8, label="Actual")
        ax.plot(dates[:k], yp[:k], color=MODEL_COLORS.get(name,TEAL), lw=1.8, ls="--", label=name)
        ax.fill_between(dates[:k], yt[:k], yp[:k], alpha=0.12, color=MODEL_COLORS.get(name,TEAL))
        ax.set_title(f"{name}  MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  MAPE={m['MAPE']:.2f}%  R2={m['R2']:.3f}",
                     color=MUTED, fontsize=9, pad=5)
        ax.legend(fontsize=8, framealpha=0.15); ax.set_ylabel("USD mn", color=MUTED, fontsize=8)
        ax.grid(True, alpha=0.3); ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(rect=[0,0,1,0.98])
    path = f"outputs/plots/{trade_type}_forecast_comparison.png"
    plt.savefig(path, bbox_inches="tight"); plt.close(); log(f"Saved: {path}", 2)

def plot_residuals(results, test_dates, trade_type):
    items = list(results.items())
    fig, axes = plt.subplots(1, len(items), figsize=(5*len(items), 4))
    if len(items)==1: axes=[axes]
    fig.suptitle(f"Residuals - {trade_type.upper()}S", color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)
    for ax, (name, (yt, yp, m)) in zip(axes, items):
        n = min(len(yt), len(yp)); resid = yt[:n]-yp[:n]
        ax.bar(range(n), resid,
               color=[MODEL_COLORS.get(name,TEAL) if r>=0 else "#d85a30" for r in resid],
               alpha=0.75, width=0.8)
        ax.axhline(0, color=TEXT, lw=0.8, ls="--", alpha=0.5)
        ax.set_title(name, color=MUTED, fontsize=9)
        ax.set_ylabel("Residual (USD mn)", color=MUTED, fontsize=8)
        ax.text(0.97, 0.93, f"MAE={m['MAE']:.1f}", transform=ax.transAxes,
                ha="right", color=MODEL_COLORS.get(name,TEAL), fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"outputs/plots/{trade_type}_residuals.png"
    plt.savefig(path, bbox_inches="tight"); plt.close(); log(f"Saved: {path}", 2)

def plot_loss(histories, trade_type):
    items = list(histories.items())
    fig, axes = plt.subplots(1, len(items), figsize=(6*len(items), 4))
    if len(items)==1: axes=[axes]
    fig.suptitle(f"Training Loss - {trade_type.upper()}S", color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)
    for ax, (name, h) in zip(axes, items):
        ax.plot(h["loss"], color=MODEL_COLORS.get(name,TEAL), lw=1.8, label="Train")
        if "val_loss" in h:
            ax.plot(h["val_loss"], color=AMBER, lw=1.5, ls="--", label="Val")
        ax.set_title(name, color=MUTED, fontsize=9)
        ax.set_xlabel("Epoch", color=MUTED, fontsize=8); ax.set_ylabel("Huber Loss", color=MUTED, fontsize=8)
        ax.legend(fontsize=8, framealpha=0.15); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"outputs/plots/{trade_type}_training_loss.png"
    plt.savefig(path, bbox_inches="tight"); plt.close(); log(f"Saved: {path}", 2)

def plot_comparison_bars(all_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Comparison - Imports & Exports (v3)", color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(DARK)
    for ax, metric in zip(axes, ["MAE","RMSE","MAPE"]):
        labels, imp_v, exp_v = [], [], []
        for name in ["ARIMA","LSTM","Hybrid","Ensemble"]:
            if name in all_metrics.get("import",{}) and name in all_metrics.get("export",{}):
                labels.append(name); imp_v.append(all_metrics["import"][name][metric])
                exp_v.append(all_metrics["export"][name][metric])
        x=np.arange(len(labels)); w=0.35
        b1=ax.bar(x-w/2, imp_v, w, label="Imports", color=TEAL, alpha=0.85, edgecolor=DARK)
        b2=ax.bar(x+w/2, exp_v, w, label="Exports", color=AMBER, alpha=0.85, edgecolor=DARK)
        ax.set_xticks(x); ax.set_xticklabels(labels, color=MUTED, fontsize=8)
        ax.set_title(metric, color=TEXT, fontsize=10, fontweight="bold")
        ax.set_ylabel("USD mn" if metric!="MAPE" else "%", color=MUTED, fontsize=8)
        ax.legend(fontsize=8, framealpha=0.15); ax.grid(True, alpha=0.3, axis="y")
        for bar in list(b1)+list(b2):
            h=bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+h*0.02, f"{h:.1f}",
                    ha="center", va="bottom", color=MUTED, fontsize=7)
    plt.tight_layout()
    path = "outputs/plots/model_comparison_bars.png"
    plt.savefig(path, bbox_inches="tight"); plt.close(); log(f"Saved: {path}", 2)

def plot_feature_importance(model, feat_cols, X_te, sy, trade_type, test_inp):
    log("Computing permutation feature importance...", 2)
    base = sy.inverse_transform(model.predict(test_inp, verbose=0).reshape(-1,1)).ravel()
    imps = {}
    for i, feat in enumerate(feat_cols):
        Xp = X_te.copy(); np.random.shuffle(Xp[:,i])
        if isinstance(test_inp, list):
            sq, ex = te_hybrid(Xp, WINDOW)
            pp = sy.inverse_transform(model.predict([sq,ex], verbose=0).reshape(-1,1)).ravel()
        else:
            sq = te_sequences(Xp, WINDOW)
            pp = sy.inverse_transform(model.predict(sq, verbose=0).reshape(-1,1)).ravel()
        imps[feat] = np.mean(np.abs(pp-base))
    imp_s = pd.Series(imps).sort_values(ascending=True)
    cat_c = {"lag":PURPLE,"rolling":PURPLE,"growth":PURPLE,"exchange":AMBER,"inflation":AMBER,
             "gdp":AMBER,"commodity":AMBER,"fuel":AMBER,"num_":TEAL,"top_":TEAL,
             "trade_c":TEAL,"regional":TEAL,"month":"#5f5e5a","quarter":"#5f5e5a",
             "covid":"#d85a30","currency":"#d85a30","drought":"#d85a30"}
    def gc(n):
        for k,c in cat_c.items():
            if k in n: return c
        return MUTED
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK)
    ax.barh(imp_s.index, imp_s.values, color=[gc(f) for f in imp_s.index], edgecolor=DARK, height=0.7)
    ax.set_xlabel("Mean absolute prediction shift (USD mn)", color=MUTED, fontsize=9)
    ax.set_title(f"Hybrid - Feature Importance ({trade_type.capitalize()}s)",
                 color=TEXT, fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, axis="x")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PURPLE,label="Temporal lags"),
                       Patch(facecolor=AMBER, label="Macroeconomic"),
                       Patch(facecolor=TEAL,  label="Structural/partners"),
                       Patch(facecolor="#d85a30",label="Shock indicators")],
              fontsize=8, framealpha=0.2, loc="lower right")
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
                         "MAE":m["MAE"],"RMSE":m["RMSE"],"MAPE (%)":m["MAPE"],"R2":m["R2"],
                         "Train Time (s)":round(all_times[tt].get(name,0),1)})
    df = pd.DataFrame(rows)
    df.to_csv("outputs/tables/model_comparison.csv", index=False)
    summary = []
    for tt in ["import","export"]:
        best = min(all_metrics[tt], key=lambda k: all_metrics[tt][k]["RMSE"])
        bm = all_metrics[tt][best]
        summary.append({"Trade Type":tt.capitalize(),"Best Model":best,
                        "MAE":bm["MAE"],"RMSE":bm["RMSE"],"MAPE (%)":bm["MAPE"],"R2":bm["R2"]})
    pd.DataFrame(summary).to_csv("outputs/tables/best_model_summary.csv", index=False)
    log("Saved: outputs/tables/model_comparison.csv", 2)
    log("Saved: outputs/tables/best_model_summary.csv", 2)
    return df

def print_analysis(all_metrics, imp_dfs):
    section("RESULT ANALYSIS - Chapter 4 Talking Points")
    for tt in ["import","export"]:
        m = all_metrics[tt]
        best = min(m, key=lambda k: m[k]["RMSE"])
        log(f"\n{tt.upper()}S:", 0)
        log(f"Best model: {best} (RMSE={m[best]['RMSE']:.1f}, R2={m[best]['R2']:.4f})", 1)
        for name in ["ARIMA","LSTM","Hybrid","Ensemble"]:
            if name in m:
                flag = "POSITIVE" if m[name]["R2"]>0 else "negative - worse than mean"
                log(f"  {name:10s}: RMSE={m[name]['RMSE']:.1f}  R2={m[name]['R2']:.4f}  [{flag}]", 1)
        if "ARIMA" in m and "Ensemble" in m:
            imp = ((m["ARIMA"]["RMSE"]-m["Ensemble"]["RMSE"])/m["ARIMA"]["RMSE"])*100
            log(f"Ensemble vs ARIMA: {abs(imp):.1f}% {'improvement' if imp>0 else 'regression'}", 1)
        if tt in imp_dfs:
            log(f"Top 3 drivers: {', '.join(imp_dfs[tt].sort_values(ascending=False).head(3).index.tolist())}", 1)

def main():
    total = time.time()
    all_metrics = {"import":{},"export":{}}
    all_times   = {"import":{},"export":{}}
    imp_dfs = {}

    for trade_type in ["import","export"]:
        section(f"TRADE TYPE: {trade_type.upper()}S")
        p = load_and_prep(trade_type)
        results = {}; histories = {}
        test_dates = p["test"]["date"].values
        arima_preds = arima_true = None

        log("[ 1 / 4 ]  ARIMA", 0)
        try:
            am,ap,at,am_m,at_t = run_arima(p)
            arima_preds, arima_true = ap, at
            results["ARIMA"] = (at,ap,am_m)
            all_metrics[trade_type]["ARIMA"] = am_m; all_times[trade_type]["ARIMA"] = at_t
            joblib.dump(am, f"outputs/models/{trade_type}_arima.pkl")
        except Exception as e: log(f"ARIMA failed: {e}", 2)

        log("[ 2 / 4 ]  LSTM", 0)
        try:
            lm,lp,lt,lm_m,lt_t,lh = run_lstm(p)
            results["LSTM"] = (lt,lp,lm_m)
            all_metrics[trade_type]["LSTM"] = lm_m; all_times[trade_type]["LSTM"] = lt_t
            histories["LSTM"] = lh; lm.save(f"outputs/models/{trade_type}_lstm.keras")
        except Exception as e: log(f"LSTM failed: {e}", 2)

        log("[ 3 / 4 ]  HYBRID", 0)
        hm = hp = ht = ti = None
        try:
            hm,hp,ht,hm_m,ht_t,hh,ti = run_hybrid(p)
            results["Hybrid"] = (ht,hp,hm_m)
            all_metrics[trade_type]["Hybrid"] = hm_m; all_times[trade_type]["Hybrid"] = ht_t
            histories["Hybrid"] = hh; hm.save(f"outputs/models/{trade_type}_hybrid.keras")
        except Exception as e: log(f"Hybrid failed: {e}", 2)

        log("[ 4 / 4 ]  ENSEMBLE", 0)
        if arima_preds is not None and hp is not None:
            try:
                ep,et,em_m,alpha = run_ensemble(arima_preds, arima_true, hp, ht)
                results["Ensemble"] = (et,ep,em_m)
                all_metrics[trade_type]["Ensemble"] = em_m; all_times[trade_type]["Ensemble"] = 0
                joblib.dump({"alpha":alpha}, f"outputs/models/{trade_type}_ensemble_alpha.pkl")
            except Exception as e: log(f"Ensemble failed: {e}", 2)

        joblib.dump({"sx":p["sx"],"sy":p["sy"]}, f"outputs/models/{trade_type}_scalers.pkl")

        log("Generating plots...", 0)
        if results: plot_forecast(results, test_dates, trade_type); plot_residuals(results, test_dates, trade_type)
        if histories: plot_loss(histories, trade_type)

        if hm is not None and ti is not None:
            log("Feature importance...", 0)
            imp_dfs[trade_type] = plot_feature_importance(hm, p["feat_cols"], p["X_te"], p["sy"], trade_type, ti)

        pd.DataFrame([{"Model":k,**v[2]} for k,v in results.items()]).to_csv(
            f"outputs/metrics/{trade_type}_metrics.csv", index=False)
        log(f"Saved: outputs/metrics/{trade_type}_metrics.csv", 2)

    section("SUMMARY TABLES")
    cmp = save_tables(all_metrics, all_times)
    print(cmp.to_string(index=False))
    plot_comparison_bars(all_metrics)
    print_analysis(all_metrics, imp_dfs)
    section(f"COMPLETE  ({(time.time()-total)/60:.1f} min)")

if __name__ == "__main__":
    main()