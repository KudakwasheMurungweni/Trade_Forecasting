import shap
import numpy as np
import matplotlib.pyplot as plt

def shap_analysis(model, X_background, X_explain, feature_names, model_type="lstm"):
    if model_type == "lstm":
        # Flatten for SHAP TreeExplainer-style — use KernelExplainer for Keras
        predict_fn = lambda x: model.predict(x.reshape(-1, *X_background.shape[1:]))
        explainer = shap.KernelExplainer(predict_fn, shap.sample(X_background, 50))
        shap_vals = explainer.shap_values(X_explain[:20])
    else:
        explainer = shap.Explainer(model, X_background)
        shap_vals = explainer(X_explain)
    return explainer, shap_vals

def plot_feature_importance(model, feature_names, top_n=15):
    """Works for sklearn models with feature_importances_."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx], importances[idx], color="#1D9E75")
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance (top features)")
    plt.tight_layout()
    return fig
