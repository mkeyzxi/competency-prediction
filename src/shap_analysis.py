"""
SHAP (Explainable AI) analysis for tree-based EWS models.

Produces, for a given frozen model + holdout set:
  - global_importance_<tag>.png   : mean |SHAP| bar chart
  - global_importance_<tag>.csv   : ranked mean |SHAP| table
  - beeswarm_<tag>.png            : SHAP summary (beeswarm) plot
  - local_<Case>_<tag>.png        : SHAP waterfall for representative TP/TN/FP/FN cases
  - shap_values_<tag>.csv         : per-sample SHAP values (BK class)

All figures are saved under results/shap/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from src.utils import load_model

SHAP_DIR = 'results/shap'


def _get_clf_and_input(model, X_test):
    """Unwrap a (possibly pipeline-wrapped) estimator and align X with it."""
    if hasattr(model, 'named_steps'):
        imputer = model.named_steps.get('imputer')
        clf = model.named_steps['model']
        if imputer is not None:
            X_t = pd.DataFrame(imputer.transform(X_test),
                               columns=X_test.columns, index=X_test.index)
        else:
            X_t = X_test
        return clf, X_t
    return model, X_test


def run_shap_analysis(model_path, X_test, y_test, y_pred, scenario, model_name):
    """Runs TreeSHAP explanation for the given model and test set."""
    os.makedirs(SHAP_DIR, exist_ok=True)
    model = load_model(model_path)
    clf, X_t = _get_clf_and_input(model, X_test)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_t)

    # Explanation for the "Belum Kompeten" class (index 0)
    vals = getattr(shap_values, 'values', shap_values)
    if isinstance(vals, list):
        shap_pos = np.asarray(vals[0])
    elif np.ndim(vals) == 3:
        shap_pos = vals[:, :, 0]
    else:
        shap_pos = np.asarray(vals)
    base_value = float(np.asarray(getattr(shap_values, 'base_values',
                                          np.asarray(shap_pos).mean(0)))
                       .flatten()[0])

    features = list(X_t.columns)

    # ---- 1. Global importance (mean |SHAP|) -------------------------------
    mean_abs = np.abs(shap_pos).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fi = (pd.DataFrame({'Feature': [features[i] for i in order],
                        'Mean_Abs_SHAP': mean_abs[order]})
          .rename(columns={'Feature': 'feature', 'Mean_Abs_SHAP': 'mean_abs_shap'})
          .reset_index().rename(columns={'index': 'rank'})
          .to_csv(f'{SHAP_DIR}/global_importance_{scenario}_{model_name}.csv',
                  index=False))

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(order) + 1.5)))
    shap.summary_plot(shap_pos, X_t, plot_type='bar', show=False,
                      max_display=15)
    ax.set_title(f'Global SHAP Importance — {scenario} ({model_name})',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{SHAP_DIR}/global_importance_{scenario}_{model_name}.png',
                dpi=150)
    plt.close(fig)

    # ---- 2. Beeswarm ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_pos, X_t, show=False, max_display=15)
    ax.set_title(f'SHAP Beeswarm — {scenario} ({model_name})', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{SHAP_DIR}/beeswarm_{scenario}_{model_name}.png', dpi=150)
    plt.close(fig)

    # ---- 3. Local waterfall plots for representative cases ------------------
    y_true = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()

    def _first(idx, fallback=None):
        return int(idx[0]) if len(idx) > 0 else fallback

    tp_idx = _first(np.where((y_true == 0) & (y_pred == 0))[0])
    tn_idx = _first(np.where((y_true == 1) & (y_pred == 1))[0])
    fp_idx = _first(np.where((y_true == 1) & (y_pred == 0))[0])
    fn_idx = _first(np.where((y_true == 0) & (y_pred == 1))[0])

    base_value_local = base_value

    for case_type, idx in {'TP': tp_idx, 'TN': tn_idx, 'FP': fp_idx,
                           'FN': fn_idx}.items():
        if idx is None:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        try:
            shap.plots.waterfall(shap.Explanation(
                values=shap_pos[idx],
                base_values=base_value_local,
                data=X_t.iloc[idx].values,
                feature_names=features,
            ), show=False)
        except Exception:
            vals = shap_pos[idx]
            pos = vals > 0
            ax.bar(range(len(vals)), vals,
                   color=np.where(pos, '#e2474b', '#4c72b0'))
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_ylabel('SHAP value')
        ax.set_title(f'Local SHAP Waterfall — {case_type} — {scenario} '
                     f'({model_name})', fontsize=12)
        fig.tight_layout()
        fig.savefig(f'{SHAP_DIR}/local_{case_type}_{scenario}_{model_name}.png',
                    dpi=150)
        plt.close(fig)

    # ---- 4. Export per-sample SHAP values -----------------------------------
    shap_df = pd.DataFrame(shap_pos, columns=features, index=X_t.index)
    shap_df.to_csv(f'{SHAP_DIR}/shap_values_{scenario}_{model_name}.csv')

    print(f"SHAP analysis complete for {scenario} {model_name}. "
          f"Top features: {[features[i] for i in order[:3]]}")
    return fi
