"""
Evaluation plots: ROC / Precision-Recall curves, threshold sensitivity,
multi-cutoff comparison, and combined confusion-matrix figure set.

All figures are written to results/plots/.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    average_precision_score, roc_auc_score,
)

PLOTS_DIR = 'results/plots'
CUTOFFS = ['W1', 'W2', 'W3', 'C_Full']


def _ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_roc_pr(model, X_test, y_test, threshold, scenario_name, model_name,
                roc_auc=None, pr_auc=None):
    """ROC + Precision-Recall curves for the holdout set, with the chosen
    operating threshold marked on the ROC curve."""
    _ensure_dir()
    y_prob_k = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(np.asarray(y_test), y_prob_k)
    p_bk = model.predict_proba(X_test)[:, 0]
    prec, rec, _ = precision_recall_curve(np.asarray(y_test), p_bk, pos_label=0)
    pr_auc_val = pr_auc if pr_auc is not None else average_precision_score(
        np.asarray(y_test), p_bk, pos_label=0)
    roc_auc_val = roc_auc if roc_auc is not None else roc_auc_score(
        np.asarray(y_test), y_prob_k)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(fpr, tpr, color='#4c72b0', lw=2,
            label=f'ROC (AUC={roc_auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.15, color='#4c72b0')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Recall BK)')
    ax.set_title(f'ROC Curve — {scenario_name} ({model_name})')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rec, prec, color='#55a868', lw=2,
            label=f'PR (AP={pr_auc_val:.3f})')
    base_rate = float(np.mean(np.asarray(y_test) == 0))
    ax.axhline(base_rate, color='k', ls=':', alpha=0.5,
               label=f'base rate (BK)={base_rate:.2f}')
    ax.set_xlabel('Recall (Sensitivity) — kelas BK')
    ax.set_ylabel('Precision — kelas BK')
    ax.set_title(f'Precision-Recall Curve — {scenario_name} ({model_name})')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = f'{PLOTS_DIR}/roc_pr_{scenario_name}_{model_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_threshold_sensitivity(results_df, scenario_name, model_name):
    """Line chart of Recall / Precision / F2 / Balanced Accuracy vs threshold,
    from the threshold sweep table (training-only evaluation)."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, color in [('recall_bk', '#e2474b'),
                       ('precision_bk', '#55a868'),
                       ('f2_bk', '#4c72b0'),
                       ('balanced_accuracy', '#999999')]:
        if col in results_df.columns:
            ax.plot(results_df['threshold'], results_df[col],
                    marker='o', ms=4, lw=1.5, color=color, label=col)
    # Mark the best threshold
    if 'balanced_accuracy' in results_df.columns and 'recall_bk' in results_df.columns:
        valid = results_df[results_df['recall_bk'] >= 0.70]
        if len(valid):
            valid = valid.sort_values(['balanced_accuracy', 'f2_bk'],
                                      ascending=False)
            best = valid.iloc[0]
            ax.axvline(best['threshold'], color='red', ls='--', alpha=0.6,
                       label=f"best t={best['threshold']:.2f}")
    ax.set_xlabel('Probability threshold (P(BK) >= t)')
    ax.set_ylabel('Score')
    ax.set_title(f'Threshold Sensitivity (train-set CV) — {scenario_name} '
                 f'({model_name})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = f'{PLOTS_DIR}/threshold_sensitivity_{scenario_name}_{model_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_cutoff_comparison(df_res, out_path=None):
    """Grouped bar chart of holdout metrics across the 4 cutoffs
    (W1, W2, W3, C_Full) + CV metrics overlay."""
    _ensure_dir()
    out_path = out_path or f'{PLOTS_DIR}/cutoff_comparison.png'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(df_res))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width / 2, df_res['Test_Recall_BK'], width,
           label='Test Recall BK', color='#e2474b')
    ax.bar(x + width / 2, df_res['Test_Specificity'], width,
           label='Test Specificity', color='#55a868')
    ax.axhline(0.5, color='k', ls=':', alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df_res['Cutoff'])
    ax.set_ylabel('Score')
    ax.set_title('Holdout: Recall BK vs Specificity per Cutoff')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.bar(x - width / 2, df_res['CV_Recall_BK'], width,
           label='CV Recall BK', color='#4c72b0')
    ax.bar(x + width / 2, df_res['CV_Balanced_Acc'], width,
           label='CV Balanced Accuracy', color='#999999')
    ax.set_xticks(x)
    ax.set_xticklabels(df_res['Cutoff'])
    ax.set_ylabel('Score')
    ax.set_title('Nested CV: Recall BK vs Balanced Accuracy per Cutoff')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    fig.suptitle('Multi-Cutoff EWS (S3_E, DecisionTree) — Holdout n=18',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_cm_grid(files, out_path=None):
    """Montage of several confusion-matrix PNGs in a single figure."""
    _ensure_dir()
    out_path = out_path or f'{PLOTS_DIR}/cm_grid.png'
    n = len(files)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, (fname, title) in enumerate(files):
        ax = axes[i]
        try:
            img = plt.imread(fname)
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
        except Exception:
            ax.text(0.5, 0.5, f'{fname}\n(file not found)',
                    ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_threshold_table(results, best_threshold, scenario_name, model_name):
    """Persist the full threshold sweep + best threshold as CSV."""
    _ensure_dir()
    df = pd.DataFrame(results)
    df['is_best'] = (abs(df['threshold'] - best_threshold) < 1e-9)
    path = (f'{PLOTS_DIR}/threshold_table_{scenario_name}_{model_name}.csv')
    df.to_csv(path, index=False)
    return path


def save_final_summary(summary_dict, path=f'{PLOTS_DIR}/../final/'):
    """Write a single consolidated JSON of all final numbers."""
    os.makedirs('results/final', exist_ok=True)
    with open('results/final/consolidated_summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=2, default=float)
