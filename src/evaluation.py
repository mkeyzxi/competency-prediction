import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, make_scorer, balanced_accuracy_score, roc_auc_score,
    average_precision_score, fbeta_score, matthews_corrcoef
)
from sklearn.utils import resample
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
import json
from src.utils import load_config
from src.tuning import build_search_estimator


def recall_bk_score(y_true, y_pred):
    """Recall for Belum Kompeten (pos_label=0)."""
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def precision_bk_score(y_true, y_pred):
    """Precision for Belum Kompeten (pos_label=0)."""
    return precision_score(y_true, y_pred, pos_label=0, zero_division=0)


def f1_bk_score(y_true, y_pred):
    """F1 for Belum Kompeten (pos_label=0)."""
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)


def f2_bk_score(y_true, y_pred):
    """F2 for Belum Kompeten (pos_label=0). Weights recall 2× more than precision."""
    return fbeta_score(y_true, y_pred, beta=2, pos_label=0, zero_division=0)


def evaluate_cv(model, model_name, X_train, y_train,
                config_path='configs/experiment_config.yaml'):
    """
    Evaluates the model using Nested Cross-Validation.

    Outer loop: RepeatedStratifiedKFold (evaluation)
    Inner loop: RandomizedSearchCV inside build_search_estimator (tuning)

    The test set is NEVER used here. Model selection is based entirely
    on these outer-fold metrics.
    """
    config = load_config(config_path)
    cv_splits = config['cv']['n_splits']
    n_repeats = config['cv'].get('n_repeats', 5)
    random_state = config['random_state']

    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits, n_repeats=n_repeats, random_state=random_state
    )

    # Build inner CV search estimator (Nested CV setup)
    search_estimator = build_search_estimator(model, model_name, config_path)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        'recall_bk': make_scorer(recall_bk_score),
        'precision_bk': make_scorer(precision_bk_score),
        'f1_bk': make_scorer(f1_bk_score),
        'f2_bk': make_scorer(f2_bk_score),
        'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba',
                               multi_class='ovr'),
        'pr_auc': make_scorer(average_precision_score,
                              response_method='predict_proba', pos_label=0),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(search_estimator, X_train, y_train,
                                cv=cv, scoring=scoring)

    results = {}
    for metric in scoring.keys():
        test_scores = scores[f'test_{metric}']
        mean_val = np.mean(test_scores)
        std_val = np.std(test_scores)
        results[f'cv_{metric}_mean'] = mean_val
        results[f'cv_{metric}_std'] = std_val

        n = len(test_scores)
        sem = stats.sem(test_scores)
        ci = stats.t.interval(0.95, n - 1, loc=mean_val, scale=sem)
        results[f'cv_{metric}_ci_lower'] = ci[0]
        results[f'cv_{metric}_ci_upper'] = ci[1]

    return results


def evaluate_test(model, X_test, y_test, scenario_name, model_name,
                  threshold=None, output_dir='results/confusion_matrix'):
    """
    Evaluates on the held-out test set. This should be called exactly ONCE
    after the model, features, and threshold are all frozen.

    Returns (results_dict, y_pred).
    """
    if threshold is not None and hasattr(model, "predict_proba"):
        try:
            # Predict 0 (BK) if probability of BK >= threshold
            p_bk = model.predict_proba(X_test)[:, 0]
            y_pred = np.where(p_bk >= threshold, 0, 1)
        except Exception:
            y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)

    results = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'test_precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'test_recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'test_f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'test_recall_bk': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        'test_precision_bk': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        'test_f1_bk': f1_score(y_test, y_pred, pos_label=0, zero_division=0),
        'test_f2_bk': fbeta_score(y_test, y_pred, beta=2, pos_label=0, zero_division=0),
    }

    try:
        y_prob_bk = model.predict_proba(X_test)[:, 0]
        y_prob_k = model.predict_proba(X_test)[:, 1]
        results['test_roc_auc'] = roc_auc_score(y_test, y_prob_k)
        results['test_pr_auc'] = average_precision_score(y_test, y_prob_bk, pos_label=0)
    except Exception:
        # Dummy / non-probability model — AUC metrics undefined
        results['test_roc_auc'] = float('nan')
        results['test_pr_auc'] = float('nan')

    # Matthews Correlation Coefficient (overall association between prediction and label)
    results['test_mcc'] = matthews_corrcoef(y_test, y_pred)

    # Confusion Matrix (labels=[0,1] => BK as "positive" class)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tp_bk, fn_bk, fp_bk, tn_bk = cm.ravel()

    results.update({
        'test_tn': int(tn_bk),
        'test_fp': int(fp_bk),
        'test_fn': int(fn_bk),
        'test_tp': int(tp_bk),
        'test_specificity': tn_bk / (tn_bk + fp_bk) if (tn_bk + fp_bk) > 0 else 0.0,
    })

    # Bootstrap Confidence Intervals (95%, B=1000 non-parametric resamples)
    n_iterations = 1000
    boot_recalls = []
    boot_bal_acc = []
    boot_f2 = []
    boot_mcc = []

    X_test_arr = np.asarray(X_test, dtype=float)
    y_test_arr = np.asarray(y_test)

    for i in range(n_iterations):
        X_bs, y_bs = resample(X_test_arr, y_test_arr, random_state=i)
        if threshold is not None and hasattr(model, "predict_proba"):
            try:
                p_bk_bs = model.predict_proba(X_bs)[:, 0]
                y_pred_bs = np.where(p_bk_bs >= threshold, 0, 1)
            except Exception:
                y_pred_bs = model.predict(X_bs)
        else:
            y_pred_bs = model.predict(X_bs)

        boot_recalls.append(recall_score(y_bs, y_pred_bs, pos_label=0, zero_division=0))
        boot_bal_acc.append(balanced_accuracy_score(y_bs, y_pred_bs))
        boot_f2.append(fbeta_score(y_bs, y_pred_bs, beta=2, pos_label=0, zero_division=0))
        boot_mcc.append(matthews_corrcoef(y_bs, y_pred_bs))

    results['test_recall_bk_ci_lower'] = np.percentile(boot_recalls, 2.5)
    results['test_recall_bk_ci_upper'] = np.percentile(boot_recalls, 97.5)
    results['test_balanced_accuracy_ci_lower'] = np.percentile(boot_bal_acc, 2.5)
    results['test_balanced_accuracy_ci_upper'] = np.percentile(boot_bal_acc, 97.5)
    results['test_f2_bk_ci_lower'] = np.percentile(boot_f2, 2.5)
    results['test_f2_bk_ci_upper'] = np.percentile(boot_f2, 97.5)
    results['test_mcc_ci_lower'] = np.percentile(boot_mcc, 2.5)
    results['test_mcc_ci_upper'] = np.percentile(boot_mcc, 97.5)

    # ---- Figures & exports ----
    os.makedirs(output_dir, exist_ok=True)
    thresh_label = f' (t={threshold})' if threshold is not None else ''

    # (1) Annotated confusion matrix with class names + % values
    # Standard orientation: rows = Actual, columns = Predicted
    # With labels=[0,1]: cm[0,0]=TP(BK), cm[0,1]=FN, cm[1,0]=FP, cm[1,1]=TN
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_pct = cm / cm.sum()
    annot = np.empty(cm.shape, dtype=object)
    labels_cm = ['TP', 'FN', 'FP', 'TN']
    for i in range(2):
        for j in range(2):
            annot[i, j] = f'{labels_cm[i*2+j]}\n{cm[i, j]} ' \
                          f'({cm_pct[i, j] * 100:.0f}%)'
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred: BK (0)', 'Pred: K (1)'],
                yticklabels=['Actual: BK (0)', 'Actual: K (1)'],
                linewidths=0)
    ax.set_title(f'Confusion Matrix — {scenario_name} — {model_name}'
                 f'{thresh_label}', fontsize=11)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    cm_path = f'{output_dir}/cm_{scenario_name}_{model_name}.png'
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # (2) Save raw counts + metric table as JSON next to the plot
    def _safe_float(v):
        if isinstance(v, (float, np.floating)):
            return float(v)
        return v  # keep int / bool / None as-is

    cm_json = {
        'scenario': scenario_name,
        'model': model_name,
        'threshold': threshold,
        'confusion_matrix': {
            'tp_bk': int(tp_bk), 'fn_bk': int(fn_bk),
            'fp_bk': int(fp_bk), 'tn_bk': int(tn_bk),
        },
        'metrics': {k: _safe_float(v) for k, v in results.items()
                    if k in ('test_accuracy', 'test_balanced_accuracy',
                             'test_recall_bk', 'test_precision_bk',
                             'test_f1_bk', 'test_f2_bk', 'test_specificity',
                             'test_mcc', 'test_roc_auc', 'test_pr_auc')},
    }
    with open(f'{output_dir}/cm_{scenario_name}_{model_name}.json', 'w') as f:
        json.dump(cm_json, f, indent=2)

    return results, y_pred
