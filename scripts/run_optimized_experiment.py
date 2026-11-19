"""
Eksperimen Optimasi Terkontrol — EWS Competency Prediction

PROTOKOL KETAT:
1. Model selection HANYA dari Outer CV metrics (training data)
2. Holdout 20% TIDAK PERNAH disentuh untuk memilih model
3. Setelah model terpilih → freeze → fit ulang di seluruh train → 1× evaluasi holdout
4. Threshold dipilih di inner CV (tidak melihat holdout)

Alur eksekusi per cutoff (W1, W2, W3, C_Full):
  1. Load data/features/{cutoff}_S3_E.csv
  2. Train/test split (stratified, 20% holdout)
  3. Nested CV (outer 5×5 repeats, inner RandomizedSearchCV 3-fold)
  4. Tune hyperparameters + threshold (inner CV only)
  5. Re-fit pada seluruh train set → evaluasi holdout SEKALI
  6. Simpan model, confusion matrix, kurva ROC/PR, tabel threshold, SHAP
"""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_cv, evaluate_test
from src.utils import save_model
from src.shap_analysis import run_shap_analysis
from src import plots

warnings.filterwarnings('ignore')

CUTOFFS = ['W1', 'W2', 'W3', 'C_Full']
SCENARIO = 'S3_E'
MODEL_NAME = 'DecisionTree'
BALANCING = 'None'
CONFIG_PATH = 'configs/experiment_config.yaml'


def evaluate_one_cutoff(cutoff: str):
    """Run the full pipeline for a single cutoff. Returns (row, final_model)."""
    file_path = f'data/features/{cutoff}_{SCENARIO}.csv'
    if not os.path.exists(file_path):
        print(f"Dataset {file_path} not found. Skipping...")
        return None

    df = pd.read_csv(file_path)
    print(f"\n{'='*60}\n--- Menguji Cutoff: {cutoff} ---\n{'='*60}")

    X_train_full, X_test_full, y_train, y_test = get_train_test_split(
        df, CONFIG_PATH)
    print(f"Train: {len(y_train)} samples "
          f"(BK={sum(y_train==0)}, K={sum(y_train==1)})")
    print(f"Test:  {len(y_test)} samples "
          f"(BK={sum(y_test==0)}, K={sum(y_test==1)}) — FROZEN")

    features = get_features(SCENARIO)
    available_features = [f for f in features if f in X_train_full.columns]
    X_train = X_train_full[available_features]
    X_test = X_test_full[available_features]

    # ---- Phase 1 & 2: Nested CV + inner tuning (train set only) ----
    base_model = get_model(MODEL_NAME, BALANCING, CONFIG_PATH)
    cv_res = evaluate_cv(base_model, MODEL_NAME, X_train, y_train, CONFIG_PATH)

    tuned_model, tuned_params = tune_hyperparameters(
        base_model, MODEL_NAME, X_train, y_train, CONFIG_PATH,
        balancing=BALANCING)
    thresh_results, best_threshold, _ = evaluate_thresholds(
        tuned_model, X_train, y_train, optimize_for='f2_bk')

    # Persist full threshold sweep (used for sensitivity figure)
    plots.save_threshold_table(
        thresh_results, best_threshold, f'{cutoff}_{SCENARIO}', MODEL_NAME)

    print(f"CV Recall BK:        {cv_res['cv_recall_bk_mean']:.4f} "
          f"± {cv_res['cv_recall_bk_std']:.4f}")
    print(f"CV Balanced Acc:     {cv_res['cv_balanced_accuracy_mean']:.4f} "
          f"± {cv_res['cv_balanced_accuracy_std']:.4f}")
    print(f"CV F2 BK:            {cv_res['cv_f2_bk_mean']:.4f}")
    print(f"Best threshold:      {best_threshold}")
    print(f"Tuned params:        {tuned_params}")

    # ---- Phase 3: freeze model, evaluate holdout exactly once ----
    final_model = tuned_model  # already fitted on the entire train set
    os.makedirs('models', exist_ok=True)
    model_path = f'models/FINAL_{MODEL_NAME}_{SCENARIO}_{BALANCING}_{cutoff}.pkl'
    save_model(final_model, model_path)

    test_results, y_pred = evaluate_test(
        final_model, X_test, y_test,
        f'EWS_{cutoff}_{SCENARIO}', MODEL_NAME,
        threshold=best_threshold,
        output_dir='results/confusion_matrix')

    # ---- Print summary ----
    print(f"\n--- {cutoff} HOLDOUT RESULTS ---")
    print(f"  Confusion Matrix: TP={test_results.get('test_tp', 0)}, "
          f"FN={test_results.get('test_fn', 0)}, "
          f"FP={test_results.get('test_fp', 0)}, "
          f"TN={test_results.get('test_tn', 0)}")
    print(f"  Recall BK:         {test_results['test_recall_bk']:.4f} "
          f"(95% CI: [{test_results['test_recall_bk_ci_lower']:.4f}, "
          f"{test_results['test_recall_bk_ci_upper']:.4f}])")
    print(f"  Balanced Acc:      {test_results['test_balanced_accuracy']:.4f} "
          f"(95% CI: [{test_results['test_balanced_accuracy_ci_lower']:.4f}, "
          f"{test_results['test_balanced_accuracy_ci_upper']:.4f}])")
    print(f"  F2 BK:             {test_results['test_f2_bk']:.4f} "
          f"(95% CI: [{test_results['test_f2_bk_ci_lower']:.4f}, "
          f"{test_results['test_f2_bk_ci_upper']:.4f}])")
    print(f"  MCC:               {test_results.get('test_mcc', 0):.4f} "
          f"(95% CI: [{test_results.get('test_mcc_ci_lower', 0):.4f}, "
          f"{test_results.get('test_mcc_ci_upper', 0):.4f}])")
    print(f"  Specificity:       {test_results.get('test_specificity', 0):.4f}")
    print(f"  ROC AUC:           {test_results.get('test_roc_auc', 0):.4f}")
    print(f"  PR AUC:            {test_results.get('test_pr_auc', 0):.4f}")

    # ---- Figures ----
    plots.plot_roc_pr(
        final_model, X_test, y_test, best_threshold,
        f'EWS_{cutoff}_{SCENARIO}', MODEL_NAME,
        roc_auc=test_results.get('test_roc_auc'),
        pr_auc=test_results.get('test_pr_auc'))
    plots.plot_threshold_sensitivity(
        pd.DataFrame(thresh_results), f'EWS_{cutoff}_{SCENARIO}', MODEL_NAME)

    # ---- SHAP explanations on the holdout set ----
    run_shap_analysis(
        model_path, X_test, y_test, y_pred,
        f'EWS_{cutoff}_{SCENARIO}', MODEL_NAME)

    # ---- JSON report per cutoff ----
    os.makedirs('results/final', exist_ok=True)
    report = {
        'cutoff': cutoff,
        'model': MODEL_NAME,
        'scenario': SCENARIO,
        'balancing': BALANCING,
        'threshold': float(best_threshold),
        'n_features': len(available_features),
        'features': available_features,
        'cv_metrics': {
            'recall_bk': float(cv_res['cv_recall_bk_mean']),
            'recall_bk_std': float(cv_res['cv_recall_bk_std']),
            'balanced_accuracy': float(cv_res['cv_balanced_accuracy_mean']),
            'f2_bk': float(cv_res['cv_f2_bk_mean']),
        },
        'holdout_metrics': {
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in test_results.items()
        },
        'best_params': tuned_params,
        'model_path': model_path,
    }
    with open(f'results/final/final_model_report_{cutoff}.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)

    # ---- Row for the comparison table ----
    row = {
        'Cutoff': cutoff,
        'Threshold': best_threshold,
        'CV_Recall_BK': cv_res['cv_recall_bk_mean'],
        'CV_Recall_BK_CI_Low': cv_res.get('cv_recall_bk_ci_lower', 0),
        'CV_Recall_BK_CI_Upper': cv_res.get('cv_recall_bk_ci_upper', 0),
        'CV_Balanced_Acc': cv_res['cv_balanced_accuracy_mean'],
        'CV_BalAcc_CI_Low': cv_res.get('cv_balanced_accuracy_ci_lower', 0),
        'CV_BalAcc_CI_Upper': cv_res.get('cv_balanced_accuracy_ci_upper', 0),
        'CV_F2_BK': cv_res.get('cv_f2_bk_mean', 0),
        'Test_Recall_BK': test_results['test_recall_bk'],
        'Test_Recall_BK_CI_Low': test_results['test_recall_bk_ci_lower'],
        'Test_Recall_BK_CI_Upper': test_results['test_recall_bk_ci_upper'],
        'Test_Precision_BK': test_results['test_precision_bk'],
        'Test_F2_BK': test_results['test_f2_bk'],
        'Test_F2_BK_CI_Low': test_results['test_f2_bk_ci_lower'],
        'Test_F2_BK_CI_Upper': test_results['test_f2_bk_ci_upper'],
        'Test_Specificity': test_results.get('test_specificity', 0),
        'Test_Balanced_Acc': test_results['test_balanced_accuracy'],
        'Test_BalAcc_CI_Low': test_results['test_balanced_accuracy_ci_lower'],
        'Test_BalAcc_CI_Upper': test_results['test_balanced_accuracy_ci_upper'],
        'Test_MCC': test_results.get('test_mcc', 0),
        'Test_ROC_AUC': test_results.get('test_roc_auc', 0),
        'Test_PR_AUC': test_results.get('test_pr_auc', 0),
        'Test_TP': test_results.get('test_tp', 0),
        'Test_FN': test_results.get('test_fn', 0),
        'Test_FP': test_results.get('test_fp', 0),
        'Test_TN': test_results.get('test_tn', 0),
    }
    return row


def evaluate_dummy_baseline():
    """Run the DummyClassifier (majority-class) baseline on the C_Full
    split. This provides the honest 'always predict the most common class'
    reference that EWS claims to beat."""
    file_path = f'data/features/C_Full_{SCENARIO}.csv'
    if not os.path.exists(file_path):
        return None
    dfc = pd.read_csv(file_path)
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(
        dfc, CONFIG_PATH)
    features = get_features(SCENARIO)
    available_features = [f for f in features if f in X_train_full.columns]
    X_test = X_test_full[available_features]

    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train_full[available_features], y_train)
    results, y_pred = evaluate_test(
        dummy, X_test, y_test,
        'EWS_C_Full_DummyBaseline', 'DummyClassifier',
        threshold=None,
        output_dir='results/confusion_matrix')

    print("\n--- DUMMY BASELINE (C_Full, most-frequent) ---")
    for k in ('test_balanced_accuracy', 'test_recall_bk',
              'test_precision_bk', 'test_mcc'):
        v = results.get(k, 0)
        print(f"  {k}: {v:.4f}" if not (isinstance(v, float) and v != v)
              else f"  {k}: N/A")
    return results


def main():
    print("=" * 60)
    print("MULTI-CUTOFF EWS EXPERIMENT (S3_E, DecisionTree)")
    print("Cutoffs: W1, W2, W3, C_Full — Holdout n=18 (frozen)")
    print("=" * 60)

    cutoff_results = []
    for cutoff in CUTOFFS:
        row = evaluate_one_cutoff(cutoff)
        if row is not None:
            cutoff_results.append(row)

    if not cutoff_results:
        print("No cutoffs could be evaluated. Check data/features/ CSVs.")
        return

    df_res = pd.DataFrame(cutoff_results)
    print("\n" + "=" * 80)
    print("HASIL KOMPARASI MULTI-CUTOFF (Holdout n=18)")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    cols = ['Cutoff', 'Threshold', 'CV_Recall_BK', 'CV_Balanced_Acc',
            'Test_Recall_BK', 'Test_Precision_BK', 'Test_F2_BK',
            'Test_Specificity', 'Test_Balanced_Acc', 'Test_MCC',
            'Test_ROC_AUC', 'Test_PR_AUC',
            'Test_TP', 'Test_FN', 'Test_FP', 'Test_TN']
    print(df_res[cols].to_string(index=False))

    os.makedirs('results/final', exist_ok=True)
    df_res.to_csv('results/final/cutoff_comparison.csv', index=False)

    # Comparison figures
    plots.plot_cutoff_comparison(df_res)
    cm_files = [
        (f'results/confusion_matrix/cm_EWS_{c}_{SCENARIO}_{MODEL_NAME}.png', c)
        for c in CUTOFFS
        if os.path.exists(f'results/confusion_matrix/cm_EWS_{c}_{SCENARIO}_{MODEL_NAME}.png')
    ]
    if cm_files:
        plots.plot_cm_grid(cm_files, out_path='results/plots/cm_grid_all_cutoffs.png')

    # Consolidated JSON summary (single file with all numbers)
    dummy_results = evaluate_dummy_baseline()

    def _safe(v):
        return (float(v) if isinstance(v, (float, np.floating)) and v == v
                else v)

    summary = {
        'experiments': cutoff_results,
        'n_holdout': 18,
        'dummy_baseline_C_Full': {
            k: _safe(v) for k, v in (dummy_results or {}).items()
            if not k.endswith('_ci_lower') and not k.endswith('_ci_upper')
        },
    }
    with open('results/final/consolidated_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\nSemua figur tersimpan di results/plots/ dan results/confusion_matrix/.")
    print("Laporan JSON per-cutoff di results/final/final_model_report_<cutoff>.json")


if __name__ == "__main__":
    main()
