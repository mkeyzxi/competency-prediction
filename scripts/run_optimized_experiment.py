"""
Eksperimen Optimasi Terkontrol — EWS Competency Prediction

PROTOKOL KETAT:
1. Model selection HANYA dari Outer CV metrics (training data)
2. Holdout 20% TIDAK PERNAH disentuh untuk memilih model
3. Setelah model terpilih → freeze → fit ulang di seluruh train → 1× evaluasi holdout
4. SMOTE vs ClassWeight vs None diuji sebagai eksperimen terpisah
5. Threshold dipilih di inner CV (tidak melihat holdout)

Matrix eksperimen:
  Models:    Dummy, DecisionTree, RandomForest, GradientBoosting
  Balancing: None, SMOTE, ClassWeight
  Scenarios: S3, S3_A, S3_B, S3_C, S3_D, S3_E, S3_EWS
"""
import sys
import os
import json
import warnings
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model, get_param_grid
from src.tuning import tune_hyperparameters, evaluate_thresholds, build_search_estimator
from src.evaluation import evaluate_cv, evaluate_test
from src.utils import load_config, save_model

warnings.filterwarnings('ignore')


def run_cv_experiment_matrix(df_featured, pop_name,
                             config_path='configs/experiment_config.yaml'):
    """
    Phase 1: Run Nested CV for all combinations.
    Model selection is based ONLY on these CV results.
    The holdout set is NOT used here.
    """
    config = load_config(config_path)

    model_names = ['Dummy', 'DecisionTree', 'RandomForest']
    balancing_strategies = ['None']
    scenarios = ['S3', 'S3_A', 'S3_B', 'S3_C', 'S3_D', 'S3_E', 'S3_EWS']

    # Split ONCE — holdout is frozen
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(
        df_featured, config_path
    )

    print(f"Train: {len(y_train)} samples (BK={sum(y_train==0)}, K={sum(y_train==1)})")
    print(f"Test:  {len(y_test)} samples (BK={sum(y_test==0)}, K={sum(y_test==1)})")
    print(f"Test set is FROZEN. Not used for model selection.\n")

    results_list = []
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)

    total = len(scenarios) * len(model_names) * len(balancing_strategies)
    counter = 0

    for scenario in scenarios:
        features = get_features(scenario)
        if not features:
            continue

        available_features = [f for f in features if f in X_train_full.columns]
        X_train = X_train_full[available_features]

        for model_name in model_names:
            for balancing in balancing_strategies:
                counter += 1

                # Skip nonsensical combinations
                if model_name == 'Dummy' and balancing != 'None':
                    continue
                if model_name == 'GradientBoosting' and balancing == 'ClassWeight':
                    # GradientBoosting doesn't support class_weight natively
                    continue

                tag = f"[{counter}/{total}] {scenario} | {model_name} | {balancing}"
                print(f"{tag}...")

                base_model = get_model(model_name, balancing, config_path)

                # --- Nested CV Evaluation (this is how we SELECT the model) ---
                cv_results = evaluate_cv(
                    base_model, model_name, X_train, y_train, config_path
                )

                # --- Threshold optimization (inside training set only) ---
                best_threshold = 0.50
                if model_name != 'Dummy':
                    tuned_model, tuned_params = tune_hyperparameters(
                        base_model, model_name, X_train, y_train,
                        config_path, balancing=balancing
                    )
                    thresh_results, best_threshold, _ = evaluate_thresholds(
                        tuned_model, X_train, y_train, optimize_for='f2_bk'
                    )
                else:
                    tuned_params = {}

                row = {
                    'population': pop_name,
                    'scenario': scenario,
                    'model': model_name,
                    'balancing': balancing,
                    'n_features': len(available_features),
                    'best_threshold': best_threshold,
                    **cv_results,
                    'best_params': json.dumps(tuned_params) if tuned_params else '{}',
                }
                results_list.append(row)

                # Print key CV metrics
                rec_bk = cv_results.get('cv_recall_bk_mean', 0)
                bal_acc = cv_results.get('cv_balanced_accuracy_mean', 0)
                f2 = cv_results.get('cv_f2_bk_mean', 0)
                print(f"  CV Recall BK={rec_bk:.3f}, BalAcc={bal_acc:.3f}, "
                      f"F2_BK={f2:.3f}, Threshold={best_threshold}")

    df_cv = pd.DataFrame(results_list)
    df_cv.to_csv('results/metrics/cv_experiment_matrix.csv', index=False)
    print(f"\n{'='*60}")
    print(f"Phase 1 complete. {len(df_cv)} experiments saved.")
    print(f"{'='*60}\n")

    return df_cv, X_train_full, X_test_full, y_train, y_test


def select_best_model(df_cv):
    """
    Phase 2: Select the best model based ONLY on Outer CV metrics.

    Selection criteria (in priority order):
    1. cv_recall_bk_mean     (primary — catch at-risk students)
    2. cv_balanced_accuracy_mean (secondary — overall balance)
    3. cv_pr_auc_mean        (tertiary — robust to class imbalance)
    4. cv_recall_bk_std      (lower is better — stability)

    Holdout results are NOT considered here.
    """
    # Exclude Dummy
    candidates = df_cv[df_cv['model'] != 'Dummy'].copy()

    # Sort by priority
    candidates = candidates.sort_values(
        by=[
            'cv_recall_bk_mean',
            'cv_balanced_accuracy_mean',
            'cv_pr_auc_mean',
        ],
        ascending=[False, False, False]
    )

    best = candidates.iloc[0]
    print("=" * 60)
    print("MODEL SELECTION (from Outer CV only — holdout NOT used)")
    print("=" * 60)
    print(f"Best: {best['model']} | {best['scenario']} | {best['balancing']}")
    print(f"  CV Recall BK:    {best['cv_recall_bk_mean']:.4f} ± {best['cv_recall_bk_std']:.4f}")
    print(f"  CV Balanced Acc: {best['cv_balanced_accuracy_mean']:.4f} ± {best['cv_balanced_accuracy_std']:.4f}")
    print(f"  CV F2 BK:        {best['cv_f2_bk_mean']:.4f} ± {best['cv_f2_bk_std']:.4f}")
    print(f"  CV PR-AUC:       {best.get('cv_pr_auc_mean', 'N/A')}")
    print(f"  Threshold:       {best['best_threshold']}")
    print()

    # Also show top 5 for comparison
    print("Top 5 candidates:")
    top5_cols = ['scenario', 'model', 'balancing',
                 'cv_recall_bk_mean', 'cv_balanced_accuracy_mean',
                 'cv_f2_bk_mean', 'best_threshold']
    available_cols = [c for c in top5_cols if c in candidates.columns]
    print(candidates.head(5)[available_cols].to_string(index=False))
    print()

    return best


def final_holdout_evaluation(best_config, df_featured,
                             X_train_full, X_test_full, y_train, y_test,
                             config_path='configs/experiment_config.yaml'):
    """
    Phase 3: FINAL holdout evaluation.
    Called exactly ONCE after model is frozen.

    Steps:
    1. Re-fit the selected model on the ENTIRE training set
    2. Apply frozen threshold
    3. Evaluate on holdout
    """
    scenario = best_config['scenario']
    model_name = best_config['model']
    balancing = best_config['balancing']
    threshold = best_config['best_threshold']

    features = get_features(scenario)
    available_features = [f for f in features if f in X_train_full.columns]
    X_train = X_train_full[available_features]
    X_test = X_test_full[available_features]

    print("=" * 60)
    print("FINAL HOLDOUT EVALUATION (called exactly once)")
    print("=" * 60)
    print(f"Model: {model_name} | Scenario: {scenario} | Balancing: {balancing}")
    print(f"Threshold: {threshold}")
    print(f"Features ({len(available_features)}): {available_features}")
    print()

    # Re-fit on entire training set
    model = get_model(model_name, balancing, config_path)
    final_model, final_params = tune_hyperparameters(
        model, model_name, X_train, y_train, config_path, balancing=balancing
    )

    # Save the frozen model
    os.makedirs('models', exist_ok=True)
    save_model(final_model, f'models/FINAL_{model_name}_{scenario}_{balancing}.pkl')

    # 1× holdout evaluation
    test_results, y_pred = evaluate_test(
        final_model, X_test, y_test,
        f'FINAL_{scenario}', model_name,
        threshold=threshold,
        output_dir='results/final'
    )

    print("\n--- HOLDOUT RESULTS ---")
    
    # Print Confusion Matrix Breakdown First
    print(f"  Confusion Matrix: TP={test_results.get('test_tp', 0)}, FN={test_results.get('test_fn', 0)}, FP={test_results.get('test_fp', 0)}, TN={test_results.get('test_tn', 0)}")
    print(f"  Specificity:      {test_results.get('test_specificity', 0.0):.4f}")
    
    for k, v in sorted(test_results.items()):
        if 'ci' in k or 'tp' in k or 'tn' in k or 'fp' in k or 'fn' in k or k == 'test_specificity':
            continue # We print CIs separately
        if isinstance(v, float):
            ci_str = ""
            if f"{k}_ci_lower" in test_results:
                ci_str = f" (95% CI: [{test_results[f'{k}_ci_lower']:.4f}, {test_results[f'{k}_ci_upper']:.4f}])"
            print(f"  {k}: {v:.4f}{ci_str}")
        else:
            print(f"  {k}: {v}")

    # Save final results
    os.makedirs('results/final', exist_ok=True)
    final_report = {
        'model': model_name,
        'scenario': scenario,
        'balancing': balancing,
        'threshold': threshold,
        'n_features': len(available_features),
        'features': available_features,
        'cv_metrics': {
            'recall_bk': float(best_config['cv_recall_bk_mean']),
            'balanced_accuracy': float(best_config['cv_balanced_accuracy_mean']),
            'f2_bk': float(best_config['cv_f2_bk_mean']),
        },
        'holdout_metrics': {k: float(v) if isinstance(v, (float, np.floating)) else v
                           for k, v in test_results.items()},
        'best_params': json.loads(best_config['best_params']),
    }

    with open('results/final/final_model_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)

    pd.DataFrame([test_results]).to_csv(
        'results/final/holdout_results.csv', index=False
    )

    return final_model, test_results, y_pred


def main():
    print("=" * 60)
    print("MULTI-CUTOFF EWS EXPERIMENT")
    print("Protokol: Evaluasi S3_E (DecisionTree) pada W1, W2, W3, C_Full")
    print("=" * 60)
    print()

    pop = 'n89'
    cutoffs = ['W1', 'W2', 'W3', 'C_Full']
    scenario = 'S3_E'
    model_name = 'DecisionTree'
    balancing = 'None'
    
    cutoff_results = []
    
    for cutoff in cutoffs:
        file_path = f'data/features/{cutoff}_{scenario}.csv'
        if not os.path.exists(file_path):
            print(f"Dataset {file_path} not found. Skipping...")
            continue
            
        df = pd.read_csv(file_path)
        print(f"\n--- Menguji Cutoff: {cutoff} ---")
        
        # We run the pipeline for this cutoff
        # We manually construct best_config to force selection of S3_E and DecisionTree
        X_train_full, X_test_full, y_train, y_test = get_train_test_split(df)
        
        features = get_features(scenario)
        available_features = [f for f in features if f in X_train_full.columns]
        X_train = X_train_full[available_features]
        X_test = X_test_full[available_features]
        
        # Inner CV for Threshold
        base_model = get_model(model_name, balancing)
        tuned_model, tuned_params = tune_hyperparameters(
            base_model, model_name, X_train, y_train, balancing=balancing
        )
        _, best_threshold, _ = evaluate_thresholds(
            tuned_model, X_train, y_train, optimize_for='f2_bk'
        )
        
        # Outer CV for Metrics (just to log them)
        cv_res = evaluate_cv(base_model, model_name, X_train, y_train)
        
        best_config = {
            'scenario': scenario,
            'model': model_name,
            'balancing': balancing,
            'best_threshold': best_threshold,
            'cv_recall_bk_mean': cv_res['cv_recall_bk_mean'],
            'cv_balanced_accuracy_mean': cv_res['cv_balanced_accuracy_mean'],
            'cv_f2_bk_mean': cv_res['cv_f2_bk_mean'],
            'best_params': json.dumps(tuned_params) if tuned_params else '{}'
        }
        
        final_model, test_results, y_pred = final_holdout_evaluation(
            best_config, df, X_train_full, X_test_full, y_train, y_test
        )
        
        cutoff_results.append({
            'Cutoff': cutoff,
            'Threshold': best_threshold,
            'CV_Recall_BK': cv_res['cv_recall_bk_mean'],
            'CV_Balanced_Acc': cv_res['cv_balanced_accuracy_mean'],
            'Test_Recall_BK': test_results['test_recall_bk'],
            'Test_Specificity': test_results.get('test_specificity', 0),
            'Test_Balanced_Acc': test_results['test_balanced_accuracy'],
            'Test_TP': test_results.get('test_tp', 0),
            'Test_FN': test_results.get('test_fn', 0),
            'Test_FP': test_results.get('test_fp', 0),
            'Test_TN': test_results.get('test_tn', 0)
        })
        
    df_res = pd.DataFrame(cutoff_results)
    print("\n" + "=" * 60)
    print("HASIL KOMPARASI MULTI-CUTOFF")
    print("=" * 60)
    print(df_res.to_string(index=False))
    
    os.makedirs('results/final', exist_ok=True)
    df_res.to_csv('results/final/cutoff_comparison.csv', index=False)


if __name__ == "__main__":
    main()
