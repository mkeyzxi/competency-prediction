"""
Baseline scenario experiments (S1–S5, Full Semester data, C_Full).

Runs Nested CV + one-shot holdout evaluation for DecisionTree and
RandomForest on each scenario, saving:
  - results/metrics/cv_experiment_matrix.csv  (full CV leaderboard)
  - results/confusion_matrix/cm_*.png         (per scenario+model)
  - results/plots/roc_pr_*.png                (ROC & PR curves)
  - results/final/baseline_leaderboard.csv    (top-N sorted by test metrics)
"""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_cv, evaluate_test
from src import plots

warnings.filterwarnings('ignore')

CONFIG_PATH = 'configs/experiment_config.yaml'
SCENARIOS = ['S1', 'S2', 'S3', 'S4', 'S5']
MODELS = ['DecisionTree', 'RandomForest']
BALANCING = 'None'


def main():
    print("=" * 60)
    print("BASELINE SCENARIO EXPERIMENTS (S1 - S5)")
    print("Population: n89 (Batas Kompetensi = 83)")
    print("=" * 60)

    results = []
    cv_rows = []

    for scenario in SCENARIOS:
        file_path = f'data/features/C_Full_{scenario}.csv'
        if not os.path.exists(file_path):
            print(f"Skipping {scenario}, file not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        X_train_full, X_test_full, y_train, y_test = get_train_test_split(
            df, CONFIG_PATH)

        features = get_features(scenario)
        available_features = [f for f in features if f in X_train_full.columns]
        X_train = X_train_full[available_features]
        X_test = X_test_full[available_features]

        for model_name in MODELS:
            print(f"[{scenario}] Training {model_name}...")

            base_model = get_model(model_name, balancing=BALANCING,
                                   config_path=CONFIG_PATH)

            # 1. Nested CV evaluation (selection criterion)
            cv_metrics = evaluate_cv(base_model, model_name, X_train, y_train,
                                     CONFIG_PATH)

            # 2. Final tuning on entire train set
            final_model, best_params = tune_hyperparameters(
                base_model, model_name, X_train, y_train, CONFIG_PATH,
                balancing=BALANCING)

            # 3. Threshold sweep on train set only
            thresh_results, best_thresh, _ = evaluate_thresholds(
                final_model, X_train, y_train)

            # 4. One-shot holdout evaluation (confusion matrix + figures)
            test_results, y_pred = evaluate_test(
                final_model, X_test, y_test,
                scenario_name=f'Baseline_{scenario}',
                model_name=model_name,
                threshold=best_thresh,
                output_dir='results/confusion_matrix')

            # 5. Figures
            plots.plot_roc_pr(
                final_model, X_test, y_test, best_thresh,
                f'Baseline_{scenario}', model_name,
                roc_auc=test_results.get('test_roc_auc'),
                pr_auc=test_results.get('test_pr_auc'))
            plots.save_threshold_table(
                thresh_results, best_thresh,
                f'Baseline_{scenario}', model_name)

            # 6. Save model
            os.makedirs('models', exist_ok=True)
            model_path = (f'models/Baseline_{model_name}_{scenario}_'
                          f'{BALANCING}.pkl')
            from src.utils import save_model
            save_model(final_model, model_path)

            row = {
                'Scenario': scenario,
                'Model': model_name,
                'Balancing': BALANCING,
                'N_Features': len(available_features),
                'Threshold': best_thresh,
                'CV_BalAcc': cv_metrics['cv_balanced_accuracy_mean'],
                'CV_BalAcc_CI_Low': cv_metrics.get('cv_balanced_accuracy_ci_lower', 0),
                'CV_BalAcc_CI_Up': cv_metrics.get('cv_balanced_accuracy_ci_upper', 0),
                'CV_Recall_BK': cv_metrics['cv_recall_bk_mean'],
                'CV_F2_BK': cv_metrics.get('cv_f2_bk_mean', 0),
                'CV_PR_AUC': cv_metrics.get('cv_pr_auc_mean', 0),
                'CV_ROC_AUC': cv_metrics.get('cv_roc_auc_mean', 0),
                'Test_BalAcc': test_results['test_balanced_accuracy'],
                'Test_Recall_BK': test_results['test_recall_bk'],
                'Test_Precision_BK': test_results['test_precision_bk'],
                'Test_F2_BK': test_results['test_f2_bk'],
                'Test_Specificity': test_results.get('test_specificity', 0),
                'Test_MCC': test_results.get('test_mcc', 0),
                'Test_ROC_AUC': test_results.get('test_roc_auc', 0),
                'Test_PR_AUC': test_results.get('test_pr_auc', 0),
                'Test_TP': test_results.get('test_tp', 0),
                'Test_FN': test_results.get('test_fn', 0),
                'Test_FP': test_results.get('test_fp', 0),
                'Test_TN': test_results.get('test_tn', 0),
                'Best_Params': json.dumps(best_params),
            }
            results.append(row)

            # Full CV metric row (for the matrix CSV)
            cv_row = {'scenario': scenario, 'model': model_name,
                      'balancing': BALANCING,
                      'n_features': len(available_features),
                      'best_threshold': best_thresh,
                      'best_params': json.dumps(best_params)}
            cv_row.update({k: float(v) for k, v in cv_metrics.items()
                           if not k.endswith('ci_lower') and not k.endswith('ci_upper')})
            cv_rows.append(cv_row)

    # Save CSVs
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/final', exist_ok=True)
    pd.DataFrame(cv_rows).to_csv('results/metrics/cv_experiment_matrix.csv',
                                 index=False)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        by=['Test_Recall_BK', 'Test_BalAcc'], ascending=[False, False])
    df_results.to_csv('results/final/baseline_leaderboard.csv', index=False)

    print("\n" + "=" * 90)
    print("FINAL LEADERBOARD (S1 - S5, Full Semester)")
    print("=" * 90)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    cols = ['Scenario', 'Model', 'N_Features', 'Threshold',
            'CV_BalAcc', 'CV_Recall_BK', 'CV_PR_AUC',
            'Test_BalAcc', 'Test_Recall_BK', 'Test_Specificity',
            'Test_MCC', 'Test_PR_AUC']
    print(df_results[cols].to_string(index=False))

    print("\nFigur tersimpan di results/plots/ dan results/confusion_matrix/.")


if __name__ == "__main__":
    main()
