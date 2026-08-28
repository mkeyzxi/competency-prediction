import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_cv, evaluate_test

def main():
    print("=" * 60)
    print("BASELINE SCENARIO EXPERIMENTS (S1 - S5)")
    print("Population: n89 (Threshold 83)")
    print("=" * 60)
    
    config_path = 'configs/experiment_config.yaml'
    scenarios = ['S1', 'S2', 'S3', 'S4', 'S5']
    models_to_test = ['DecisionTree', 'RandomForest']
    
    results = []
    
    for scenario in scenarios:
        file_path = f'data/features/C_Full_{scenario}.csv'
        if not os.path.exists(file_path):
            print(f"Skipping {scenario}, file not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        X_train_full, X_test_full, y_train, y_test = get_train_test_split(df, config_path)
        
        # Ensure we only use features defined in the registry
        features = get_features(scenario)
        available_features = [f for f in features if f in X_train_full.columns]
        
        X_train = X_train_full[available_features]
        X_test = X_test_full[available_features]
        
        for model_name in models_to_test:
            print(f"[{scenario}] Training {model_name}...")
            
            # Setup Model Pipeline (without SMOTE)
            base_model = get_model(model_name, balancing='None', config_path=config_path)
            
            # 1. Evaluate Nested CV
            cv_metrics = evaluate_cv(base_model, model_name, X_train, y_train, config_path)
            
            # 2. Final tuning on entire Train set
            final_model, best_params = tune_hyperparameters(base_model, model_name, X_train, y_train, config_path)
            
            # 3. Find optimal threshold from training data
            thresh_metrics, best_thresh, _ = evaluate_thresholds(final_model, X_train, y_train)
            
            # 4. Evaluate on Test set
            test_results, y_pred = evaluate_test(
                final_model, X_test, y_test, 
                scenario_name=f'Baseline_{scenario}',
                model_name=model_name,
                threshold=best_thresh
            )
            
            row = {
                'Scenario': scenario,
                'Model': model_name,
                'CV_BalAcc': cv_metrics['cv_balanced_accuracy_mean'],
                'CV_Recall_BK': cv_metrics['cv_recall_bk_mean'],
                'Test_BalAcc': test_results['test_balanced_accuracy'],
                'Test_Recall_BK': test_results['test_recall_bk'],
                'Test_PR_AUC': test_results.get('test_pr_auc', 0.0),
                'Threshold': best_thresh,
                'Features_Count': len(available_features)
            }
            results.append(row)
            
    # Print Summary Table
    df_results = pd.DataFrame(results)
    
    # Sort by Test Recall BK then Test BalAcc
    df_results = df_results.sort_values(by=['Test_Recall_BK', 'Test_BalAcc'], ascending=[False, False])
    
    print("\n" + "=" * 90)
    print("FINAL LEADERBOARD (S1 - S5)")
    print("=" * 90)
    
    # Format the table neatly
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
