"""
Eksperimen Utama: Fokus pada populasi P2 (Strict Eligible).
Model: Dummy, LogisticRegression, DecisionTree, RandomForest.
Skenario: S1, S2, S3, S4, S5.
Tambahan: Top-K Feature Selection untuk RF pada S5.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
from src.experiments import run_all_experiments
from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_cv, evaluate_test
from src.utils import load_config, save_model

def run_topk_feature_selection(df_featured, pop_name, config_path='configs/experiment_config.yaml'):
    """
    Runs Top-K feature selection experiment for RandomForest on S5.
    1. Train RF on full S5 features.
    2. Extract feature_importances_.
    3. Re-train on Top-10, 15, 20, 25 features.
    4. Compare CV and test performance.
    """
    config = load_config(config_path)
    features_s5 = get_features('S5')
    
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured, config_path)
    
    available_features = [f for f in features_s5 if f in X_train_full.columns]
    X_train = X_train_full[available_features]
    X_test = X_test_full[available_features]
    
    print(f"\n[{pop_name}] Top-K Feature Selection on S5 ({len(available_features)} features)...")
    
    # Step 1: Train full RF to get importances
    base_model = get_model('RandomForest', config_path)
    full_model, full_params = tune_hyperparameters(base_model, 'RandomForest', X_train, y_train, config_path)
    
    # Extract feature importances from the pipeline
    clf = full_model.named_steps['model']
    imputer = full_model.named_steps['imputer']
    
    # Get transformed feature names (imputer may add indicator columns)
    X_train_transformed = imputer.transform(X_train)
    if hasattr(imputer, 'get_feature_names_out'):
        transformed_names = list(imputer.get_feature_names_out(available_features))
    else:
        transformed_names = available_features
    
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': transformed_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    os.makedirs('results/reports', exist_ok=True)
    importance_df.to_csv(f'results/reports/feature_importances_{pop_name}_S5_RF.csv', index=False)
    print(f"  Feature importances saved. Top 10:")
    print(importance_df.head(10).to_string(index=False))
    
    # Step 2: Top-K experiments
    top_k_values = [10, 15, 20, 25, len(available_features)]
    topk_results = []
    
    for k in top_k_values:
        # Get top-k original features (exclude indicator columns for selection)
        original_features_imp = importance_df[~importance_df['feature'].str.startswith('missingindicator_')]
        top_features = original_features_imp.head(k)['feature'].tolist()
        
        # Filter to only features that exist in the original X_train columns
        top_features_available = [f for f in top_features if f in X_train.columns]
        if not top_features_available:
            continue
        
        X_train_k = X_train[top_features_available]
        X_test_k = X_test[top_features_available]
        
        print(f"\n  Top-{k} features ({len(top_features_available)} available)...")
        
        base_k = get_model('RandomForest', config_path)
        best_k, best_params_k = tune_hyperparameters(base_k, 'RandomForest', X_train_k, y_train, config_path)
        
        cv_results = evaluate_cv(best_k, X_train_k, y_train, config_path)
        
        thresh_metrics, best_threshold, _ = evaluate_thresholds(best_k, X_train_k, y_train)
        
        test_results, y_pred = evaluate_test(
            best_k, X_test_k, y_test,
            f'{pop_name}_S5_Top{k}', 'RandomForest',
            threshold=best_threshold
        )
        
        save_model(best_k, f'models/{pop_name}_S5_Top{k}_RandomForest.pkl')
        
        row = {
            'population': pop_name,
            'scenario': f'S5_Top{k}',
            'model': 'RandomForest',
            'n_features': len(top_features_available),
            'best_threshold': best_threshold,
            'features_used': ', '.join(top_features_available),
            **cv_results,
            **test_results,
            'best_params': json.dumps(best_params_k)
        }
        topk_results.append(row)
        
        print(f"  Top-{k}: CV Acc={cv_results['cv_accuracy_mean']:.4f}, "
              f"Test Acc={test_results['test_accuracy']:.4f}, "
              f"Test BalAcc={test_results['test_balanced_accuracy']:.4f}")
    
    return pd.DataFrame(topk_results)


def main():
    print("=" * 60)
    print("EKSPERIMEN UTAMA — Populasi P2 (Strict Eligible)")
    print("=" * 60)
    
    # Use P2 as the gold standard population
    primary_pop = 'P2'
    file_path = f'data/processed/featured_{primary_pop}_full.csv'
    
    if not os.path.exists(file_path):
        print(f"Featured dataset {primary_pop} not found. Please run build_features.py first.")
        return
    
    # Clean old report files
    old_files = [
        'results/reports/class_distribution.csv',
        'results/reports/error_analysis.csv',
        'results/reports/threshold_analysis.csv',
        'results/reports/baseline_comparison.csv',
        'results/reports/repeated_cv_results.csv',
        'results/reports/topk_results.csv',
    ]
    for f in old_files:
        if os.path.exists(f):
            os.remove(f)
    
    df = pd.read_csv(file_path)
    
    # --- Phase 1: Standard S1-S5 experiments ---
    print(f"\n--- Phase 1: Standard Experiments S1-S5 ---")
    results = run_all_experiments(df, primary_pop)
    
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    results.to_csv('results/reports/repeated_cv_results.csv', index=False)
    results.to_csv('results/metrics/model_comparison.csv', index=False)
    
    # Baseline comparison
    baseline_df = results[results['model'].isin(['Dummy', 'LogisticRegression'])]
    baseline_df.to_csv('results/reports/baseline_comparison.csv', index=False)
    
    # --- Phase 2: Top-K Feature Selection ---
    print(f"\n--- Phase 2: Top-K Feature Selection (RF on S5) ---")
    topk_results = run_topk_feature_selection(df, primary_pop)
    topk_results.to_csv('results/reports/topk_results.csv', index=False)
    
    # Combine all results
    all_results = pd.concat([results, topk_results], ignore_index=True)
    all_results.to_csv('results/metrics/model_comparison.csv', index=False)
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY — Main Experiment Results")
    print("=" * 60)
    
    metrics = [
        'population', 'scenario', 'model', 'best_threshold',
        'cv_accuracy_mean', 'test_accuracy',
        'cv_balanced_accuracy_mean', 'test_balanced_accuracy',
        'cv_f1_mean', 'test_f1',
        'test_recall_belum_kompeten'
    ]
    available_metrics = [m for m in metrics if m in all_results.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(all_results[available_metrics].to_string(index=False))

if __name__ == "__main__":
    main()
