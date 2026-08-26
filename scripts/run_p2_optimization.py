import sys
import os
import pandas as pd
import numpy as np
import json
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.preprocessing import preprocess_data
from src.feature_engineering import compute_features
from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_param_grid, get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_cv, evaluate_test
from src.utils import load_config, save_model
from src.feature_selection import DynamicTopKSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def generate_error_analysis(model, X_test, y_test, df_test_raw, y_pred, y_prob, output_file):
    """
    Generates a detailed CSV with False Positives, False Negatives, etc.
    """
    df_analysis = df_test_raw.copy()
    df_analysis['Actual'] = y_test.values
    df_analysis['Predicted'] = y_pred
    if y_prob is not None:
        df_analysis['Probability_Kompeten'] = y_prob
    
    conditions = [
        (df_analysis['Actual'] == 1) & (df_analysis['Predicted'] == 1),
        (df_analysis['Actual'] == 0) & (df_analysis['Predicted'] == 0),
        (df_analysis['Actual'] == 0) & (df_analysis['Predicted'] == 1),
        (df_analysis['Actual'] == 1) & (df_analysis['Predicted'] == 0),
    ]
    choices = ['TP', 'TN', 'FP', 'FN']
    df_analysis['Prediction_Type'] = np.select(conditions, choices, default='Unknown')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_analysis.to_csv(output_file, index=False)
    
    print(f"  Error analysis saved to {output_file}")
    print("  Breakdown:")
    print(df_analysis['Prediction_Type'].value_counts())

def run_experiment(df_featured, pop_name, cutoff, scenario, model_name, use_selector=False, config_path='configs/experiment_config.yaml'):
    print(f"\n--- Running {model_name} on {cutoff} {scenario} (Feature Selection: {use_selector}) ---")
    
    features = get_features(scenario)
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured, config_path)
    
    available_features = [f for f in features if f in X_train_full.columns]
    X_train = X_train_full[available_features]
    X_test = X_test_full[available_features]
    
    print(f"  Features available: {len(available_features)}")
    
    # Build Pipeline
    imputer = SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)
    
    if model_name == 'DecisionTree':
        clf = DecisionTreeClassifier(random_state=42)
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model {model_name}")
        
    steps = [
        ('imputer', imputer),
        ('smote', SMOTE(random_state=42))
    ]
    if use_selector:
        steps.append(('selector', DynamicTopKSelector(random_state=42)))
        
    steps.append(('model', clf))
    pipeline = Pipeline(steps)
    
    # Configure Param Grid
    param_grid = get_param_grid(model_name)
    if use_selector:
        param_grid['selector__k'] = [5, 8, 10, 12, 15, 20, 25, 'all']
        param_grid['selector__importance_type'] = ['gini', 'permutation']
        
    # Tune
    best_model, best_params = tune_hyperparameters(pipeline, model_name, X_train, y_train, config_path, custom_param_grid=param_grid)
    print(f"  Best params: {best_params}")
    
    # Evaluate CV
    cv_results = evaluate_cv(best_model, X_train, y_train, config_path)
    print(f"  CV Acc: {cv_results['cv_accuracy_mean']:.4f}, BalAcc: {cv_results['cv_balanced_accuracy_mean']:.4f}")
    
    # Evaluate Thresholds (inside CV on training set)
    thresh_metrics, best_threshold, best_thresh_val = evaluate_thresholds(best_model, X_train, y_train)
    print(f"  Best Threshold: {best_threshold} (BalAcc: {best_thresh_val:.4f})")
    
    # Evaluate Test
    test_results, y_pred = evaluate_test(
        best_model, X_test, y_test,
        f'{pop_name}_{scenario}_{model_name}', model_name,
        threshold=best_threshold
    )
    print(f"  Test Acc: {test_results['test_accuracy']:.4f}, BalAcc: {test_results['test_balanced_accuracy']:.4f}")
    
    # Error Analysis
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None
        
    df_test_raw = df_featured.loc[X_test.index].copy()
    generate_error_analysis(
        best_model, X_test, y_test, df_test_raw, y_pred, y_prob, 
        f'results/reports/error_analysis_{pop_name}_{cutoff}_{scenario}_{model_name}.csv'
    )
    
    save_model(best_model, f'models/{pop_name}_{cutoff}_{scenario}_{model_name}_optimized.pkl')
    
    return {
        'population': pop_name,
        'scenario': scenario,
        'model': model_name,
        'feature_selection': use_selector,
        'best_threshold': best_threshold,
        **cv_results,
        **test_results,
        'best_params': json.dumps(best_params)
    }

def main():
    print("=" * 60)
    print("EARLY WARNING SYSTEM OPTIMIZATION (Strict Eligible - P2)")
    print("=" * 60)
    
    df_interim = pd.read_csv('data/interim/combined_data.csv')
    df_p0, df_p1, df_p2 = preprocess_data(df_interim)
    
    results = []
    
    cutoffs = ['C1', 'C2', 'C3', 'C4', 'C_Full']
    
    for cutoff in cutoffs:
        print(f"\n{'='*40}")
        print(f"EVALUATING TEMPORAL CUTOFF: {cutoff}")
        print(f"{'='*40}")
        
        df_p2_featured = compute_features(df_p2, cutoff_session=cutoff)
        print(f"P2 Dataset shape for {cutoff}: {df_p2_featured.shape}")
        
        # We focus on the best feature scenario S6 for the Temporal Analysis
        # Both with and without Feature Selection
        scenario = 'S6'
        
        for use_sel in [False, True]:
            for model in ['DecisionTree', 'RandomForest']:
                res = run_experiment(df_p2_featured, 'P2', cutoff, scenario, model, use_selector=use_sel)
                res['cutoff'] = cutoff
                results.append(res)
                
    df_results = pd.DataFrame(results)
    
    # Format CV metrics as Mean ± SD
    df_results['cv_accuracy'] = df_results.apply(
        lambda row: f"{row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}", axis=1
    )
    df_results['cv_balanced_accuracy'] = df_results.apply(
        lambda row: f"{row['cv_balanced_accuracy_mean']:.4f} ± {row['cv_balanced_accuracy_std']:.4f}", axis=1
    )
    
    os.makedirs('results/metrics', exist_ok=True)
    df_results.to_csv('results/metrics/temporal_optimization_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("SUMMARY — Temporal Optimization Results (Early Warning System)")
    print("=" * 60)
    
    metrics = [
        'cutoff', 'scenario', 'model', 'feature_selection', 'best_threshold',
        'cv_accuracy', 'test_accuracy',
        'cv_balanced_accuracy', 'test_balanced_accuracy'
    ]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results[metrics].to_string(index=False))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
