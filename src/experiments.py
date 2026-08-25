import pandas as pd
import json
import os
from src.utils import load_config, save_model
from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters
from src.evaluation import evaluate_cv, evaluate_test

def run_context_analysis(test_predictions, df_results):
    os.makedirs('results/reports', exist_ok=True)
    
    # We want to calculate accuracy, precision, recall for each Class
    report = []
    for col in test_predictions.columns:
        if col.endswith('_pred'):
            scenario_model = col.replace('_pred', '')
            
            for cls in ['A', 'B', 'C', 'D', 'E']:
                mask = test_predictions['Class'] == cls
                if mask.sum() == 0:
                    continue
                
                subset = test_predictions[mask]
                y_true = subset['true_label']
                y_pred = subset[col]
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                report.append({
                    'model_scenario': scenario_model,
                    'class': cls,
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
                    'support': len(y_true)
                })
                
    df_report = pd.DataFrame(report)
    df_report.to_csv('results/reports/context_analysis.csv', index=False)
    print("Context analysis saved to results/reports/context_analysis.csv")

def run_all_experiments(df_featured: pd.DataFrame, config_path: str = 'configs/experiment_config.yaml'):
    config = load_config(config_path)
    scenarios = config['experiments']
    model_names = config['models']
    
    forbidden_columns = ['final', 'nilai_akhir', 'predikat', 'flowchart', 'kodingan', 'final_kelompok', 'nim', 'nama']
    
    # We do split once so that test set is identical for all experiments
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured, config_path)
    
    results_list = []
    
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Test set predictions tracking
    test_predictions = pd.DataFrame({
        'row_id': X_test_full.index,
        'true_label': y_test.values,
        'Class': df_featured.loc[X_test_full.index, 'Class'].values
    })
    
    for scenario in scenarios:
        features = get_features(scenario)
        if not features:
            continue
            
        # Anti-leakage audit for X
        for col in forbidden_columns:
            if col == 'final':
                assert not any(c.lower() == 'final' or c.lower() == 'final_individu' for c in features), "Leakage detected: Final is in X."
            else:
                assert not any(col in str(c).lower() for c in features), f"Leakage detected: {col} is in the X features for {scenario}."
        
        # Select only features for this scenario
        X_train = X_train_full[features]
        X_test = X_test_full[features]
        
        for model_name in model_names:
            print(f"Running {scenario} with {model_name}...")
            
            # Get base model
            base_model = get_model(model_name, config_path)
            
            # Tune
            best_model, best_params = tune_hyperparameters(base_model, model_name, X_train, y_train, config_path)
            
            # Save model
            save_model(best_model, f'models/{scenario}_{model_name}.pkl')
            
            # Evaluate CV on best model
            cv_results = evaluate_cv(best_model, X_train, y_train, config_path)
            
            # Evaluate Test
            test_results, y_pred = evaluate_test(best_model, X_test, y_test, scenario, model_name)
            
            # Get probabilities if possible
            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(X_test)
                test_predictions[f'{scenario}_{model_name}_prob_kompeten'] = y_prob[:, 1]
                test_predictions[f'{scenario}_{model_name}_prob_belum_kompeten'] = y_prob[:, 0]
            
            test_predictions[f'{scenario}_{model_name}_pred'] = y_pred
            test_predictions[f'{scenario}_{model_name}_correct'] = (y_pred == y_test.values).astype(int)
            
            # Combine results
            row = {
                'scenario': scenario,
                'model': model_name,
                **cv_results,
                **test_results,
                'best_params': json.dumps(best_params)
            }
            results_list.append(row)
            
    # Save comparison table
    df_results = pd.DataFrame(results_list)
    df_results.to_csv('results/metrics/model_comparison.csv', index=False)
    
    # Save final predictions
    test_predictions.to_csv('results/predictions/final_predictions.csv', index=False)
    
    # Run context analysis
    run_context_analysis(test_predictions, df_results)
    
    return df_results
