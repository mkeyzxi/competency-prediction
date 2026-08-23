import pandas as pd
import json
import os
from src.utils import load_config, save_model
from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters
from src.evaluation import evaluate_cv, evaluate_test

def run_all_experiments(df_featured: pd.DataFrame, config_path: str = 'configs/experiment_config.yaml'):
    config = load_config(config_path)
    scenarios = config['experiments']
    model_names = config['models']
    
    # We do split once so that test set is identical for all experiments
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured, config_path)
    
    results_list = []
    
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Test set predictions tracking
    test_predictions = pd.DataFrame({
        'row_id': X_test_full.index,
        'true_label': y_test.values
    })
    
    for scenario in scenarios:
        features = get_features(scenario)
        if not features:
            continue
            
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
    
    return df_results
