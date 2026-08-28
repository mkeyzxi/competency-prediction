import pandas as pd
import json
import os
from src.utils import load_config, save_model
from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_cv, evaluate_test

def run_error_analysis(test_predictions, df_featured, pop_name):
    os.makedirs('results/reports', exist_ok=True)
    errors = []
    for col in test_predictions.columns:
        if col.endswith('_pred'):
            model_scenario = col.replace('_pred', '')
            
            fp_mask = (test_predictions['true_label'] == 0) & (test_predictions[col] == 1)
            fn_mask = (test_predictions['true_label'] == 1) & (test_predictions[col] == 0)
            
            for idx, row in test_predictions[fp_mask].iterrows():
                prob_col = f'{model_scenario}_prob_kompeten'
                prob = row[prob_col] if prob_col in row else None
                errors.append({
                    'Population': pop_name,
                    'Model_Scenario': model_scenario,
                    'NIM': df_featured.loc[row['row_id'], 'NIM'],
                    'Class': df_featured.loc[row['row_id'], 'Class'],
                    'True_Label': row['true_label'],
                    'Predicted_Label': row[col],
                    'Predicted_Probability': prob,
                    'Error_Type': 'False Positive'
                })
                
            for idx, row in test_predictions[fn_mask].iterrows():
                prob_col = f'{model_scenario}_prob_kompeten'
                prob = row[prob_col] if prob_col in row else None
                errors.append({
                    'Population': pop_name,
                    'Model_Scenario': model_scenario,
                    'NIM': df_featured.loc[row['row_id'], 'NIM'],
                    'Class': df_featured.loc[row['row_id'], 'Class'],
                    'True_Label': row['true_label'],
                    'Predicted_Label': row[col],
                    'Predicted_Probability': prob,
                    'Error_Type': 'False Negative'
                })
    
    if errors:
        df_errors = pd.DataFrame(errors)
        error_file = 'results/reports/error_analysis.csv'
        if os.path.exists(error_file):
            df_existing = pd.read_csv(error_file)
            df_combined = pd.concat([df_existing, df_errors], ignore_index=True)
            df_combined.to_csv(error_file, index=False)
        else:
            df_errors.to_csv(error_file, index=False)

def run_all_experiments(df_featured: pd.DataFrame, pop_name: str, config_path: str = 'configs/experiment_config.yaml'):
    config = load_config(config_path)
    # The models to test are defined here. Let's explicitly test Dummy, DT, RF
    model_names = ['Dummy', 'DecisionTree', 'RandomForest']
    scenarios = ['S1', 'S2', 'S3', 'S4', 'S5']
    
    forbidden_columns = ['final', 'nilai_akhir', 'predikat', 'flowchart', 'kodingan', 'final_kelompok', 'nim', 'nama']
    
    # Stratified split 
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured, config_path)
    
    results_list = []
    threshold_results_list = []
    
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Class distribution audit
    os.makedirs('results/reports', exist_ok=True)
    dist_file = 'results/reports/class_distribution.csv'
    dist_data = pd.DataFrame({
        'Population': pop_name,
        'Class': df_featured['Class'],
        'Label': df_featured['Competency_Label']
    }).value_counts().reset_index(name='Count')
    if os.path.exists(dist_file):
        pd.concat([pd.read_csv(dist_file), dist_data], ignore_index=True).to_csv(dist_file, index=False)
    else:
        dist_data.to_csv(dist_file, index=False)
        
    test_predictions = pd.DataFrame({
        'row_id': X_test_full.index,
        'true_label': y_test.values,
        'Class': df_featured.loc[X_test_full.index, 'Class'].values
    })
    
    for scenario in scenarios:
        features = get_features(scenario)
        if not features:
            continue
            
        # Select only features for this scenario
        available_features = [f for f in features if f in X_train_full.columns]
        X_train = X_train_full[available_features]
        X_test = X_test_full[available_features]
        
        # Save feature correlation matrix
        corr = X_train.corr(numeric_only=True)
        corr.to_csv(f'results/reports/feature_correlation_{pop_name}_{scenario}.csv')
        
        for model_name in model_names:
            print(f"[{pop_name}] Running {scenario} with {model_name}...")
            
            base_model = get_model(model_name, config_path)
            
            # 1. Nested CV Evaluation (evaluates the model selection process)
            cv_results = evaluate_cv(base_model, model_name, X_train, y_train, config_path)
            
            # 2. Final tuning on full training set
            best_model, best_params = tune_hyperparameters(base_model, model_name, X_train, y_train, config_path)
            
            save_model(best_model, f'models/{pop_name}_{scenario}_{model_name}.pkl')
            
            # Threshold Optimization (on training set predictions to avoid data leakage)
            thresh_metrics, best_threshold, best_thresh_bal_acc = evaluate_thresholds(best_model, X_train, y_train)
            for tm in thresh_metrics:
                tm['Population'] = pop_name
                tm['Scenario'] = scenario
                tm['Model'] = model_name
                threshold_results_list.append(tm)
            
            # Test Evaluation — use optimized threshold
            test_results, y_pred = evaluate_test(
                best_model, X_test, y_test, 
                f'{pop_name}_{scenario}', model_name,
                threshold=best_threshold
            )
            
            if hasattr(best_model, "predict_proba"):
                try:
                    y_prob = best_model.predict_proba(X_test)
                    test_predictions[f'{scenario}_{model_name}_prob_kompeten'] = y_prob[:, 1]
                    test_predictions[f'{scenario}_{model_name}_prob_belum_kompeten'] = y_prob[:, 0]
                except Exception:
                    pass
            
            test_predictions[f'{scenario}_{model_name}_pred'] = y_pred
            
            row = {
                'population': pop_name,
                'scenario': scenario,
                'model': model_name,
                'best_threshold': best_threshold,
                **cv_results,
                **test_results,
                'best_params': json.dumps(best_params)
            }
            results_list.append(row)
            
    df_results = pd.DataFrame(results_list)
    
    # Save Error Analysis
    run_error_analysis(test_predictions, df_featured, pop_name)
    
    # Save Threshold Analysis
    if threshold_results_list:
        df_thresh = pd.DataFrame(threshold_results_list)
        thresh_file = 'results/reports/threshold_analysis.csv'
        if os.path.exists(thresh_file):
            pd.concat([pd.read_csv(thresh_file), df_thresh], ignore_index=True).to_csv(thresh_file, index=False)
        else:
            df_thresh.to_csv(thresh_file, index=False)
            
    test_predictions.to_csv(f'results/predictions/final_predictions_{pop_name}.csv', index=False)
    
    return df_results
