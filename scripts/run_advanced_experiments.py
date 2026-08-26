import pandas as pd
import os
import json
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_data
from src.feature_engineering import compute_features
from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters
from src.evaluation import evaluate_cv, evaluate_test
from src.experiments import run_all_experiments
from src.utils import load_config, save_model

def run_experiment_1():
    print("="*50)
    print("EXPERIMENT 1: Model Comparison (S1/S2/S3 x DT/RF)")
    print("="*50)
    df_interim = pd.read_csv('data/interim/combined_data.csv')
    df_eligible = preprocess_data(df_interim)
    df_featured = compute_features(df_eligible, cutoff_session='PreFinal')
    
    # run_all_experiments handles S1/S2/S3 and DT/RF and calls evaluate_cv which we updated to RepeatedStratifiedKFold
    df_results = run_all_experiments(df_featured)
    print("Experiment 1 completed. Results in results/metrics/model_comparison.csv\n")

def run_experiment_2():
    print("="*50)
    print("EXPERIMENT 2: Temporal Early Warning")
    print("="*50)
    
    df_interim = pd.read_csv('data/interim/combined_data.csv')
    df_eligible = preprocess_data(df_interim)
    
    cutoffs = ['M3', 'M5', 'M7', 'PreFinal']
    models = ['DecisionTree', 'RandomForest']
    scenario = 'S2'
    features = get_features(scenario)
    
    results_list = []
    
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/temporal', exist_ok=True)
    
    for cutoff in cutoffs:
        print(f"--- Cutoff: {cutoff} ---")
        df_featured = compute_features(df_eligible, cutoff_session=cutoff)
        
        # Split using standard pipeline split
        X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured)
        
        X_train = X_train_full[features]
        X_test = X_test_full[features]
        
        for model_name in models:
            print(f"  Training {model_name}...")
            base_model = get_model(model_name)
            
            # Tune
            best_model, best_params = tune_hyperparameters(base_model, model_name, X_train, y_train)
            
            # Save model for this cutoff
            save_model(best_model, f'models/temporal_{cutoff}_{model_name}.pkl')
            
            # Evaluate CV
            cv_results = evaluate_cv(best_model, X_train, y_train)
            
            # Evaluate Test
            test_results, y_pred = evaluate_test(best_model, X_test, y_test, f"Temporal_{cutoff}", model_name, output_dir='results/temporal')
            
            row = {
                'cutoff': cutoff,
                'model': model_name,
                **cv_results,
                **test_results
            }
            results_list.append(row)
            
    df_results = pd.DataFrame(results_list)
    df_results.to_csv('results/metrics/temporal_early_warning.csv', index=False)
    print("Experiment 2 completed. Results in results/metrics/temporal_early_warning.csv\n")

def run_experiment_3():
    print("="*50)
    print("EXPERIMENT 3: Context Robustness (Leave-Group-Out)")
    print("="*50)
    
    df_interim = pd.read_csv('data/interim/combined_data.csv')
    df_eligible = preprocess_data(df_interim)
    df_featured = compute_features(df_eligible, cutoff_session='PreFinal')
    
    scenario = 'S2'
    features = get_features(scenario)
    
    results_list = []
    
    # We will test DT and RF
    models = ['DecisionTree', 'RandomForest']
    
    os.makedirs('results/robustness', exist_ok=True)
    
    for model_name in models:
        # Train on AC -> Test on BDE
        print(f"--- {model_name} : Train AC -> Test BDE ---")
        ac_mask = df_featured['Scoring_Scheme'] == 'AC'
        bde_mask = df_featured['Scoring_Scheme'] == 'BDE'
        
        X_train_ac = df_featured.loc[ac_mask, features]
        y_train_ac = df_featured.loc[ac_mask, 'Competency_Label']
        
        X_test_bde = df_featured.loc[bde_mask, features]
        y_test_bde = df_featured.loc[bde_mask, 'Competency_Label']
        
        base_model = get_model(model_name)
        best_model_ac, _ = tune_hyperparameters(base_model, model_name, X_train_ac, y_train_ac)
        
        cv_ac = evaluate_cv(best_model_ac, X_train_ac, y_train_ac)
        test_bde, _ = evaluate_test(best_model_ac, X_test_bde, y_test_bde, "TrainAC_TestBDE", model_name, output_dir='results/robustness')
        
        results_list.append({
            'experiment': 'Train AC -> Test BDE',
            'model': model_name,
            **cv_ac,
            **test_bde
        })
        
        # Train on BDE -> Test on AC
        print(f"--- {model_name} : Train BDE -> Test AC ---")
        X_train_bde = df_featured.loc[bde_mask, features]
        y_train_bde = df_featured.loc[bde_mask, 'Competency_Label']
        
        X_test_ac = df_featured.loc[ac_mask, features]
        y_test_ac = df_featured.loc[ac_mask, 'Competency_Label']
        
        best_model_bde, _ = tune_hyperparameters(base_model, model_name, X_train_bde, y_train_bde)
        
        cv_bde = evaluate_cv(best_model_bde, X_train_bde, y_train_bde)
        test_ac, _ = evaluate_test(best_model_bde, X_test_ac, y_test_ac, "TrainBDE_TestAC", model_name, output_dir='results/robustness')
        
        results_list.append({
            'experiment': 'Train BDE -> Test AC',
            'model': model_name,
            **cv_bde,
            **test_ac
        })

    df_results = pd.DataFrame(results_list)
    df_results.to_csv('results/metrics/context_robustness.csv', index=False)
    print("Experiment 3 completed. Results in results/metrics/context_robustness.csv\n")

if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
