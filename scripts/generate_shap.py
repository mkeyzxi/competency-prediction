import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.shap_analysis import run_shap_analysis
from src.split import get_train_test_split
from src.feature_registry import get_features
from src.utils import load_config

def main():
    print("Generating SHAP explanations...")
    
    if not os.path.exists('data/processed/featured_full.csv'):
        print("Featured dataset not found. Please run build_features.py first.")
        return
        
    df = pd.read_csv('data/processed/featured_full.csv')
    
    # We need X_test for the models.
    # The split is deterministic because of random_state
    config_path = 'configs/experiment_config.yaml'
    config = load_config(config_path)
    scenarios = config['experiments']
    model_names = config['models']
    
    X_train_full, X_test_full, _, _ = get_train_test_split(df, config_path)
    
    for scenario in scenarios:
        features = get_features(scenario)
        if not features:
            continue
            
        X_test = X_test_full[features]
        
        for model_name in model_names:
            model_path = f'models/{scenario}_{model_name}.pkl'
            if os.path.exists(model_path):
                try:
                    run_shap_analysis(model_path, X_test, scenario, model_name)
                except Exception as e:
                    print(f"Error running SHAP for {scenario} {model_name}: {e}")
            else:
                print(f"Model {model_path} not found. Skipping SHAP.")
                
    print("SHAP generation completed.")

if __name__ == "__main__":
    main()
