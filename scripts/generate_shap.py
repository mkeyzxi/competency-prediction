import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.shap_analysis import run_shap_analysis
from src.split import get_train_test_split
from src.feature_registry import get_features
from src.utils import load_config, load_model
from src.preprocessing import preprocess_data
from src.feature_engineering import compute_features

def main():
    print("Generating SHAP explanations for Random Forest...")
    
    if not os.path.exists('data/interim/combined_data.csv'):
        print("Dataset not found.")
        return
        
    df_interim = pd.read_csv('data/interim/combined_data.csv')
    df_eligible = preprocess_data(df_interim)
    df_featured = compute_features(df_eligible, cutoff_session='PreFinal')
    
    config_path = 'configs/experiment_config.yaml'
    config = load_config(config_path)
    scenarios = config['experiments']
    
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured, config_path)
    
    for scenario in scenarios:
        features = get_features(scenario)
        if not features:
            continue
            
        X_test = X_test_full[features]
        
        model_name = 'RandomForest'
        model_path = f'models/{scenario}_{model_name}.pkl'
        
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                y_pred = model.predict(X_test)
                run_shap_analysis(model_path, X_test, y_test, y_pred, scenario, model_name)
            except Exception as e:
                print(f"Error running SHAP for {scenario} {model_name}: {e}")
        else:
            print(f"Model {model_path} not found. Skipping SHAP.")
                
    print("SHAP generation completed.")

if __name__ == "__main__":
    main()
