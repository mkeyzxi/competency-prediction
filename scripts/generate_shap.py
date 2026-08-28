import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.shap_analysis import run_shap_analysis
from src.split import get_train_test_split
from src.feature_registry import get_features
from src.utils import load_config, load_model

def main():
    print("Generating SHAP explanations for the Final Optimized Model...")
    
    config_path = 'configs/experiment_config.yaml'
    file_path = 'data/features/C_Full_S3_EWS.csv'
    
    if not os.path.exists(file_path):
        print(f"Dataset {file_path} not found.")
        return
        
    df_featured = pd.read_csv(file_path)
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured, config_path)
    
    # Target the winner: DecisionTree | S3_E | None
    model_name = 'DecisionTree'
    scenario = 'S3_E'
    balancing = 'None'
    
    features = get_features(scenario)
    available_features = [f for f in features if f in X_test_full.columns]
    X_test = X_test_full[available_features]
    
    model_path = f'models/FINAL_{model_name}_{scenario}_{balancing}.pkl'
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            y_pred = model.predict(X_test)
            run_shap_analysis(
                model_path, X_test, y_test, y_pred,
                f'FINAL_{scenario}', model_name
            )
            print("SHAP generation completed successfully.")
        except Exception as e:
            print(f"Error running SHAP for {model_path}: {e}")
    else:
        print(f"Model {model_path} not found. Did you run run_optimized_experiment.py?")

if __name__ == "__main__":
    main()
