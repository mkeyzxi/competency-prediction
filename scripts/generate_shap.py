import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.shap_analysis import run_shap_analysis
from src.split import get_train_test_split
from src.feature_registry import get_features
from src.utils import load_config, load_model
from src.feature_engineering import compute_features

def main():
    print("Generating SHAP explanations for best tree models...")
    
    populations = ['P2']
    config_path = 'configs/experiment_config.yaml'
    config = load_config(config_path)
    scenarios = ['S6']
    
    for pop in populations:
        for cutoff in ['C1', 'C2', 'C3', 'C4', 'C_Full']:
            file_path = f'data/processed/featured_{pop}_{cutoff}_full.csv'
            if not os.path.exists(file_path):
                continue
                
            print(f"\n--- SHAP for Population {pop} Cutoff {cutoff} ---")
            df_featured = pd.read_csv(file_path)
            
            X_train_full, X_test_full, y_train, y_test = get_train_test_split(df_featured, config_path)
            
            for scenario in scenarios:
                features = get_features(scenario)
                if not features:
                    continue
                
                available_features = [f for f in features if f in X_test_full.columns]
                X_test = X_test_full[available_features]
                
                # Run SHAP for both DT and RF
                for model_name in ['DecisionTree', 'RandomForest']:
                    model_path = f'models/{pop}_{cutoff}_{scenario}_{model_name}_optimized.pkl'
                    
                    if os.path.exists(model_path):
                        try:
                            model = load_model(model_path)
                            y_pred = model.predict(X_test)
                            run_shap_analysis(
                                model_path, X_test, y_test, y_pred,
                                f'{pop}_{cutoff}_{scenario}', model_name
                            )
                        except Exception as e:
                            print(f"Error running SHAP for {pop} {cutoff} {scenario} {model_name}: {e}")
                    else:
                        print(f"Model {model_path} not found. Skipping SHAP.")
                    
    print("\nSHAP generation completed.")

if __name__ == "__main__":
    main()
