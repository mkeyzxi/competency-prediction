import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters
from src.shap_analysis import run_shap_analysis
from src.utils import save_model

def main():
    print("Training S3 Dasar and generating SHAP...")
    
    config_path = 'configs/experiment_config.yaml'
    file_path = 'data/features/C_Full_S3.csv'
    
    df = pd.read_csv(file_path)
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(df, config_path)
    
    scenario = 'S3'
    model_name = 'DecisionTree'
    
    features = get_features(scenario)
    available_features = [f for f in features if f in X_train_full.columns]
    
    X_train = X_train_full[available_features]
    X_test = X_test_full[available_features]
    
    base_model = get_model(model_name, balancing='None', config_path=config_path)
    final_model, best_params = tune_hyperparameters(base_model, model_name, X_train, y_train, config_path)
    
    model_path = f'models/Baseline_{model_name}_{scenario}_None.pkl'
    os.makedirs('models', exist_ok=True)
    save_model(final_model, model_path)
    
    y_pred = final_model.predict(X_test)
    
    run_shap_analysis(
        model_path, X_test, y_test, y_pred,
        f'Baseline_{scenario}', model_name
    )
    
    print("SHAP generation for S3 Dasar completed.")

if __name__ == "__main__":
    main()
