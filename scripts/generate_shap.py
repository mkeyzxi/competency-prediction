"""
Generate SHAP explanations for the FINAL EWS model (C_Full, S3_E, DecisionTree).

This script loads the frozen model saved by run_optimized_experiment.py
(models/FINAL_DecisionTree_S3_E_None_C_Full.pkl) and produces:
  - results/shap/global_importance_FINAL_S3_E_DecisionTree.png
  - results/shap/beeswarm_FINAL_S3_E_DecisionTree.png
  - results/shap/local_{TP,TN,FP,FN}_FINAL_S3_E_DecisionTree.png
  - results/shap/shap_values_FINAL_S3_E_DecisionTree.csv
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.shap_analysis import run_shap_analysis
from src.split import get_train_test_split
from src.feature_registry import get_features
from src.utils import load_model
from src.models import get_model
from src.tuning import evaluate_thresholds

MODEL_NAME = 'DecisionTree'
SCENARIO = 'S3_E'
BALANCING = 'None'
CUTOFF = 'C_Full'
CONFIG_PATH = 'configs/experiment_config.yaml'
FILE_PATH = f'data/features/{CUTOFF}_{SCENARIO}.csv'


def main():
    print(f"Generating SHAP explanations for FINAL {SCENARIO} ({MODEL_NAME})...")

    if not os.path.exists(FILE_PATH):
        print(f"Dataset {FILE_PATH} not found. "
              f"Run 'python src/features.py' first.")
        return

    df_featured = pd.read_csv(FILE_PATH)
    X_train_full, X_test_full, y_train, y_test = get_train_test_split(
        df_featured, CONFIG_PATH)

    features = get_features(SCENARIO)
    available_features = [f for f in features if f in X_test_full.columns]
    X_test = X_test_full[available_features]

    # Locate the frozen model (saved by run_optimized_experiment.py)
    model_path = f'models/FINAL_{MODEL_NAME}_{SCENARIO}_{BALANCING}_{CUTOFF}.pkl'
    if not os.path.exists(model_path):
        # Fallback: rebuild & tune on the train set
        print(f"Model {model_path} not found. Rebuilding from train set...")
        base_model = get_model(MODEL_NAME, BALANCING, CONFIG_PATH)
        from src.tuning import tune_hyperparameters
        final_model, _ = tune_hyperparameters(
            base_model, MODEL_NAME, X_train_full[available_features], y_train,
            CONFIG_PATH, balancing=BALANCING)
        os.makedirs('models', exist_ok=True)
        from src.utils import save_model
        save_model(final_model, model_path)
        model = final_model
    else:
        model = load_model(model_path)

    # Use the frozen threshold (recompute on train set — same as pipeline)
    base_model = get_model(MODEL_NAME, BALANCING, CONFIG_PATH)
    from src.tuning import tune_hyperparameters
    tuned, _ = tune_hyperparameters(base_model, MODEL_NAME,
                                    X_train_full[available_features], y_train,
                                    CONFIG_PATH, balancing=BALANCING)
    _, best_threshold, _ = evaluate_thresholds(tuned, X_train_full[available_features],
                                               y_train, optimize_for='f2_bk')
    p_bk = model.predict_proba(X_test)[:, 0]
    y_pred = np.where(p_bk >= best_threshold, 0, 1)

    run_shap_analysis(
        model_path, X_test, y_test, y_pred,
        f'FINAL_{SCENARIO}', MODEL_NAME)
    print(f"SHAP generation for {model_path} completed. "
          f"Figures in results/shap/ (prefix FINAL_{SCENARIO}).")


if __name__ == "__main__":
    main()
