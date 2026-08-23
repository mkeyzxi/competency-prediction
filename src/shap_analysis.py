import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.utils import load_model

def run_shap_analysis(model_path: str, X_test: pd.DataFrame, scenario: str, model_name: str):
    """
    Runs TreeSHAP explanation for the given model and test set.
    """
    model = load_model(model_path)
    
    # Initialize explainer
    # TreeExplainer works for RandomForest and DecisionTree
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Note: For classification, shap_values might be a list of arrays (one for each class)
    # We want the explanation for the positive class (Competen, index 1)
    if isinstance(shap_values.values, list) or len(shap_values.shape) == 3:
        # shap_values[..., 1] gets the values for the positive class
        shap_values_pos = shap_values[..., 1]
    else:
        shap_values_pos = shap_values
        
    os.makedirs('results/shap', exist_ok=True)
    
    # Global Importance Plot (Bar)
    plt.figure()
    shap.summary_plot(shap_values_pos, X_test, plot_type="bar", show=False)
    plt.title(f'Global SHAP Importance - {scenario} {model_name}')
    plt.tight_layout()
    plt.savefig(f'results/shap/global_importance_{scenario}_{model_name}.png')
    plt.close()
    
    # Beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values_pos, X_test, show=False)
    plt.title(f'SHAP Beeswarm - {scenario} {model_name}')
    plt.tight_layout()
    plt.savefig(f'results/shap/beeswarm_{scenario}_{model_name}.png')
    plt.close()
    
    # Local explanation (first 2 cases as example)
    for i in range(min(2, len(X_test))):
        plt.figure()
        shap.plots.waterfall(shap_values_pos[i], show=False)
        plt.title(f'Local SHAP Waterfall - Case {i} - {scenario} {model_name}')
        plt.tight_layout()
        plt.savefig(f'results/shap/local_case_{i}_{scenario}_{model_name}.png')
        plt.close()
        
    # Export SHAP values for the positive class to CSV for manual analysis
    shap_df = pd.DataFrame(shap_values_pos.values, columns=X_test.columns)
    shap_df.to_csv(f'results/shap/shap_values_{scenario}_{model_name}.csv', index=False)
    
    print(f"SHAP analysis complete for {scenario} {model_name}.")
