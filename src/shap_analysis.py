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
    
    # Extract explainer and transform X_test if the model is a Pipeline
    if hasattr(model, 'named_steps'):
        # Assuming the pipeline steps are 'imputer' and 'model' as defined in models.py
        clf = model.named_steps['model']
        imputer = model.named_steps['imputer']
        X_test_transformed = imputer.transform(X_test)
        
        # Get new feature names if the imputer added indicators
        if hasattr(imputer, 'get_feature_names_out'):
            new_cols = imputer.get_feature_names_out(X_test.columns)
        else:
            new_cols = X_test.columns
            
        # Convert back to DataFrame to preserve feature names for SHAP
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=new_cols, index=X_test.index)
    else:
        clf = model
        X_test_transformed = X_test
    
    # Initialize explainer
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_test_transformed)
    
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
    shap.summary_plot(shap_values_pos, X_test_transformed, plot_type="bar", show=False)
    plt.title(f'Global SHAP Importance - {scenario} {model_name}')
    plt.tight_layout()
    plt.savefig(f'results/shap/global_importance_{scenario}_{model_name}.png')
    plt.close()
    
    # Beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values_pos, X_test_transformed, show=False)
    plt.title(f'SHAP Beeswarm - {scenario} {model_name}')
    plt.tight_layout()
    plt.savefig(f'results/shap/beeswarm_{scenario}_{model_name}.png')
    plt.close()
    
    # Local explanation (first 2 cases as example)
    for i in range(min(2, len(X_test_transformed))):
        plt.figure()
        shap.plots.waterfall(shap_values_pos[i], show=False)
        plt.title(f'Local SHAP Waterfall - Case {i} - {scenario} {model_name}')
        plt.tight_layout()
        plt.savefig(f'results/shap/local_case_{i}_{scenario}_{model_name}.png')
        plt.close()
        
    # Export SHAP values for the positive class to CSV for manual analysis
    shap_df = pd.DataFrame(shap_values_pos.values, columns=X_test_transformed.columns)
    shap_df.to_csv(f'results/shap/shap_values_{scenario}_{model_name}.csv', index=False)
    
    print(f"SHAP analysis complete for {scenario} {model_name}.")
