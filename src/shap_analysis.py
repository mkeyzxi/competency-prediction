import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from src.utils import load_model

def run_shap_analysis(model_path: str, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, scenario: str, model_name: str):
    """
    Runs TreeSHAP explanation for the given model and test set.
    """
    model = load_model(model_path)
    
    if hasattr(model, 'named_steps'):
        clf = model.named_steps['model']
        imputer = model.named_steps['imputer']
        X_test_transformed = imputer.transform(X_test)
        
        if hasattr(imputer, 'get_feature_names_out'):
            new_cols = imputer.get_feature_names_out(X_test.columns)
        else:
            new_cols = X_test.columns
            
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=new_cols, index=X_test.index)
    else:
        clf = model
        X_test_transformed = X_test
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_test_transformed)
    
    # We want the explanation for the positive class (Belum Kompeten, index 0)
    if isinstance(shap_values.values, list) or len(shap_values.shape) == 3:
        shap_values_pos = shap_values[..., 0]
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
    
    # Categorize cases: Positive = Belum Kompeten (0), Negative = Kompeten (1)
    y_true = y_test.values
    
    # TP: Actual 0, Pred 0
    tp_indices = np.where((y_true == 0) & (y_pred == 0))[0]
    # TN: Actual 1, Pred 1
    tn_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    # FP: Actual 1, Pred 0
    fp_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    # FN: Actual 0, Pred 1
    fn_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    
    cases_to_plot = {
        'TP': tp_indices[0] if len(tp_indices) > 0 else None,
        'TN': tn_indices[0] if len(tn_indices) > 0 else None,
        'FP': fp_indices[0] if len(fp_indices) > 0 else None,
        'FN': fn_indices[0] if len(fn_indices) > 0 else None
    }
    
    # Specific detection for FN Late-Bloomers
    if 'Performance_Trend' in X_test_transformed.columns and len(fn_indices) > 0:
        for idx in fn_indices:
            if X_test_transformed.iloc[idx]['Performance_Trend'] < 0:
                cases_to_plot['FN_LateBloomer'] = idx
                break
    
    for case_type, idx in cases_to_plot.items():
        if idx is not None:
            plt.figure()
            shap.plots.waterfall(shap_values_pos[idx], show=False)
            plt.title(f'Local SHAP Waterfall - {case_type} - {scenario} {model_name}')
            plt.tight_layout()
            plt.savefig(f'results/shap/local_{case_type}_{scenario}_{model_name}.png')
            plt.close()
        
    # Export SHAP values to CSV
    shap_df = pd.DataFrame(shap_values_pos.values, columns=X_test_transformed.columns)
    shap_df.to_csv(f'results/shap/shap_values_{scenario}_{model_name}.csv', index=False)
    
    print(f"SHAP analysis complete for {scenario} {model_name}.")
