import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def run_shap_analysis():
    os.makedirs('outputs/shap', exist_ok=True)
    os.makedirs('outputs/error_analysis', exist_ok=True)
    
    # 1. Error Analysis from OOF
    df_oof = pd.read_csv('outputs/oof_predictions.csv')
    
    # Select best model config: C_Full, S5, RF, Class_Weight
    # We aggregate predictions across repeats (e.g. majority vote) or just take the mean probability
    df_model = df_oof[(df_oof['Cutoff'] == 'C_Full') & 
                      (df_oof['Feature_Set'] == 'S5') & 
                      (df_oof['Model'] == 'RF') & 
                      (df_oof['Balancing'] == 'Class_Weight')]
                      
    if df_model.empty:
        print("Could not find OOF data for C_Full, S5, RF, Class_Weight")
    else:
        # Average probability across the 3 repeats
        agg = df_model.groupby(['NIM', 'Nama', 'Kelas', 'Actual', 'Label_Reason']).agg({
            'Probability_BK': 'mean'
        }).reset_index()
        
        # If mean prob >= 0.5 -> Predicted BK (0), else Kompeten (1)
        agg['Predicted'] = (agg['Probability_BK'] < 0.5).astype(int)
        
        def get_error_type(r):
            # Positif = Alarm (Belum Kompeten = 0)
            if r['Actual'] == 1 and r['Predicted'] == 0: return 'False Positive' # Aman tapi diprediksi BK (False Alarm)
            if r['Actual'] == 0 and r['Predicted'] == 1: return 'False Negative' # BK tapi diprediksi Aman (Missed Alarm)
            return 'Correct'
            
        agg['Error_Type'] = agg.apply(get_error_type, axis=1)
        
        agg.to_csv('outputs/error_analysis/oof_aggregated_errors.csv', index=False)
        
        # Subgroup Summary
        print("=== Error Analysis Summary ===")
        print(agg['Error_Type'].value_counts())
        
        print("\n=== BK Subgroups Breakdown (Actual=0) ===")
        bk_only = agg[agg['Actual'] == 0]
        print(pd.crosstab(bk_only['Label_Reason'], bk_only['Error_Type']))
    
    
    # 2. SHAP Sensitivity Analysis
    df = pd.read_csv('data/features/C_Full_S5.csv')
    df = df.dropna(subset=['Competency_Label'])
    
    X = df.drop(columns=['Nama', 'Kelas', 'Competency_Label', 'Final_Score', 'Final_Status', 'Label_Reason'])
    y = df['Competency_Label']
    
    def generate_shap(X_data, y_data, suffix):
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X_data), columns=X_data.columns)
        
        model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
        model.fit(X_imputed, y_data)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_imputed)
        
        if isinstance(shap_values, list):
            shap_vals_pos = shap_values[1]
        else:
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    shap_vals_pos = shap_values.values[:, :, 1]
                else:
                    shap_vals_pos = shap_values.values
            else:
                if len(shap_values.shape) == 3:
                    shap_vals_pos = shap_values[:, :, 1]
                else:
                    shap_vals_pos = shap_values
                    
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals_pos, X_imputed, show=False)
        plt.savefig(f'outputs/shap/shap_summary_{suffix}.png', bbox_inches='tight')
        plt.close()
        
        mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
        feat_importance = pd.DataFrame({
            'Feature': X_data.columns,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        
        feat_importance.to_csv(f'outputs/shap/feature_importance_{suffix}.csv', index=False)
        return feat_importance
        
    # Main Analysis (n=89)
    print("\nGenerating SHAP for Main (n=89)...")
    fi_main = generate_shap(X, y, 'Main_n89')
    
    # Sensitivity Analysis (n=86, removing No_Final_Attendance)
    print("Generating SHAP for Sensitivity (n=86)...")
    mask_86 = df['Label_Reason'] != 'No_Final_Attendance'
    df_86 = df[mask_86]
    X_86 = df_86.drop(columns=['Nama', 'Kelas', 'Competency_Label', 'Final_Score', 'Final_Status', 'Label_Reason'])
    y_86 = df_86['Competency_Label']
    fi_sens = generate_shap(X_86, y_86, 'Sensitivity_n86')
    
    # Merge for comparison
    comparison = fi_main.merge(fi_sens, on='Feature', suffixes=('_Main', '_Sens'))
    comparison.to_csv('outputs/shap/feature_importance_comparison.csv', index=False)
    print("SHAP and Error analysis generated.")

if __name__ == '__main__':
    run_shap_analysis()
