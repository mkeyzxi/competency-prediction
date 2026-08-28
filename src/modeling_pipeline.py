import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score, precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import os

warnings.filterwarnings('ignore')

def get_models(balancing):
    # balancing can be 'None', 'Class_Weight', 'SMOTE'
    cw = 'balanced' if balancing == 'Class_Weight' else None
    
    if balancing in ['None', 'SMOTE']:
        models = {
            'DT': DecisionTreeClassifier(random_state=42),
            'RF': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        if balancing == 'None':
            models['Dummy'] = DummyClassifier(strategy='prior')
    else:
        # Class_Weight
        models = {
            'DT': DecisionTreeClassifier(random_state=42, class_weight=cw),
            'RF': RandomForestClassifier(random_state=42, n_estimators=100, class_weight=cw)
        }
    return models

def run_evaluation():
    os.makedirs('outputs', exist_ok=True)
    cutoffs = ['C1', 'C2', 'C3', 'C_Full']
    sets = ['S1', 'S2', 'S3', 'S4', 'S5']
    
    results = []
    oof_predictions = []
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    for c in cutoffs:
        for s in sets:
            df = pd.read_csv(f'data/features/{c}_{s}.csv')
            df = df.dropna(subset=['Competency_Label']).reset_index(drop=True)
            
            X_cols = [col for col in df.columns if col not in ['Nama', 'Kelas', 'Competency_Label', 'Final_Score', 'Final_Status', 'Label_Reason']]
            X = df[X_cols]
            y = df['Competency_Label'].astype(int)
            
            for balancing in ['None', 'Class_Weight', 'SMOTE']:
                models = get_models(balancing)
                for model_name, model in models.items():
                    
                    steps = [('imputer', SimpleImputer(strategy='median'))]
                    if balancing == 'SMOTE' and model_name != 'Dummy':
                        steps.append(('smote', SMOTE(random_state=42)))
                    steps.append(('classifier', model))
                    
                    pipeline = ImbPipeline(steps)
                    
                    # Manual CV loop to capture OOF predictions
                    fold_metrics = []
                    repeat_idx = 1
                    fold_idx = 1
                    
                    for train_idx, test_idx in cv.split(X, y):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        
                        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                            y_prob = pipeline.predict_proba(X_test)[:, 0] # Prob for class 0 (Belum Kompeten)
                        else:
                            y_prob = np.where(y_pred == 0, 1.0, 0.0)
                            
                        # Store OOF
                        for i, t_idx in enumerate(test_idx):
                            oof_predictions.append({
                                'NIM': df.iloc[t_idx].get('NIM', f'idx_{t_idx}'),
                                'Nama': df.iloc[t_idx]['Nama'],
                                'Kelas': df.iloc[t_idx]['Kelas'],
                                'Actual': y_test.iloc[i],
                                'Predicted': y_pred[i],
                                'Probability_BK': y_prob[i],
                                'Final_Score': df.iloc[t_idx]['Final_Score'],
                                'Final_Status': df.iloc[t_idx]['Final_Status'],
                                'Label_Reason': df.iloc[t_idx]['Label_Reason'],
                                'Cutoff': c,
                                'Feature_Set': s,
                                'Model': model_name,
                                'Balancing': balancing,
                                'Repeat': repeat_idx,
                                'Fold': fold_idx
                            })
                            
                        fold_metrics.append({
                            'bal_acc': balanced_accuracy_score(y_test, y_pred),
                            'recall_bk': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
                            'f1_macro': f1_score(y_test, y_pred, average='macro'),
                            'prec_bk': precision_score(y_test, y_pred, pos_label=0, zero_division=0)
                        })
                        
                        fold_idx += 1
                        if fold_idx > 5:
                            fold_idx = 1
                            repeat_idx += 1
                            
                    fm_df = pd.DataFrame(fold_metrics)
                    results.append({
                        'Cutoff': c,
                        'FeatureSet': s,
                        'Model': model_name,
                        'Balancing': balancing,
                        'BalAcc_Mean': fm_df['bal_acc'].mean(),
                        'BalAcc_Std': fm_df['bal_acc'].std(),
                        'RecallBK_Mean': fm_df['recall_bk'].mean(),
                        'RecallBK_Std': fm_df['recall_bk'].std(),
                        'F1Macro_Mean': fm_df['f1_macro'].mean(),
                        'F1Macro_Std': fm_df['f1_macro'].std(),
                        'PrecBK_Mean': fm_df['prec_bk'].mean()
                    })
                    
    df_results = pd.DataFrame(results)
    df_results.to_csv('outputs/model_results_v2.csv', index=False)
    
    df_oof = pd.DataFrame(oof_predictions)
    df_oof.to_csv('outputs/oof_predictions.csv', index=False)
    
    print("Evaluation completed. Saved to outputs/model_results_v2.csv and oof_predictions.csv")

if __name__ == '__main__':
    run_evaluation()
