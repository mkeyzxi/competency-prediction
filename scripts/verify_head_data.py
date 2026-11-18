"""Verifikasi: apakah data HEAD (n89) mereproduksi angka laporan?"""
import os, sys, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_test

HEAD_DIR = r'C:\Users\muhma\AppData\Local\Temp\opencode\feat_head'
NO_SAVE = True

def run(cutoff, scenario, model_name='DecisionTree', balancing='None', label=''):
    df = pd.read_csv(os.path.join(HEAD_DIR, f'{cutoff}_{scenario}.csv'))
    X_tr_f, X_te_f, y_tr, y_te = get_train_test_split(df)
    feats = [f for f in get_features(scenario) if f in X_tr_f.columns]
    X_tr, X_te = X_tr_f[feats], X_te_f[feats]
    base = get_model(model_name, balancing)
    tuned, _ = tune_hyperparameters(base, model_name, X_tr, y_tr, balancing=balancing)
    _, best_t, _ = evaluate_thresholds(tuned, X_tr, y_tr, optimize_for='f2_bk')
    final_model, _ = tune_hyperparameters(get_model(model_name, balancing), model_name,
                                          X_tr, y_tr, balancing=balancing)
    if NO_SAVE:
        # evaluate_test tanpa menyimpan PNG: panggil manual intinya
        import numpy as np
        from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score
        p_bk = final_model.predict_proba(X_te)[:, 0]
        y_pred = np.where(p_bk >= best_t, 0, 1)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tp, fn, fp, tn = cm.ravel()
        rec = recall_score(y_te, y_pred, pos_label=0, zero_division=0)
        ba = balanced_accuracy_score(y_te, y_pred)
        spec = tn / (tn + fp) if (tn + fp) else 0.0
    print(f"{label:28s} t={best_t} | TP={tp}, FN={fn}, FP={fp}, TN={tn} | "
          f"RecallBK={rec*100:.2f}% Spec={spec*100:.2f}% BA={ba*100:.2f}%")

print("=== EWS S3_E DT pada data HEAD (n89) ===")
for c in ['W1', 'W2', 'W3', 'C_Full']:
    run(c, 'S3_E', label=f'EWS {c} (referensi CSV)')

print()
print("=== Baseline C_Full pada data HEAD (n89) ===")
for sc in ['S1', 'S2', 'S3']:
    for mn in ['DecisionTree', 'RandomForest']:
        df = pd.read_csv(os.path.join(HEAD_DIR, f'C_Full_{sc}.csv'))
        X_tr_f, X_te_f, y_tr, y_te = get_train_test_split(df)
        feats = [f for f in get_features(sc) if f in X_tr_f.columns]
        X_tr, X_te = X_tr_f[feats], X_te_f[feats]
        base = get_model(mn, balancing='None')
        final_model, _ = tune_hyperparameters(base, mn, X_tr, y_tr)
        _, best_t, _ = evaluate_thresholds(final_model, X_tr, y_tr)
        import numpy as np
        from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score
        from sklearn.metrics import average_precision_score
        p_bk = final_model.predict_proba(X_te)[:, 0]
        y_pred = np.where(p_bk >= best_t, 0, 1)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tp, fn, fp, tn = cm.ravel()
        rec = recall_score(y_te, y_pred, pos_label=0, zero_division=0)
        ba = balanced_accuracy_score(y_te, y_pred)
        prauc = average_precision_score(y_te, p_bk, pos_label=0)
        print(f"Baseline {sc} {mn:13s} t={best_t} | TP={tp}, FN={fn}, FP={fp}, TN={tn} | "
              f"RecallBK={rec*100:.2f}% BA={ba*100:.2f}% PR-AUC={prauc:.3f}")
