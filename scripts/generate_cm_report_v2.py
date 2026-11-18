"""
Generate CONFUSION MATRIX untuk laporan - data HEAD (n89) = data eksperimen
asli yang menghasilkan seluruh angka laporan (cutoff_comparison.csv,
leaderboard baseline, cm_FINAL).

Replika PERSIS pipeline otoritatif:
  A) EWS Multi-Cutoff (S3_E + DecisionTree)  [run_optimized_experiment.py]
  B) Baseline Retrospektif S1-S5 x DT/RF     [run_baseline_experiments.py]

Verifikasi otomatis terhadap angka laporan. Output ke results/confusion_matrix/.
"""
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_test

# Data eksperimen asli (ekstraksi git HEAD = n89: 45 BK + 44 K)
DATA_DIR = r'C:\Users\muhma\AppData\Local\Temp\opencode\feat_head'
OUT_DIR = 'results/confusion_matrix'
CONFIG = 'configs/experiment_config.yaml'

os.makedirs(OUT_DIR, exist_ok=True)


def run_ews():
    print("=" * 72)
    print("A) EWS MULTI-CUTOFF - S3_E + DecisionTree (data n89, eksperimen asli)")
    print("=" * 72)
    scenario, model_name, balancing = 'S3_E', 'DecisionTree', 'None'
    ref = {'W1': (8, 1, 9, 0, 0.26), 'W2': (9, 0, 4, 5, 0.2),
           'W3': (9, 0, 4, 5, 0.24), 'C_Full': (8, 1, 3, 6, 0.52)}
    all_ok = True
    for cutoff in ['W1', 'W2', 'W3', 'C_Full']:
        df = pd.read_csv(os.path.join(DATA_DIR, f'{cutoff}_{scenario}.csv'))
        X_tr_f, X_te_f, y_tr, y_te = get_train_test_split(df, CONFIG)
        feats = [f for f in get_features(scenario) if f in X_tr_f.columns]
        X_tr, X_te = X_tr_f[feats], X_te_f[feats]

        base = get_model(model_name, balancing)
        tuned, _ = tune_hyperparameters(base, model_name, X_tr, y_tr,
                                        balancing=balancing)
        _, best_t, _ = evaluate_thresholds(tuned, X_tr, y_tr,
                                           optimize_for='f2_bk')
        final_model, _ = tune_hyperparameters(
            get_model(model_name, balancing), model_name,
            X_tr, y_tr, balancing=balancing)
        res, _ = evaluate_test(final_model, X_te, y_te,
                               scenario_name=f'EWS_{cutoff}_S3_E',
                               model_name=model_name,
                               threshold=best_t, output_dir=OUT_DIR)
        tp, fn, fp, tn = (res['test_tp'], res['test_fn'],
                          res['test_fp'], res['test_tn'])
        rtp, rfn, rfp, rtn, rt = ref[cutoff]
        ok = (tp, fn, fp, tn) == (rtp, rfn, rfp, rtn) and abs(best_t - rt) < 1e-9
        all_ok &= ok
        print(f"[{'OK ' if ok else 'BEDA'}] {cutoff:7s} t={best_t} | "
              f"TP={tp}, FN={fn}, FP={fp}, TN={tn} | "
              f"RecallBK={res['test_recall_bk']*100:.2f}% "
              f"Spec={res['test_specificity']*100:.2f}% "
              f"BA={res['test_balanced_accuracy']*100:.2f}%")
        if not ok:
            print(f"        !! referensi laporan: TP={rtp},FN={rfn},FP={rfp},TN={rtn} (t={rt})")
        print(f"        -> cm_EWS_{cutoff}_S3_E_{model_name}.png")
    return all_ok


def run_baseline():
    print("\n" + "=" * 72)
    print("B) BASELINE RETROSPEKTIF S1-S5 x DT/RF (data n89, eksperimen asli)")
    print("=" * 72)
    ref = {('S3', 'DecisionTree'): (9, 0, 4, 5, 0.52)}
    for scenario in ['S1', 'S2', 'S3', 'S4', 'S5']:
        df = pd.read_csv(os.path.join(DATA_DIR, f'C_Full_{scenario}.csv'))
        X_tr_f, X_te_f, y_tr, y_te = get_train_test_split(df, CONFIG)
        feats = [f for f in get_features(scenario) if f in X_tr_f.columns]
        X_tr, X_te = X_tr_f[feats], X_te_f[feats]
        for model_name in ['DecisionTree', 'RandomForest']:
            base = get_model(model_name, balancing='None', config_path=CONFIG)
            final_model, _ = tune_hyperparameters(base, model_name,
                                                  X_tr, y_tr, CONFIG)
            _, best_t, _ = evaluate_thresholds(final_model, X_tr, y_tr)
            res, _ = evaluate_test(final_model, X_te, y_te,
                                   scenario_name=f'Baseline_{scenario}',
                                   model_name=model_name,
                                   threshold=best_t, output_dir=OUT_DIR)
            tp, fn, fp, tn = (res['test_tp'], res['test_fn'],
                              res['test_fp'], res['test_tn'])
            key = (scenario, model_name)
            mark = ''
            if key in ref:
                rtp, rfn, rfp, rtn, rt = ref[key]
                mark = (' [OK: cocok leaderboard]' if (tp, fn, fp, tn) == (rtp, rfn, rfp, rtn)
                        else ' [BEDA dari leaderboard!]')
            print(f"  {scenario} {model_name:13s} t={best_t} | "
                  f"TP={tp}, FN={fn}, FP={fp}, TN={tn} | "
                  f"RecallBK={res['test_recall_bk']*100:.2f}% "
                  f"BA={res['test_balanced_accuracy']*100:.2f}% "
                  f"PR-AUC={res['test_pr_auc']:.3f}{mark}")
            print(f"                -> cm_Baseline_{scenario}_{model_name}.png")


def main():
    ok = run_ews()
    run_baseline()
    print("\n" + "=" * 72)
    print("SELESAI. Verifikasi EWS vs angka laporan:",
          "SEMUA COCOK" if ok else "ADA PERBEDAAN (lihat di atas)")


if __name__ == '__main__':
    main()
