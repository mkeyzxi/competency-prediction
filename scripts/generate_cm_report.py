"""
Generate ulang CONFUSION MATRIX yang dibutuhkan laporan (populasi n86).

Replikasi PERSIS pipeline otoritatif:
  A) EWS Multi-Cutoff (S3_E + DecisionTree, balancing None)
     -> replika scripts/run_optimized_experiment.py (W1, W2, W3, C_Full)
     -> verifikasi otomatis terhadap results/final/cutoff_comparison.csv
  B) Baseline Retrospektif (S1-S5 x DecisionTree/RandomForest)
     -> replika scripts/run_baseline_experiments.py (tanpa evaluate_cv,
        karena CV tidak memengaruhi evaluasi holdout)

Output PNG ditulis ke results/confusion_matrix/ dengan gaya plot yang sama
dengan src/evaluation.py (sns heatmap 'Blues', label BK(0)/K(1)).
Tidak ada file di results/final yang ditimpa.
"""
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import json
import pandas as pd

from src.feature_registry import get_features
from src.split import get_train_test_split
from src.models import get_model
from src.tuning import tune_hyperparameters, evaluate_thresholds
from src.evaluation import evaluate_test

OUT_DIR = 'results/confusion_matrix'
CONFIG = 'configs/experiment_config.yaml'

os.makedirs(OUT_DIR, exist_ok=True)


def fmt_cm(res):
    return (res.get('test_tp', 0), res.get('test_fn', 0),
            res.get('test_fp', 0), res.get('test_tn', 0))


# ============================================================
# A) EWS MULTI-CUTOFF (S3_E, DecisionTree) - replika persis
#    scripts/run_optimized_experiment.py
# ============================================================
def run_ews_cutoffs():
    print("=" * 70)
    print("BAGIAN A: EWS MULTI-CUTOFF (S3_E, DecisionTree, balancing=None)")
    print("=" * 70)
    scenario, model_name, balancing = 'S3_E', 'DecisionTree', 'None'
    rows = []

    for cutoff in ['W1', 'W2', 'W3', 'C_Full']:
        file_path = f'data/features/{cutoff}_{scenario}.csv'
        df = pd.read_csv(file_path)
        print(f"\n--- Cutoff {cutoff} ({len(df)} mahasiswa) ---")

        X_train_full, X_test_full, y_train, y_test = get_train_test_split(
            df, CONFIG)
        feats = [f for f in get_features(scenario) if f in X_train_full.columns]
        X_train, X_test = X_train_full[feats], X_test_full[feats]

        # 1) Tuning (inner CV) - persis seperti script asli
        base_model = get_model(model_name, balancing)
        tuned_model, tuned_params = tune_hyperparameters(
            base_model, model_name, X_train, y_train, balancing=balancing)

        # 2) Threshold dari training data (f2_bk)
        _, best_threshold, _ = evaluate_thresholds(
            tuned_model, X_train, y_train, optimize_for='f2_bk')

        # 3) Final holdout evaluation (re-fit seluruh train, dipanggil sekali)
        final_model, _ = tune_hyperparameters(
            get_model(model_name, balancing), model_name,
            X_train, y_train, balancing=balancing)
        res, y_pred = evaluate_test(
            final_model, X_test, y_test,
            scenario_name=f'EWS_{cutoff}', model_name=model_name,
            threshold=best_threshold, output_dir=OUT_DIR)

        tp, fn, fp, tn = fmt_cm(res)
        print(f"  Threshold : {best_threshold}")
        print(f"  CM        : TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        print(f"  Recall BK : {res['test_recall_bk']*100:.2f}% | "
              f"Spec: {res['test_specificity']*100:.2f}% | "
              f"BA: {res['test_balanced_accuracy']*100:.2f}%")
        print(f"  Disimpan   : {OUT_DIR}/cm_EWS_{cutoff}_{model_name}.png")

        rows.append({'Cutoff': cutoff, 'Threshold': best_threshold,
                     'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
                     'Recall_BK': res['test_recall_bk'],
                     'Specificity': res['test_specificity'],
                     'BA': res['test_balanced_accuracy']})
    return pd.DataFrame(rows)


# ============================================================
# B) BASELINE RETROSPEKTIF (S1-S5 x DT/RF) - replika persis
#    scripts/run_baseline_experiments.py (tanpa evaluate_cv)
# ============================================================
def run_baselines():
    print("\n" + "=" * 70)
    print("BAGIAN B: BASELINE RETROSPEKTIF (S1-S5 x DT/RF, data C_Full n86)")
    print("=" * 70)
    rows = []
    for scenario in ['S1', 'S2', 'S3', 'S4', 'S5']:
        file_path = f'data/features/C_Full_{scenario}.csv'
        df = pd.read_csv(file_path)
        print(f"\n--- Skenario {scenario} ({len(df)} mahasiswa) ---")

        X_train_full, X_test_full, y_train, y_test = get_train_test_split(
            df, CONFIG)
        feats = [f for f in get_features(scenario) if f in X_train_full.columns]
        X_train, X_test = X_train_full[feats], X_test_full[feats]

        for model_name in ['DecisionTree', 'RandomForest']:
            # Persis run_baseline_experiments.py: get_model(balancing='None')
            # lalu tune_hyperparameters(config_path) -> param grid default
            base_model = get_model(model_name, balancing='None',
                                   config_path=CONFIG)
            final_model, _ = tune_hyperparameters(
                base_model, model_name, X_train, y_train, CONFIG)
            _, best_thresh, _ = evaluate_thresholds(final_model, X_train, y_train)
            res, y_pred = evaluate_test(
                final_model, X_test, y_test,
                scenario_name=f'Baseline_{scenario}', model_name=model_name,
                threshold=best_thresh, output_dir=OUT_DIR)

            tp, fn, fp, tn = fmt_cm(res)
            print(f"  {model_name:13s} t={best_thresh} | "
                  f"CM TP={tp}, FN={fn}, FP={fp}, TN={tn} | "
                  f"Recall BK {res['test_recall_bk']*100:.2f}% | "
                  f"BA {res['test_balanced_accuracy']*100:.2f}% | "
                  f"PR-AUC {res.get('test_pr_auc', float('nan')):.3f}")
            print(f"               -> {OUT_DIR}/cm_Baseline_{scenario}_{model_name}.png")

            rows.append({'Scenario': scenario, 'Model': model_name,
                         'Threshold': best_thresh,
                         'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
                         'Recall_BK': res['test_recall_bk'],
                         'Specificity': res['test_specificity'],
                         'BA': res['test_balanced_accuracy'],
                         'PR_AUC': res.get('test_pr_auc')})
    return pd.DataFrame(rows)


# ============================================================
# VERIFIKASI terhadap cutoff_comparison.csv (otoritatif)
# ============================================================
def verify_ews(df_new):
    ref_path = 'results/final/cutoff_comparison.csv'
    if not os.path.exists(ref_path):
        print("[WARN] cutoff_comparison.csv tidak ditemukan - lewati verifikasi")
        return
    ref = pd.read_csv(ref_path).set_index('Cutoff')
    print("\n" + "=" * 70)
    print("VERIFIKASI vs results/final/cutoff_comparison.csv")
    print("=" * 70)
    all_ok = True
    for _, r in df_new.iterrows():
        c = r['Cutoff']
        if c not in ref.index:
            continue
        g = ref.loc[c]
        ok = (int(g['Test_TP']) == r['TP'] and int(g['Test_FN']) == r['FN']
              and int(g['Test_FP']) == r['FP'] and int(g['Test_TN']) == r['TN']
              and abs(float(g['Threshold']) - r['Threshold']) < 1e-9)
        status = 'COCOK ✓' if ok else 'TIDAK COCOK ✗'
        if not ok:
            all_ok = False
        print(f"  {c:7s}: baru TP={r['TP']},FN={r['FN']},FP={r['FP']},TN={r['TN']} "
              f"(t={r['Threshold']}) vs referensi TP={int(g['Test_TP'])},"
              f"FN={int(g['Test_FN'])},FP={int(g['Test_FP'])},TN={int(g['Test_TN'])} "
              f"(t={g['Threshold']}) -> {status}")
    print("\nKESIMPULAN:", "SEMUA COCOK - gambar autentik dari pipeline yang sama"
          if all_ok else "ADA PERBEDAAN - perlu investigasi data/pipeline")


def main():
    df_ews = run_ews_cutoffs()
    df_base = run_baselines()
    verify_ews(df_ews)

    df_ews.to_csv(os.path.join(OUT_DIR, '_summary_ews_n86.csv'), index=False)
    df_base.to_csv(os.path.join(OUT_DIR, '_summary_baseline_n86.csv'), index=False)
    print(f"\nRingkasan disimpan: {OUT_DIR}/_summary_ews_n86.csv, "
          f"{OUT_DIR}/_summary_baseline_n86.csv")


if __name__ == '__main__':
    main()
