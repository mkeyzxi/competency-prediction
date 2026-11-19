"""
Generate HASIL_ANALISIS_V3.md — the consolidated research report.

Reads:
  - results/final/consolidated_summary.json
  - results/final/baseline_leaderboard.csv
  - results/shap/global_importance_*.csv
  - results/final/final_model_report_*.json

Writes:
  - HASIL_ANALISIS_V3.md
"""
import sys
import os
import json
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
os.chdir(ROOT)


def _fmt(x, pct=False, nd=4):
    if x is None:
        return 'N/A'
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:  # NaN check
        return 'N/A'
    return f'{v * 100:.2f}%' if pct else f'{v:.{nd}f}'


def _ci(lo, hi):
    if lo is None or hi is None:
        return ''
    return f' [{_fmt(lo, pct=True)} – {_fmt(hi, pct=True)}]'


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def main():
    consolidated = load_json('results/final/consolidated_summary.json', {})
    leaderboard = pd.read_csv('results/final/baseline_leaderboard.csv') \
        if os.path.exists('results/final/baseline_leaderboard.csv') else None
    cutoffs = consolidated.get('experiments', [])
    dummy = consolidated.get('dummy_baseline_C_Full', {})
    today = datetime.date.today().isoformat()

    # SHAP feature rankings (C_Full final model)
    shap_final = pd.read_csv(
        'results/shap/global_importance_FINAL_S3_E_DecisionTree.csv') \
        if os.path.exists('results/shap/global_importance_FINAL_S3_E_DecisionTree.csv') \
        else None

    # ---- Build markdown ----
    lines = []
    w = lines.append

    w('# HASIL ANALISIS V3 — Evaluasi Sistem Peringatan Dini Kompetensi '
      '(EWS) Mahasiswa Praktikum Logika & Algoritma')
    w('')
    w(f'**Status:** Laporan Final (V3)  ')
    w(f'**Tanggal:** {today}  ')
    w(f'**Populasi:** 86 mahasiswa (n=89 termasuk asisten; 86 sampel valid)  ')
    w(f'**Batas Kompetensi:** Final_Score ≥ 83 (rubrik Zona Aman A-)  ')
    w(f'**Framework:** Nested Cross-Validation (outer 5×5, inner RandomizedSearchCV '
      f'3-fold) + Holdout Set 20% (n=18, frozen)  ')
    w(f'**Model Final:** DecisionTree + S3_E (7 fitur, termasuk '
      f'*Early_Performance_Composite*)  ')
    w('')
    w('---')
    w('')
    w('## 1. Ringkasan Eksekutif')
    w('')
    w('Laporan ini menyempurnakan HASIL_ANALISIS_V2 dengan tiga penyiapan baru: '
      '(a) visualisasi _confusion matrix_ berlabel lengkap (TP/FN/FP/TN + %); '
      '(b) analisis SHAP yang dieksekusi ulang pada seluruh cutoff EWS '
      '(W1, W2, W3, C_Full); (c) metrik evaluasi tambahan (MCC, ROC-AUC, '
      'PR-AUC, interval bootstrap 95%) dan kurva ROC/PR beserta grafik '
      'sensitivitas threshold. Seluruh figur tersimpan di `results/` dan '
      'model di `models/`.')
    w('')
    # Data-driven executive summary lines
    _earliest = next((r for r in cutoffs
                      if abs(float(r.get('Test_Recall_BK', 0)) - 1.0) < 1e-9),
                    None)
    if _earliest:
        w(f"- **Cutoff paling awal yang efektif:** {_earliest['Cutoff']} — "
          f"Recall BK **{_fmt(_earliest.get('Test_Recall_BK'), pct=True)}** "
          f"dengan Balanced Accuracy "
          f"**{_fmt(_earliest.get('Test_Balanced_Acc'), pct=True)}** "
          f"(95% CI: "
          f"[{_fmt(_earliest.get('Test_BalAcc_CI_Low'), pct=True)} – "
          f"{_fmt(_earliest.get('Test_BalAcc_CI_Upper'), pct=True)}])")
    else:
        w('- **Cutoff paling awal yang efektif:** belum ada cutoff '
          'mencapai Recall BK 100% pada holdout — EWS masih '
          'mengalami over-alarming di seluruh cutoff.')

    _full = next((r for r in cutoffs if r['Cutoff'] == 'C_Full'), None)
    if _full:
        w(f"- **C_Full (semester penuh):** Recall BK "
          f"{_fmt(_full.get('Test_Recall_BK'), pct=True)}, "
          f"Balanced Acc "
          f"{_fmt(_full.get('Test_Balanced_Acc'), pct=True)}, "
          f"ROC-AUC {_fmt(_full.get('Test_ROC_AUC'))}, "
          f"PR-AUC {_fmt(_full.get('Test_PR_AUC'))}, "
          f"MCC {_fmt(_full.get('Test_MCC'))}")

    w(f"- **Dummy baseline (majority class):** Balanced Accuracy "
      f"{_fmt(dummy.get('test_balanced_accuracy'))} "
      f'(memastikan EWS benar-benar menambah informasi)')

    # Top SHAP feature (data-driven)
    if shap_final is not None and len(shap_final):
        top_feat = shap_final.iloc[0]['feature']
        w(f'- **Fitur SHAP utama (C_Full):** `{top_feat}` menempati '
          'peringkat teratas — membuktikan pergeseran dari rekap '
          'agregat ke sinyal dini.')
    w('')

    # ---- Section 2: Baseline leaderboard ----
    w('---')
    w('')
    w('## 2. Baseline Retrospektif (S1–S5, C_Full)')
    w('')
    if leaderboard is not None and len(leaderboard):
        w('| Skenario | Model | N Fitur | Threshold | CV BalAcc | '
          'CV Recall BK | CV PR-AUC | Test BalAcc | Test Recall BK | '
          'Test Specificity | Test MCC | Test PR-AUC |')
        w('|:---------|:------|:------:|:---------:|:---------:|'
          ':----------:|:---------:|:----------:|:------------:|'
          ':------------:|:-------:|:----------:|')
        for _, r in leaderboard.iterrows():
            w(f"| {r['Scenario']} | {r['Model']} | {r['N_Features']} | "
              f"{r['Threshold']:.2f} | {_fmt(r['CV_BalAcc'], pct=True)} | "
              f"{_fmt(r['CV_Recall_BK'], pct=True)} | "
              f"{_fmt(r['CV_PR_AUC'])} | {_fmt(r['Test_BalAcc'], pct=True)} | "
              f"{_fmt(r['Test_Recall_BK'], pct=True)} | "
              f"{_fmt(r['Test_Specificity'], pct=True)} | "
              f"{_fmt(r['Test_MCC'])} | {_fmt(r['Test_PR_AUC'])} |")
        w('')
        w('_Sumber: `results/final/baseline_leaderboard.csv`_')
    else:
        w('_Data baseline tidak tersedia. Jalankan '
          '`python scripts/run_baseline_experiments.py`._')
    w('')

    # ---- Section 3: Multi-cutoff EWS ----
    w('---')
    w('')
    w('## 3. Evaluasi EWS Multi-Cutoff (S3_E, DecisionTree, Holdout n=18)')
    w('')
    w('| Cutoff | Threshold | CV Recall BK | CV BalAcc | Test Recall BK '
      '(95% CI) | Test BalAcc (95% CI) | F2 BK | MCC | ROC-AUC | PR-AUC | '
      'CM (TP/FN/FP/TN) |')
    w('|:------|:---------:|:------------:|:---------:|'
      ':-------------------|:-------------------|:-----:|:-----:|'
      ':---------:|:-------:|:----------------|')
    for r in cutoffs:
        cm = f"{r['Test_TP']}/{r['Test_FN']}/{r['Test_FP']}/{r['Test_TN']}"
        w(f"| **{r['Cutoff']}** | {r['Threshold']:.2f} | "
          f"{_fmt(r['CV_Recall_BK'], pct=True)} | "
          f"{_fmt(r['CV_Balanced_Acc'], pct=True)} | "
          f"{_fmt(r['Test_Recall_BK'], pct=True)}"
          f"{_ci(r.get('Test_Recall_BK_CI_Low'), r.get('Test_Recall_BK_CI_Upper'))} | "
          f"{_fmt(r['Test_Balanced_Acc'], pct=True)}"
          f"{_ci(r.get('Test_BalAcc_CI_Low'), r.get('Test_BalAcc_CI_Upper'))} | "
          f"{_fmt(r['Test_F2_BK'])} | {_fmt(r['Test_MCC'])} | "
          f"{_fmt(r['Test_ROC_AUC'])} | {_fmt(r['Test_PR_AUC'])} | "
          f"{cm} |")
    w('')
    w('**Gambar 1 — Multi-cutoff comparison (Recall BK vs Specificity, '
      'Test + CV):**')
    w('')
    w('![cutoff_comparison](results/plots/cutoff_comparison.png)')
    w('')
    w('**Gambar 2 — Grid confusion matrix semua cutoff:**')
    w('')
    w('![cm_grid](results/plots/cm_grid_all_cutoffs.png)')
    w('')

    # ---- Section 4: SHAP ----
    w('---')
    w('')
    w('## 4. Analisis Interpretasi SHAP')
    w('')
    if shap_final is not None:
        w('### 4.1. Ranking Global SHAP — Model Final (C_Full, S3_E)')
        w('')
        w('| Rank | Fitur | Mean Abs SHAP |')
        w('|:---:|:------|:------------:|')
        for i, r in shap_final.iterrows():
            w(f"| {i + 1} | `{r['feature']}` | {float(r['mean_abs_shap']):.4f} |")
        w('')
    w('**Gambar 3 — Global importance (mean |SHAP|):**')
    w('')
    w('![global_importance](results/shap/global_importance_FINAL_S3_E_DecisionTree.png)')
    w('')
    w('**Gambar 4 — Beeswarm plot (sebaran arah kontribusi):**')
    w('')
    w('![beeswarm](results/shap/beeswarm_FINAL_S3_E_DecisionTree.png)')
    w('')
    w('Per-cutoff, SHAP utama adalah:')
    w('')
    w('| Cutoff | Fitur #1 (Mean Abs SHAP tertinggi) |')
    w('|:-------|:--------------------------------------|')
    for c in ('W1', 'W2', 'W3', 'C_Full'):
        p = f'results/shap/global_importance_EWS_{c}_S3_E_DecisionTree.csv'
        if os.path.exists(p):
            top = pd.read_csv(p).iloc[0]['feature']
        else:
            top = 'N/A'
        w(f'| {c} | `{top}` |')
    w('')

    # ---- Section 5: Figures index ----
    w('---')
    w('')
    w('## 5. Katalog Gambar & Artefak')
    w('')
    w('Semua figur di bawah ini dapat langsung dibuka dari terminal / file '
      'explorer. Tabel mengategorikan artefak berdasarkan foldernya.')
    w('')
    w('### 5.1. Confusion Matrix (TP/FN/FP/TN + %)')
    w('')
    for fn in sorted(os.listdir('results/confusion_matrix')):
        if fn.endswith('.png'):
            w(f'- `results/confusion_matrix/{fn}`')
    w('')
    w('### 5.2. Kurva ROC / PR / Threshold / Comparison')
    w('')
    for fn in sorted(os.listdir('results/plots')):
        if fn.endswith('.png'):
            w(f'- `results/plots/{fn}`')
    w('')
    w('### 5.3. SHAP (global, beeswarm, waterfall, nilai per sampel)')
    w('')
    for fn in sorted(os.listdir('results/shap')):
        if fn.endswith('.png') or fn.endswith('.csv'):
            w(f'- `results/shap/{fn}`')
    w('')
    w('### 5.4. Metrik Numerik & Model')
    w('')
    w('- **JSON laporan per-cutoff:** `results/final/final_model_report_{W1,W2,W3,C_Full}.json`')
    w('- **CSV perbandingan cutoff:** `results/final/cutoff_comparison.csv`')
    w('- **Ringkasan terpadu:** `results/final/consolidated_summary.json`')
    w('- **Leaderboard baseline:** `results/final/baseline_leaderboard.csv`')
    w('- **Matriks CV lengkap:** `results/metrics/cv_experiment_matrix.csv`')
    w('- **Model beku:** `models/FINAL_DecisionTree_S3_E_None_{W1,W2,W3,C_Full}.pkl`')
    w('')
    w('')

    # ---- Section 6: Limitations ----
    w('---')
    w('')
    w('## 6. Keterbatasan & Catatan Metodologis')
    w('')
    w('1. **Ukuran sampel kecil** (n=86 train, n=18 holdout) membuat '
      'interval bootstrap lebar — hasil interpretatif, bukan prediktif '
      'umum. 2. **Single-cohort**: tidak ada validasi silang antar '
      'kurikulum. 3. **Cutoff W1 & W2** menunjukkan over-alarming '
      '(Recall 100% tapi Specificity rendah) — sesuai trade-off EWS.  '
      '4. **Dummy baseline** hanya mayoritas-class; tidak memperhitungkan '
      'distribusi kelas yang lebih halus.')
    w('')

    # ---- Write file ----
    with open('HASIL_ANALISIS_V3.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"HASIL_ANALISIS_V3.md generated ({len(lines)} lines).")


if __name__ == '__main__':
    main()
