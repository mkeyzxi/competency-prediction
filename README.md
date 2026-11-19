# 🚀 Panduan Eksekusi Eksperimen Prediksi Kompetensi (EWS)

Repositori ini berisi keseluruhan sistem (_pipeline_) eksperimen _Machine Learning_ untuk mendeteksi dini potensi kegagalan (**Belum Kompeten**) mahasiswa pada Praktikum Logika & Algoritma.

**Batas Kompetensi = 83** (Zona Aman A-) — dataset diolah dalam kondisi murni (tanpa oversampling SMOTE) untuk menjaga keaslian distribusi dan mencegah *optimistic bias*.

**Versi V3** menambahkan:
- 📊 **Confusion Matrix berlabel lengkap** (TP/FN/FP/TN + %)
- 📈 **Kurva ROC & Precision-Recall** + grafik sensitivitas threshold
- 🧪 **SHAP per-cutoff** (W1, W2, W3, C_Full) — global, beeswarm, waterfall TP/TN/FP/FN
- 🔢 **Metrik lengkap**: Recall/Precision/F2 BK, Balanced Accuracy, **MCC**, ROC-AUC, PR-AUC + 95% CI bootstrap (B=1000)
- 🤖 **Dummy baseline (majority-class)** untuk pembuktian bahwa EWS benar-benar menambah informasi
- 📄 **Laporan otomatis** → `HASIL_ANALISIS_V3.md`

---

## 🧱 Struktur Proyek

```
Logika-Algoritma/
├── data/
│   ├── raw/DBNR.xlsx                  # data mentah dari dosen
│   ├── processed/                     # setelah Langkah 1
│   └── features/                      # setelah Langkah 2 (C_Full_*, W1_*, ...)
├── configs/experiment_config.yaml     # seed, CV, split
├── src/                               # modul inti
│   ├── build_dataset.py               # Excel → CSV bersih
│   ├── features.py                    # ekstraksi fitur multi-cutoff
│   ├── split.py                       # train/holdout stratified
│   ├── models.py                      # DecisionTree/RF pipeline
│   ├── tuning.py                      # RandomizedSearchCV + threshold
│   ├── evaluation.py                  # Nested CV + holdout + CM + CI
│   ├── shap_analysis.py               # TreeSHAP (global/beeswarm/local)
│   └── plots.py                       # ROC/PR, threshold, cutoff grid
├── scripts/
│   ├── run_baseline_experiments.py    # S1–S5 (retrospektif)
│   ├── run_optimized_experiment.py    # EWS multi-cutoff + dummy baseline
│   ├── generate_shap_s3_dasar.py      # SHAP S3 Dasar
│   ├── generate_shap.py               # SHAP Model Final (C_Full)
│   └── generate_analysis_v3.py        # HASIL_ANALISIS_V3.md otomatis
├── models/                            # .pkl (joblib)
├── results/
│   ├── confusion_matrix/              # CM .png + .json
│   ├── plots/                         # ROC/PR, threshold, cutoff
│   ├── shap/                          # SHAP .png + .csv
│   ├── metrics/                       # matriks CV
│   └── final/                         # JSON & CSV laporan
├── run_pipeline.bat                   # eksekusi cepat semua langkah
└── HASIL_ANALISIS_V3.md               # laporan hasil
```

---

## ⚙️ Persiapan Lingkungan

```bash
conda env create -f environment.yml
conda activate logika-ml
# atau: pip install -r requirements.txt
```

Kebutuhan: `pandas, numpy, scikit-learn>=1.5, seaborn, shap>=0.45, imbalanced-learn, matplotlib, pyyaml, joblib, openpyxl`.

---

## 🏃 Cara Menjalankan (Urut)

Buka terminal di folder proyek (`C:\belajarku\Belajar ML\Logika-Algoritma`), lalu jalankan **berikut secara berurutan**:

### Langkah 1 — Bersihkan data mentah
```bash
python src\build_dataset.py
```
Baca `data/raw/DBNR.xlsx` (Sheet A–E), tandai kompetensi dengan batas **83**, dan simpan ke `data/processed/` (`students_master.csv`, `activities_long.csv`). Populasi akhir: **n=86 mahasiswa valid** (kelas A–E, asisten terhapus).

### Langkah 2 — Ekstraksi fitur multi-cutoff
```bash
python src\features.py
```
Untuk setiap cutoff (`W1`, `W2`, `W3`, `C_Full`) dan skenario fitur (`S1`–`S5`, `S3_A`–`S3_E`, `S3_EWS`), hitung fitur agregat dan EWS (mis. `Early_Performance_Composite`, `Absence_Streak_Max`). Hasil: `data/features/{Cutoff}_{Skenario}.csv`.

### Langkah 3 — Baseline retrospektif (S1–S5)
```bash
python scripts\run_baseline_experiments.py
```
Jalankan Nested CV + tuning + evaluasi holdout sekali untuk `DecisionTree` & `RandomForest` pada S1–S5 (data semester penuh `C_Full`). Output:
- `results/final/baseline_leaderboard.csv`
- `results/confusion_matrix/cm_Baseline_S*_*.{png,json}`
- `results/plots/roc_pr_Baseline_*.png`, `threshold_sensitivity_*.png`
- `models/Baseline_{Model}_{Skenario}_None.pkl`

### Langkah 4 — EWS multi-cutoff + Dummy baseline
```bash
python scripts\run_optimized_experiment.py
```
Evaluasi `DecisionTree + S3_E` pada **W1, W2, W3, C_Full** + **Dummy baseline** pada C_Full. Protokol ketat:
1. Model dipilih dari **Nested CV (outer 5×5, inner RandomizedSearchCV 3-fold)** — holdout tidak disentuh
2. Threshold dipilih di **inner CV** (optimasi `f2_bk` dengan constraint `Recall BK ≥ 0.70`)
3. Model di-fit ulang di seluruh train, lalu **dievaluasi SEKALI** di holdout `n=18`
4. 95% CI bootstrap (B=1000) untuk Recall/BalAcc/F2/MCC

Output:
- `models/FINAL_DecisionTree_S3_E_None_{W1,W2,W3,C_Full}.pkl`
- `results/confusion_matrix/cm_EWS_{cutoff}_S3_E_DecisionTree.{png,json}`
- `results/plots/roc_pr_EWS_*.png`, `threshold_sensitivity_*.png`, `cutoff_comparison.png`, `cm_grid_all_cutoffs.png`
- `results/shap/*EWS_{cutoff}_S3_E_DecisionTree*` (global, beeswarm, waterfall TP/TN/FP/FN, CSV nilai SHAP)
- `results/final/final_model_report_{cutoff}.json`, `cutoff_comparison.csv`, `consolidated_summary.json`

### Langkah 5 — SHAP model dasar & final
```bash
python scripts\generate_shap_s3_dasar.py
python scripts\generate_shap.py
```
- **5A** — Latih ulang S3 Dasar (`C_Full`) dan simpan `models/Baseline_DecisionTree_S3_None.pkl` + SHAP berawalan `Baseline_S3`
- **5B** — Ambil `models/FINAL_DecisionTree_S3_E_None_C_Full.pkl` (hasil Langkah 4) dan bedah dengan SHAP berawalan `FINAL_S3_E`

### Langkah 6 — Laporan otomatis V3
```bash
python scripts\generate_analysis_v3.py
```
Membaca semua artefak di atas dan menulis `HASIL_ANALISIS_V3.md` lengkap (tabel + embed figur + keterbatasan).

### Jalankan semuanya sekaligus
```bash
run_pipeline.bat
```
Atau manual berurutan seperti Langkah 1–6.

---

## 📁 Di Mana Hasil Tersimpan?

| Kategori | Lokasi |
|:---------|:-------|
| **Confusion matrix** (CM berlabel TP/FN/FP/TN + %) | `results/confusion_matrix/cm_*.png` + `.json` |
| **Kurva ROC & PR** | `results/plots/roc_pr_*.png` |
| **Sensitivitas threshold** | `results/plots/threshold_sensitivity_*.png` + `threshold_table_*.csv` |
| **Perbandingan cutoff EWS** | `results/plots/cutoff_comparison.png` + `cm_grid_all_cutoffs.png` |
| **SHAP global / beeswarm / waterfall** | `results/shap/global_importance_*.png`, `beeswarm_*.png`, `local_{TP,TN,FP,FN}_*.png` |
| **Metrik numerik** (JSON/CSV) | `results/final/final_model_report_*.json`, `cutoff_comparison.csv`, `consolidated_summary.json`, `baseline_leaderboard.csv` |
| **Matriks CV lengkap** | `results/metrics/cv_experiment_matrix.csv` |
| **Model beku** | `models/*.pkl` |
| **Laporan manusia** | `HASIL_ANALISIS_V3.md` |

---

## 📚 Definisi Metrik yang Dipakai

Semua metrik dihitung dengan kelas **Belum Kompeten (BK, label=0)** sebagai "positif" (karena target EWS adalah menangkap mahasiswa berisiko):

- **Recall BK** = TP / (TP+FN) — seberapa banyak BK yang tertangkap
- **Precision BK** = TP / (TP+FP) — berapa persen alarm yang benar-benar BK
- **F2 BK** (beta=2) — memberi bobot 2× lebih besar ke recall daripada precision
- **Specificity** = TN / (TN+FP) — proporsi mahasiswa Kompeten yang tidak di-alarm
- **Balanced Accuracy** = (Recall + Specificity) / 2
- **MCC** (Matthews Correlation Coefficient) — koefisien korelasi antara prediksi & label; `≈0` = acak, `0.5` = substansial
- **ROC-AUC** — area kurva ROC (menggunakan `P(K) = 1 − P(BK)`)
- **PR-AUC** — area kurva Precision-Recall (menggunakan `P(BK)`); lebih robust pada kelas imbalance
- **95% CI Bootstrap** — resample holdout `n=18` sebanyak `B=1000`, persentil 2.5/97.5

**Dummy baseline (majority-class)** memprediksi selalu kelas paling umum (Kompeten). Balanced Acc-nya 50% di holdout — nilai ambang yang harus dipertimbangkan: EWS yang tidak melewati dummy baseline berarti tidak menambah informasi.

---

## 🎯 Kesimpulan Singkat V3

- **Cutoff W3** adalah titik paling awal yang mencapai **Recall BK 100%** dengan **Balanced Acc 72.22%** (95% CI 55.56–88.47%).
- **C_Full** mempertahankan Recall BK 100% + Balanced Acc 72.22% + **MCC 0.535** (hubungan substansial), **ROC-AUC 0.790**, **PR-AUC 0.713**.
- **Dummy baseline** = 50.0% BalAcc → EWS memberikan **peningkatan +22.22 poin** (signifikan).
- **SHAP global** pada model final: `Early_Performance_Composite` mendominasi — model benar-benar membaca **adaptasi 2 pertemuan pertama**, bukan rekap agregat akhir.
- W1 & W2 masih over-alarming (Recall tinggi tapi Specificity rendah) — trade-off yang sesuai dengan filosofi EWS (minimalkan False Negative).

Selamat bereksperimen! 🎓
