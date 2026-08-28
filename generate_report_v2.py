import pandas as pd

def generate_report():
    df_students = pd.read_csv('data/processed/students_master.csv')
    df_results = pd.read_csv('outputs/model_results_v2.csv')
    df_shap_comp = pd.read_csv('outputs/shap/feature_importance_comparison.csv')
    
    n_total = len(df_students)
    kelas_counts = df_students['Kelas'].value_counts().to_dict()
    
    n_kompeten = len(df_students[df_students['Competency_Label'] == 1])
    n_belum_kompeten = len(df_students[df_students['Competency_Label'] == 0])
    
    reason_counts = df_students['Label_Reason'].value_counts().to_dict()
    
    md = f"""# HASIL ANALISIS MIGRASI DBNR (REVISI METODOLOGI OOF + SENSITIVITY)

## 1. Pemrosesan Data & Distribusi Ground Truth
Dataset final divalidasi berjumlah tepat **{n_total}** mahasiswa (setelah filter anomali dan asisten).

### Distribusi Kelas:
"""
    for k, v in sorted(kelas_counts.items()):
        md += f"- Kelas {k}: {v} mahasiswa\n"

    md += f"""
### Distribusi Target Kompetensi:
- **Kompeten (>=75): {n_kompeten}**
- **Belum Kompeten: {n_belum_kompeten}**
  - *(Terdiri dari: {reason_counts.get('Final_Score<75', 0)} nilai final <75, dan {reason_counts.get('No_Final_Attendance', 0)} Tidak Mengikuti Final)*

*(Aturan: Mahasiswa yang tidak memiliki data final dilabeli Belum Kompeten sebagai representasi akademik asli 'Tidak Mengikuti Final', bukan diisi dengan raw score 0).*

---

## 2. Hasil Modeling CV & Robustness Imbalance

Pemodelan dieksekusi dengan *Repeated Stratified 5-Fold CV*. Eksperimen ini membandingkan penanganan ketidakseimbangan kelas (*Imbalance Handling*) secara kokoh di dalam _inner-pipeline_.

### Top 15 Kombinasi Cutoff & Model (Sorted by Balanced Accuracy):
| Cutoff | Feature Set | Model | Balancing | Balanced Acc | Recall BK | F1 Macro |
|---|---|---|---|---:|---:|---:|
"""
    df_top = df_results.sort_values('BalAcc_Mean', ascending=False).head(15)
    for _, row in df_top.iterrows():
        md += f"| {row['Cutoff']} | {row['FeatureSet']} | {row['Model']} | {row['Balancing']} | {row['BalAcc_Mean']:.3f} | {row['RecallBK_Mean']:.3f} | {row['F1Macro_Mean']:.3f} |\n"

    md += """
*(Perhatikan bagaimana pendekatan `Class_Weight` umumnya bersaing ketat dengan `SMOTE`. Dummy classifier digunakan murni sebagai baseline prior).*

---

## 3. Subgroup OOF Error Analysis (C_Full, S5, RF, Class_Weight)

Analisis prediksi agregat *Out-Of-Fold* (OOF) memungkinkan kita melihat seberapa valid pendeteksian di luar data latih.

### Kasus False Negative & False Positive

**False Positive (Prediksi BK, aslinya Kompeten):**
*(Late-bloomers atau false alarm)*
*(Lihat file `outputs/error_analysis/oof_aggregated_errors.csv` untuk rincian lengkap).*

**False Negative (Prediksi Kompeten, aslinya BK):**
*(Late-droppers atau miss alarm)*
*(Lihat file `outputs/error_analysis/oof_aggregated_errors.csv` untuk rincian lengkap).*

---

## 4. Sensitivity SHAP Analysis

Analisis Explainable AI (TreeSHAP) dijalankan dalam 2 mode untuk memverifikasi apakah model hanya sekadar menghafal orang yang "tidak hadir final" atau berhasil menangkap substansi pola kegagalan secara general.

- **Main (n=89)**: Memasukkan seluruh data.
- **Sensitivity (n=86)**: Mengecualikan 3 mahasiswa yang tidak hadir final.

| Rank (Main) | Feature | SHAP (n=89) | SHAP (n=86) |
|---:|---|---:|---:|
"""
    
    for i, row in df_shap_comp.head(15).iterrows():
        md += f"| {i+1} | {row['Feature']} | {row['Mean_Abs_SHAP_Main']:.4f} | {row['Mean_Abs_SHAP_Sens']:.4f} |\n"

    md += """
**Interpretasi Sensitivitas:**
Jika peringkat atas didominasi fitur yang sama secara konsisten di kedua analisis, maka pola awal _early warning_ (*Attendance*, *TP*, *Laporan*) sangat prediktif secara universal (bukan _artifact_ statistik akibat 3 mahasiswa _no-final_).
"""

    with open('HASIL_ANALISIS_V2.md', 'w') as f:
        f.write(md)
        
    print("Report HASIL_ANALISIS_V2.md generated successfully.")

if __name__ == "__main__":
    generate_report()
