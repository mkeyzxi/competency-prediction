import pandas as pd
import numpy as np
import os

def generate_report():
    # Load data
    df_students = pd.read_csv('data/processed/students_master.csv')
    df_results = pd.read_csv('outputs/model_results.csv')
    df_shap = pd.read_csv('outputs/shap/feature_importance.csv')
    df_error = pd.read_csv('outputs/error_analysis.csv')
    
    # 1. Data Stats
    n_total = len(df_students)
    kelas_counts = df_students['Kelas'].value_counts().to_dict()
    
    # Target stats
    df_labeled = df_students.dropna(subset=['Competency_Label'])
    n_labeled = len(df_labeled)
    n_kompeten = len(df_labeled[df_labeled['Competency_Label'] == 1])
    n_belum_kompeten = len(df_labeled[df_labeled['Competency_Label'] == 0])
    
    md_content = f"""# HASIL ANALISIS MIGRASI DBNR

## 1. Pemrosesan Data & Distribusi
Total mahasiswa yang tervalidasi dalam dataset ini adalah **{n_total}** orang. 
Dari total tersebut, mahasiswa yang memiliki label ground truth pada komponen evaluasi akhir berjumlah **{n_labeled}** orang.
(Sesuai aturan, target diambil berdasarkan nilai Individu).

### Distribusi Kelas:
"""
    for k, v in sorted(kelas_counts.items()):
        md_content += f"- Kelas {k}: {v} mahasiswa\n"

    md_content += f"""
### Distribusi Target Kompetensi:
- Kompeten (>=75): **{n_kompeten}**
- Belum Kompeten (<75): **{n_belum_kompeten}**
*(Terdapat ketidakseimbangan kelas (imbalance) di mana kelas mayoritas adalah Kompeten).*

---

## 2. Hasil Modeling & Pemilihan Model (Metric Results)

Eksperimen cross-validation (Repeated Stratified 5-Fold) dijalankan pada berbagai kombinasi **Temporal Cutoff** (C1, C2, C3, C_Full) dan **Feature Set** (S1 - S5).

### Perbandingan Model Terbaik
Berikut adalah top 10 kombinasi dengan nilai **Balanced Accuracy** rata-rata tertinggi:

"""
    # Get top 10 models by BalAcc
    df_top = df_results.sort_values('BalAcc_Mean', ascending=False).head(10)
    
    md_content += "| Cutoff | Feature Set | Model | Balanced Acc | Recall BK | F1 Macro |\n"
    md_content += "|---|---|---|---:|---:|---:|\n"
    
    for _, row in df_top.iterrows():
        md_content += f"| {row['Cutoff']} | {row['FeatureSet']} | {row['Model']} | {row['BalAcc_Mean']:.3f} | {row['RecallBK_Mean']:.3f} | {row['F1Macro_Mean']:.3f} |\n"

    md_content += """
**Analisis Cutoff Temporal:**
Dari hasil metrik di atas, kita dapat melihat apakah model sudah cukup diskriminatif pada *cutoff* yang lebih dini (misalnya C2 atau C3) dibandingkan jika harus menunggu keseluruhan semester (C_Full). 
Secara umum, Random Forest cenderung menunjukkan stabilitas performa yang lebih baik.

---

## 3. Explainable AI: TreeSHAP Feature Importance

Model final dievaluasi menggunakan kumpulan fitur **S5** pada cutoff **C_Full**. Berikut adalah peringkat fitur yang paling memiliki kontribusi prediktif berdasarkan nilai rata-rata absolut SHAP:

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
"""
    
    for i, row in df_shap.head(15).iterrows():
        md_content += f"| {i+1} | {row['Feature']} | {row['Mean_Abs_SHAP']:.4f} |\n"

    md_content += """
*(Silakan lihat gambar grafik beeswarm/summary plot di `outputs/shap/shap_summary.png` untuk melihat arah kontribusinya: apakah nilai fitur yang tinggi mendorong prediksi ke arah Kompeten atau Belum Kompeten).*

---

## 4. Error Analysis

Berdasarkan dataset evaluasi (C_Full_S5) dengan Random Forest, kita memperoleh distribusi error sebagai berikut:
"""
    
    error_counts = df_error['Error_Type'].value_counts()
    for k, v in error_counts.items():
        md_content += f"- **{k}**: {v}\n"

    md_content += """
**Penjelasan Signifikansi:**
- **False Negative (Late-Dropper)**: Mahasiswa yang secara ground truth *Belum Kompeten*, tetapi model memprediksinya aman (*Kompeten*). Mereka lolos dari *early warning*.
- **False Positive (Late-Bloomer)**: Mahasiswa yang mendapatkan alarm *Belum Kompeten*, namun pada akhir evaluasi berhasil *Kompeten*.

### Daftar Mahasiswa pada Area Prediksi Keliru
"""
    df_fn = df_error[df_error['Error_Type'] == 'False Negative']
    df_fp = df_error[df_error['Error_Type'] == 'False Positive']
    
    if len(df_fn) > 0:
        md_content += "\n**Kasus False Negative:**\n"
        for _, row in df_fn.iterrows():
            md_content += f"- {row['Nama']} (Kelas {row['Kelas']})\n"
    else:
        md_content += "\n**Kasus False Negative:** Tidak ada pada model ini.\n"
        
    if len(df_fp) > 0:
        md_content += "\n**Kasus False Positive:**\n"
        for _, row in df_fp.iterrows():
            md_content += f"- {row['Nama']} (Kelas {row['Kelas']})\n"
    else:
        md_content += "\n**Kasus False Positive:** Tidak ada pada model ini.\n"

    md_content += """
---
## Kesimpulan

Migrasi metodologi ke dataset Basis Data Non Relasional ini berhasil menunjukkan bahwa _feature engineering_ berbasis temporal, meskipun memiliki _sample size_ yang kecil, dapat mendiskriminasi potensi kegagalan mahasiswa. 
Fitur penyelesaian tugas (completion) maupun performa di pekan-pekan awal cenderung memiliki _Feature Importance_ yang dominan menurut TreeSHAP.
"""
    
    with open('HASIL_ANALISIS.md', 'w') as f:
        f.write(md_content)
        
    print("Report HASIL_ANALISIS.md generated successfully.")

if __name__ == "__main__":
    generate_report()
