# HASIL ANALISIS MIGRASI DBNR

## 1. Pemrosesan Data & Distribusi
Total mahasiswa yang tervalidasi dalam dataset ini adalah **89** orang. 
Dari total tersebut, mahasiswa yang memiliki label ground truth pada komponen evaluasi akhir berjumlah **86** orang.
(Sesuai aturan, target diambil berdasarkan nilai Individu).

### Distribusi Kelas:
- Kelas A: 12 mahasiswa
- Kelas B: 23 mahasiswa
- Kelas C: 24 mahasiswa
- Kelas D: 19 mahasiswa
- Kelas E: 11 mahasiswa

### Distribusi Target Kompetensi:
- Kompeten (>=75): **78**
- Belum Kompeten (<75): **8**
*(Terdapat ketidakseimbangan kelas (imbalance) di mana kelas mayoritas adalah Kompeten).*

---

## 2. Hasil Modeling & Pemilihan Model (Metric Results)

Eksperimen cross-validation (Repeated Stratified 5-Fold) dijalankan pada berbagai kombinasi **Temporal Cutoff** (C1, C2, C3, C_Full) dan **Feature Set** (S1 - S5).

### Perbandingan Model Terbaik
Berikut adalah top 10 kombinasi dengan nilai **Balanced Accuracy** rata-rata tertinggi:

| Cutoff | Feature Set | Model | Balanced Acc | Recall BK | F1 Macro |
|---|---|---|---:|---:|---:|
| C_Full | S1 | DT | 0.977 | 0.967 | 0.960 |
| C1 | S3 | RF | 0.933 | 0.867 | 0.941 |
| C1 | S2 | RF | 0.933 | 0.867 | 0.941 |
| C3 | S2 | RF | 0.931 | 0.867 | 0.934 |
| C2 | S2 | RF | 0.929 | 0.867 | 0.921 |
| C1 | S5 | DT | 0.929 | 0.867 | 0.922 |
| C1 | S5 | RF | 0.929 | 0.867 | 0.921 |
| C2 | S1 | DT | 0.927 | 0.867 | 0.914 |
| C2 | S1 | RF | 0.927 | 0.867 | 0.914 |
| C2 | S3 | DT | 0.927 | 0.867 | 0.914 |

**Analisis Cutoff Temporal:**
Dari hasil metrik di atas, kita dapat melihat apakah model sudah cukup diskriminatif pada *cutoff* yang lebih dini (misalnya C2 atau C3) dibandingkan jika harus menunggu keseluruhan semester (C_Full). 
Secara umum, Random Forest cenderung menunjukkan stabilitas performa yang lebih baik.

---

## 3. Explainable AI: TreeSHAP Feature Importance

Model final dievaluasi menggunakan kumpulan fitur **S5** pada cutoff **C_Full**. Berikut adalah peringkat fitur yang paling memiliki kontribusi prediktif berdasarkan nilai rata-rata absolut SHAP:

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | Laporan_Mean | 0.0291 |
| 2 | Performance_Mean | 0.0258 |
| 3 | Performance_Late_Mean | 0.0211 |
| 4 | Laporan_Min | 0.0189 |
| 5 | Laporan_First2_Mean | 0.0182 |
| 6 | Laporan_Last2_Mean | 0.0166 |
| 7 | Laporan_Max | 0.0109 |
| 8 | Absence_Count | 0.0092 |
| 9 | TP_First2_Mean | 0.0088 |
| 10 | Performance_Std | 0.0054 |
| 11 | TP_Mean | 0.0044 |
| 12 | TP_Last2_Mean | 0.0033 |
| 13 | TP_Max | 0.0032 |
| 14 | TP_Min | 0.0016 |
| 15 | Laporan_Std | 0.0015 |

*(Silakan lihat gambar grafik beeswarm/summary plot di `outputs/shap/shap_summary.png` untuk melihat arah kontribusinya: apakah nilai fitur yang tinggi mendorong prediksi ke arah Kompeten atau Belum Kompeten).*

---

## 4. Error Analysis

Berdasarkan dataset evaluasi (C_Full_S5) dengan Random Forest, kita memperoleh distribusi error sebagai berikut:
- **Correct**: 86

**Penjelasan Signifikansi:**
- **False Negative (Late-Dropper)**: Mahasiswa yang secara ground truth *Belum Kompeten*, tetapi model memprediksinya aman (*Kompeten*). Mereka lolos dari *early warning*.
- **False Positive (Late-Bloomer)**: Mahasiswa yang mendapatkan alarm *Belum Kompeten*, namun pada akhir evaluasi berhasil *Kompeten*.

### Daftar Mahasiswa pada Area Prediksi Keliru

**Kasus False Negative:** Tidak ada pada model ini.

**Kasus False Positive:** Tidak ada pada model ini.

---
## Kesimpulan

Migrasi metodologi ke dataset Basis Data Non Relasional ini berhasil menunjukkan bahwa _feature engineering_ berbasis temporal, meskipun memiliki _sample size_ yang kecil, dapat mendiskriminasi potensi kegagalan mahasiswa. 
Fitur penyelesaian tugas (completion) maupun performa di pekan-pekan awal cenderung memiliki _Feature Importance_ yang dominan menurut TreeSHAP.
