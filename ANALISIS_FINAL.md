# Laporan Hasil Analisis Komprehensif (SINTA 2) - *Audit Recall Teroptimasi*

Laporan ini menyajikan hasil dari 4 eksperimen utama untuk memprediksi kelulusan (kompetensi) mahasiswa pada praktikum Logika Pemrograman. Data dikumpulkan dari dua dosen dengan skema penilaian yang berbeda (AC dan BDE). Berdasarkan audit terbaru, parameter optimasi (`GridSearchCV`) dan pengaturan bobot (`class_weight='balanced_subsample'`) telah diperbaiki untuk menargetkan performa metrik pada kelas **Belum Kompeten (0)** demi kebutuhan *Early Warning System*.

## 1. Perbandingan Model dan Set Fitur (Eksperimen 1)

Eksperimen pertama bertujuan untuk membandingkan 3 skenario ekstraksi fitur (S1: Basic Means, S2: Completion Rates, S3: Relational Gaps). Evaluasi dilakukan dengan `RepeatedStratifiedKFold` (5 split, 5 ulangan) guna mendapatkan Confidence Interval (CI) 95%. 

**Tabel 1: Rata-Rata Cross Validation (Fokus: Belum Kompeten)**

| Skenario | Model | CV F1-Score | CI 95% F1-Score | CV Recall | CV Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S1 | Decision Tree | 0.607 ± 0.104 | [0.563, 0.651] | 0.691 | 0.556 |
| S1 | Random Forest | 0.640 ± 0.092 | [0.601, 0.679] | 0.690 | 0.615 |
| S2 | Decision Tree | 0.593 ± 0.105 | [0.548, 0.637] | 0.655 | 0.556 |
| S2 | Random Forest | 0.640 ± 0.107 | [0.595, 0.685] | 0.681 | 0.625 |
| S3 | Decision Tree | 0.552 ± 0.105 | [0.507, 0.596] | 0.611 | 0.520 |
| S3 | Random Forest | 0.659 ± 0.121 | [0.608, 0.710] | 0.701 | 0.639 |

**Analisis Skenario 1 (Pascaperbaikan):**
- **Kenaikan Signifikan:** Melalui optimasi yang menargetkan kelas "Belum Kompeten", nilai CV Recall melonjak tinggi ke kisaran **68%-70%** (berbanding ~50-57% sebelum audit).
- **Rekomendasi Set Fitur:** Meskipun S3 secara CV menunjukkan performa tertinggi (F1 0.659), S2 sangat dekat (F1 0.640, Recall 0.681). Mengingat S2 difokuskan pada proporsi penyelesaian (Completion Rate) yang lebih kebal terhadap perbedaan skema, S2 dipertahankan sebagai fokus *Context Robustness*. Precision di angka ~62% merupakan *trade-off* yang masuk akal (beberapa alarm palsu, tapi tidak berlebihan, demi menangkap sebagian besar yang akan gagal).

## 2. Analisis Peringatan Dini / Temporal Early Warning (Eksperimen 2)

Eksperimen ini mengevaluasi apakah model Random Forest (dengan fitur S2) dapat memprediksi ketidaklulusan mahasiswa secara dini. 

**Tabel 2: Performa Prediksi Seiring Berjalannya Waktu**

| Cutoff | CV F1-Score | CI 95% F1 | CV Recall | Test F1-Score | Test Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Minggu 3 (M3)** | 0.618 ± 0.128 | [0.564, 0.672] | 0.659 | 0.636 | 0.700 |
| **Minggu 5 (M5)** | 0.644 ± 0.098 | [0.602, 0.686] | 0.723 | 0.608 | 0.700 |
| **Minggu 7 (M7)** | 0.661 ± 0.103 | [0.617, 0.704] | 0.723 | 0.500 | 0.500 |
| **Pre-Final** | 0.640 ± 0.107 | [0.595, 0.685] | 0.681 | 0.555 | 0.500 |

**Analisis Skenario 2:**
- **Momen Kritis:** Menariknya, prediksi pada Minggu ke-5 (M5) sudah sangat optimal, mencapai CV Recall **72.3%** dan Test Recall **70%**. 
- Ini membuktikan bahwa **asisten laboratorium bisa meluncurkan intervensi seawal M5**, mengingat menunda hingga M7 tidak memberikan kenaikan Recall yang berarti pada dataset tak kasat mata (Test Set).

## 3. Ketahanan Konteks / Context Robustness (Eksperimen 3)

Pendekatan *Leave-Group-Out* digunakan untuk menguji apakah fitur S2 kebal (robust) terhadap perbedaan aturan main dosen.

**Tabel 3: Generalisasi Skema Penilaian (Model: Random Forest S2)**

| Skenario Latih -> Uji | Test Accuracy | Test Recall (Belum Kompeten) | Test Precision | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Latih di AC -> Uji di BDE** | 0.615 | 0.969 | 0.563 | 0.713 |
| **Latih di BDE -> Uji di AC** | 0.672 | 0.053 | 1.000 | 0.100 |

**Analisis Skenario 3:**
- **Ketahanan Asimetris yang Ekstrem:** Model yang dilatih dengan standar pengumpulan ketat (AC) memiliki Recall yang sangat agresif (96.9%) saat dibawa ke kelas fleksibel (BDE). Artinya ia berhasil membunyikan alarm bagi hampir seluruh mahasiswa berisiko, dengan Precision 56.3% (tingkat toleransi *false positive* yang cukup baik).
- Sebaliknya, model yang dididik dalam budaya telat (BDE) menjadi sangat "pemaaf" dan buta terhadap standar AC, sehingga hanya mampu mendeteksi 5.3% mahasiswa gagal.
- **Nilai Novelty SINTA 2:** Data pendidikan heterogen **tidak boleh** sekadar digabung (pooled). Model prediksi kinerja akademik membawa "nilai dan standar" bawaan dari budaya kelas asalnya.

## 4. Penjelasan Model (TreeSHAP) (Eksperimen 4)

TreeSHAP (fokus pada prediksi Belum Kompeten) mengungkapkan mekanisme pengambilan keputusan. File lengkap dapat dilihat di folder `results/shap/`.

1. **Global Importance:**
   - Fitur terkait Laporan (*Laporan_Completion_Rate*, *Laporan_Mean*) tetap menjadi variabel dominan. Ini mencerminkan bahwa komitmen penulisan laporan praktikum adalah cerminan utama dari potensi kompetensi individu di akhir semester.

2. **Local Interpretability (Waterfall Plots):**
   - **True Positive (Tepat Prediksi Gagal):** Mahasiswa yang terdeteksi secara dini sering kali sudah menunjukkan pola bolong-bolong dalam pengumpulan *Laporan* sejak minggu ke-2 dan ke-3.
   - **False Negative (Gagal Memprediksi):** Kasus mahasiswa rajin hadir dan mengumpulkan tugas asal-asalan sering kali menipu model. Kehadiran fisik (yang terekam) tidak menjamin perolehan nilai final yang tinggi, menyebabkan prediksi *miss*.

## 5. Kesimpulan Publikasi

1. **Efektivitas Evaluasi Metrik:** Penggunaan optimasi hiperparameter berbasis *Recall (pos_label=0)* dan *balanced_subsample* sukses meningkatkan laju deteksi mahasiswa berisiko dari ~55% menjadi ~70%. Penurunan *Precision* akibat hal ini (berada di angka ~60%) sangat bisa ditoleransi dalam skema Sistem Peringatan Dini.
2. **Timing & Konteks:** Mitigasi optimal berada di **Minggu ke-5**, dan pengembangan sistem lintas-dosen di masa depan harus selalu menggunakan model yang dilatih pada himpunan data dengan kebijakan yang lebih ketat, guna menghindari fenomena kebutaan prediktif (seperti kasus BDE->AC).
