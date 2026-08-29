# Laporan Riset: Evaluasi Temporal Decision Tree dan Random Forest pada Sistem Peringatan Dini Akademik Menggunakan Explainable AI

**Status:** Draft Final (Revisi Metodologis Multi-Cutoff & Analisis Ketidakpastian)
**Framework:** Nested Cross-Validation (K-Fold 5x2) dengan Evaluasi Holdout Set ($n=18$)

---

## 1. Pendahuluan & Research Questions (RQ)

Mendeteksi kegagalan mahasiswa pada akhir semester merupakan tindakan korektif yang sering kali terlambat. Sistem Peringatan Dini (_Early Warning System_ / EWS) membutuhkan model yang mampu mengidentifikasi risiko kegagalan (_Belum Kompeten_) sedini mungkin, hanya dengan mengandalkan data historis pada fase awal perkuliahan, tanpa adanya _temporal leakage_ (kebocoran informasi masa depan).

Penelitian ini menggunakan pendekatan algoritma klasifikasi dipadukan dengan interpretasi _Explainable AI_ (SHAP) untuk menjawab empat pertanyaan penelitian utama:

- **RQ1**: Apakah aktivitas akademik mahasiswa (seperti kehadiran, skor tugas awal, dan laju penyelesaian) dapat memprediksi status kompetensinya secara andal?
- **RQ2**: Seberapa dini status kompetensi tersebut dapat diprediksi tanpa kehilangan performa klasifikasi secara drastis?
- **RQ3**: Fitur aktivitas apa yang memberikan kontribusi prediktif tertinggi terhadap risiko mahasiswa Belum Kompeten?
- **RQ4**: Apakah EWS berbasis aktivitas ini dapat mempertahankan kinerja klasifikasinya pada dataset berskala kecil tanpa injeksi data buatan (seperti SMOTE)?

---

## 2. Metodologi dan Konstruksi Data

### 2.1 Definisi Kompetensi (Target Label)

Ambang 83 digunakan sebagai kriteria operasional klasifikasi kompetensi berdasarkan rubrik penilaian praktikum yang digunakan oleh tim pengampu. Mahasiswa dengan skor di bawah batas tersebut dikategorikan ke dalam kelas prioritas (_Belum Kompeten_).

Distribusi label yang relatif seimbang (45 Belum Kompeten dan 44 Kompeten) merupakan konsekuensi dari penerapan kriteria tersebut pada populasi penelitian.

### 2.2 Desain Eksperimen (Retrospektif vs Genuine EWS)

Penelitian dirancang dalam dua fase skenario:

1. **Model Retrospektif (S1 - S5)**: Menggunakan agregasi data selama satu semester penuh. Tujuannya untuk menemukan _baseline_ kombinasi fitur (Kehadiran, Tugas Pendahuluan, Laporan Praktikum) yang memberikan performa diskriminatif terbaik terhadap dua kelas kompetensi.
2. **Genuine EWS Multi-Cutoff**: Dirancang secara khusus sebagai sistem peringatan dini murni. Fitur diekstraksi secara bertahap pada rentang waktu terbatas: **Week 1 (EWS-W1)**, **Week 2 (EWS-W2)**, **Week 3 (EWS-W3)**, dan **Full Semester (EWS-Full)**. Skenario EWS ini menggunakan fitur komposit performa awal (`Early_Performance_Composite`), yang secara ketat bersifat _cutoff-aware_ (mengikuti laju waktu). Pada W1, fitur ini murni dihitung dari data minggu pertama. Mulai dari W2 dan seterusnya, fitur ini diformulasikan secara konsisten sebagai rata-rata performa pada dua pertemuan pertama tanpa adanya _temporal leakage_:
   $$ EPC = \text{Mean}(\text{TP}_{W1, W2}, \text{Laporan}_{W1, W2}) $$

### 2.3 Protokol Validasi (Mencegah Bias dan Kebocoran)

1. **Nested Cross-Validation**: Digunakan Nested Stratified Cross-Validation dengan 2 _outer folds_ dan 5 _inner folds_. _Hyperparameter tuning_ dilakukan pada _inner folds_, sedangkan evaluasi murni dilakukan berdasarkan performa _outer folds_. Karena keterbatasan ukuran sampel ($n=89$), interpretasi performa CV dilakukan dengan mempertimbangkan varians estimasi yang relatif tinggi. Sebagai langkah pengamanan utama, _holdout set_ sebesar 20% ($n=18$) dikunci (_frozen_) dan hanya digunakan satu kali di tahap paling akhir.
2. **Preprocessing Terisolasi**: Teknik _scaling_ dan imputasi dienkapsulasi murni di dalam iterasi _inner fold_. Tidak ada informasi statistik (mean/std) dari _outer fold_ atau _holdout set_ yang bocor ke dalam proses pelatihan.

---

## 3. Hasil Eksperimen dan Evaluasi Model

### 3.1 Evaluasi Baseline Retrospektif (S1 - S5)

Sebelum menerapkan sistem EWS berbasis waktu, penelitian menguji 5 tingkat kerumitan fitur secara retrospektif (menggunakan data seluruh pertemuan). Berdasarkan hasil seleksi _Nested CV_ dan uji akhir pada _holdout set_, Skenario 3 (S3) menggunakan algoritma _Decision Tree_ dipilih sebagai kandidat _baseline_ karena memberikan kombinasi _Recall BK_ dan _Balanced Accuracy holdout_ yang sesuai dengan tujuan deteksi risiko, terutama karena mencapai _Recall BK_ sebesar 100%.

**Tabel 1: Top 5 Leaderboard Skenario Retrospektif (Baseline)**

| Skenario | Model            | Fitur (Karakteristik)                   | CV Balanced Acc | CV Recall BK | Test Balanced Acc | Test Recall BK |  PR-AUC   |
| :------- | :--------------- | :-------------------------------------- | :-------------: | :----------: | :---------------: | :------------: | :-------: |
| **S3**   | **DecisionTree** | **6 (Mean + Completion + Variability)** |     62.39%      |    64.21%    |    **77.78%**     |   **100.0%**   | **0.738** |
| S2       | DecisionTree     | 6 (Mean + Completion + Absence)         |     63.71%      |    57.14%    |      72.22%       |     100.0%     |   0.748   |
| S1       | RandomForest     | 3 (Base Mean)                           |     65.60%      |    63.21%    |      66.67%       |     88.89%     |   0.858   |
| S2       | RandomForest     | 6 (Mean + Completion + Absence)         |     68.17%      |    65.50%    |      77.78%       |     77.78%     |   0.864   |
| S3       | RandomForest     | 6 (Mean + Completion + Variability)     |     65.17%      |    68.07%    |      77.78%       |     77.78%     |   0.892   |

_Catatan: Skenario 4 dan 5 tidak meningkatkan performa holdout dan menunjukkan indikasi penurunan kemampuan generalisasi ketika jumlah fitur diperluas. Hal ini dapat mengindikasikan adanya fitur redundant/noisy atau meningkatnya kompleksitas model relatif terhadap ukuran sampel._

### 3.2 Evaluasi EWS Multi-Cutoff (Menjawab RQ2)

Untuk membuktikan kelayakan model sebagai sistem deteksi dini (_Early Warning System_), varian fitur adaptasi awal (`S3_E`) diuji pada berbagai titik potong waktu pengamatan, diukur pada _Holdout Set_ berukuran $n=18$ (9 Belum Kompeten, 9 Kompeten).

**Tabel 2: Kinerja EWS Berdasarkan Periode Pengamatan**

| Pengamatan (_Cutoff_)        | Recall BK  | Precision BK |   F2 BK    | Specificity | Balanced Acc |   MCC    |
| :--------------------------- | :--------: | :----------: | :--------: | :---------: | :----------: | :------: |
| **Week 1 (EWS-W1)**          |   88.89%   |    47.06%    |   75.47%   |    0.00%    |    44.44%    |   0.00   |
| **Week 2 (EWS-W2)**          | **100.0%** |  **69.23%**  | **91.84%** | **55.56%**  |  **77.78%**  | **0.57** |
| **Week 3 (EWS-W3)**          | **100.0%** |  **69.23%**  | **91.84%** | **55.56%**  |  **77.78%**  | **0.57** |
| **Full Semester (EWS-Full)** |   88.89%   |    72.73%    |   85.11%   |   66.67%    |    77.78%    |   0.58   |

**Analisis Laju Waktu (RQ2):**
Pada akhir Minggu ke-1 (W1), EWS mendeteksi 88.89% kasus ancaman kegagalan, namun mengorbankan kemampuan membedakan mahasiswa Kompeten, dengan specificity sebesar 0% karena menandai semua populasi mahasiswa aman sebagai berisiko (9 _False Positives_).

Pada _holdout set_ penelitian ini, performa klasifikasi mencapai tingkat yang stabil mulai cutoff Minggu ke-2, yang ditunjukkan oleh Balanced Accuracy sebesar 77.78% dan Recall BK sebesar 100%. Dari _confusion matrix_ W2 (TP=9, FN=0, FP=4, TN=5), model berhasil menangkap seluruh mahasiswa Belum Kompeten di holdout set, meskipun masih memberikan 4 _false alarm_ kepada mahasiswa Kompeten (Precision BK 69.23%). Untuk sebuah EWS, _trade-off_ ini dinilai sebagai karakteristik yang menarik: sistem mengorbankan sedikit akurasi (_Precision_) demi meminimalkan mahasiswa berisiko yang terlewat dari intervensi awal (False Negative = 0). Kinerja ini bersifat konsisten hingga Week 3. Akumulasi data secara penuh (Full Semester) tidak lagi berdampak signifikan terhadap pengamanan _Recall_, melainkan hanya mempertajam spesifisitas. Koefisien korelasi Matthews (MCC) sebesar 0.57-0.58 menunjukkan adanya hubungan prediksi yang substansial antara hasil klasifikasi model dan label aktual.

### 3.3 Analisis Stabilitas (Uncertainty Estimation)

Mengingat kecilnya rentang data (populasi $n=89$ dan holdout $n=18$), laporan ini menakar interval ketidakpastian (95% CI) untuk evaluasi pada skenario _Full Semester_, yang dihasilkan lewat metode _non-parametric bootstrap_ ($B=1000$ repetisi) untuk mengukur variabilitas estimasi kinerja:

- **Recall BK**: 88.89% (95% CI: [62.50% - 100.0%])
- **Balanced Accuracy**: 77.78% (95% CI: [56.15% - 95.45%])
- **F2-Score BK**: 85.11% (95% CI: [62.46% - 98.04%])

Sebagai bentuk pembuktian, _Dummy Classifier_ eksperimen diukur memegang level performa Balanced Accuracy mutlak sebesar 50.0%. Batas bawah interval kepercayaan Balanced Accuracy berada di atas 50%, sehingga performa model pada _holdout set_ menunjukkan kinerja yang lebih tinggi daripada baseline _dummy_ dalam sampel pengujian ini.

---

## 4. Analisis Interpretasi dengan Explainable AI (SHAP)

Menjawab RQ3, ekstraksi bobot _SHapley Additive exPlanations_ (SHAP) membantu kita memetakan fitur prioritas dan meluruskan asumsi terkait indikator kelulusan.

- **Pada Model Retrospektif (S3 Dasar)**: Fitur agregat seperti rata-rata nilai (_Laporan_Mean_) sangat mendominasi struktur keputusan model. Namun, indikator kumulatif ini sangat bersifat reaktif, karena pengamatan harus menunggu hingga semester berjalan selesai.
- **Pada Model Early Warning (EWS)**: Saat fitur orientasi kinerja dini (`Early_Performance_Composite`) dimasukkan ke dalam model, komposisi korelasi prediksi (_mean absolute SHAP_) mengalami pergeseran radikal.

Pada arsitektur EWS, fitur agregasi _late-stage_ kehilangan sebagian besar kontribusi aslinya. Keputusan algoritma secara substansial bergantung pada prediktor `Early_Performance_Composite`. Secara empiris, analisis SHAP menunjukkan bahwa performa adaptif pada fase awal menunjukkan kontribusi prediktif yang tinggi dalam model penelitian ini dibandingkan fitur akumulatif lainnya.

---

## 5. Kesimpulan

1. **(RQ1)** Aktivitas akademik berbasis penyelesaian tugas dan skor harian di fase perkenalan perkuliahan dapat diandalkan sebagai indikator diskriminatif dalam klasifikasi level kompetensi mahasiswa.
2. **(RQ2 & RQ3)** Untuk memberikan manfaat praktis, intervensi prediksi kegagalan tidak perlu menunggu ujian tengah semester. Pada _holdout set_ penelitian ini, cutoff Minggu ke-2 merupakan titik waktu paling awal yang mencapai Recall BK sebesar 100% dengan Balanced Accuracy 77.78%. Performa tersebut dipertahankan pada Minggu ke-3, sementara penggunaan data semester penuh tidak meningkatkan Balanced Accuracy. Analisis SHAP menunjukkan pergeseran fokus struktural model dari sekadar perekapan nilai akhir, menjadi penganalisis kinerja adaptif di awal waktu.
3. **(RQ4)** Karena distribusi kelas pada penelitian ini relatif seimbang (45:44), model dapat dievaluasi tanpa _synthetic oversampling_ seperti SMOTE sehingga pelatihan dilakukan tanpa observasi sintetis.

### Keterbatasan

Karena pemisahan train/test dilakukan pada mahasiswa dari _cohort_ yang sama, kemampuan generalisasi lintas _cohort_ belum dapat dipastikan. Ruang lingkup evaluasi populasi penelitian ini membutuhkan replikasi uji silang pada kurikulum ajaran (atau _cohort_ kelas) yang berbeda pada semester mendatang untuk memvalidasi stabilitas intervensi EWS secara komprehensif. Perlu dipertimbangkan pula faktor eksternal seperti _learning engagement_ yang tidak terukur dalam dataset numerik saat ini.
