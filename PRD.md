# PRD — Sistem Prediksi Kompetensi Mahasiswa Praktikum Logika Pemrograman

**Product Requirements Document (PRD)**  
**Versi:** 1.0  
**Basis:** Rancangan Penelitian _Prediksi Kompetensi Mahasiswa pada Praktikum Logika Pemrograman Menggunakan Random Forest dan Explainable Artificial Intelligence_  
**Platform pengembangan:** Python + Anaconda/Conda  
**Jenis proyek:** Research / Machine Learning Classification / Explainable AI

---

## 1. Ringkasan Produk

Produk yang dibangun adalah pipeline machine learning untuk memprediksi **kompetensi individual mahasiswa pada Praktikum Logika Pemrograman** berdasarkan rekam aktivitas dan performa mahasiswa selama proses praktikum.

Data yang digunakan mencakup kehadiran, Tugas Pendahuluan (TP), responsi, laporan/asistensi, dan Final Individu. **Final Individu digunakan hanya untuk membentuk label kompetensi**, sedangkan komponen penilaian sebelum final digunakan sebagai prediktor. Rancangan penelitian menetapkan:

- **Kompeten:** Final Individu `>= 75`
- **Belum Kompeten:** Final Individu `< 75`

Final Individu wajib dikeluarkan dari fitur setelah label dibuat agar tidak terjadi data leakage. Nilai `0` dan `40` tidak boleh dihapus secara otomatis karena dapat merupakan nilai akademik yang valid, misalnya tidak mengerjakan aktivitas atau penalti keterlambatan.

Model utama terdiri dari:

1. **Decision Tree** sebagai baseline.
2. **Random Forest** sebagai model pembanding/utama.
3. **TreeSHAP** untuk menjelaskan kontribusi fitur terhadap prediksi Random Forest.

Produk tidak dimaksudkan untuk membuktikan penyebab mahasiswa menjadi kompeten atau belum kompeten. Interpretasi SHAP harus selalu diposisikan sebagai **kontribusi fitur terhadap prediksi model**, bukan hubungan sebab-akibat.

---

## 2. Tujuan Produk

### 2.1 Tujuan utama

Membangun pipeline penelitian yang reproducible untuk:

1. Memvalidasi dan membersihkan data penilaian praktikum berdasarkan makna akademik.
2. Membentuk label kompetensi mahasiswa dari Final Individu.
3. Menghasilkan fitur prediktif yang tidak mengalami leakage.
4. Membandingkan Decision Tree dan Random Forest secara adil.
5. Menguji manfaat feature engineering melalui tiga skenario eksperimen.
6. Mengukur performa model menggunakan metrik klasifikasi.
7. Menjelaskan model Random Forest menggunakan TreeSHAP.
8. Menghasilkan artefak hasil eksperimen yang dapat digunakan langsung untuk analisis skripsi/artikel.

### 2.2 Tujuan non-fungsional

Pipeline harus:

- dapat dijalankan ulang dengan random seed yang sama;
- mencatat versi Python dan library;
- menyimpan konfigurasi eksperimen;
- memisahkan data mentah, data olahan, model, hasil evaluasi, dan visualisasi;
- mencegah penggunaan test set saat tuning atau feature selection;
- mudah dilacak dari data → feature engineering → model → evaluasi → XAI.

---

## 3. Ruang Lingkup

### 3.1 In scope

Produk mencakup:

- ingestion dataset CSV/XLSX;
- validasi struktur dataset;
- pemeriksaan missing value;
- pemeriksaan rentang dan tipe nilai;
- pemahaman nilai 0 dan 40 berdasarkan aturan penilaian;
- pembentukan label kompetensi;
- pencegahan leakage;
- feature engineering;
- tiga skenario fitur S1, S2, S3;
- stratified train-test split 80:20;
- 5-fold stratified cross-validation pada training set;
- Decision Tree;
- Random Forest;
- hyperparameter tuning yang wajar;
- Accuracy, Precision, Recall, F1-Score;
- confusion matrix;
- mean ± standard deviation cross-validation;
- TreeSHAP global;
- TreeSHAP local;
- analisis distribusi per kelas A/B/C/D;
- export hasil dalam CSV/JSON/PNG;
- reproducibility melalui konfigurasi dan environment Conda.

### 3.2 Out of scope

Hal berikut tidak menjadi bagian inti versi pertama:

- kuesioner sebagai prediktor utama;
- Final Kelompok sebagai target utama;
- Total sebagai fitur utama;
- NIM atau Nama sebagai fitur prediktif;
- model deep learning;
- membuat empat model utama terpisah untuk kelas A, B, C, dan D;
- sistem intervensi otomatis kepada mahasiswa;
- klaim kausalitas;
- deployment produksi berskala besar.

---

## 4. Pengguna Produk

### 4.1 Peneliti / mahasiswa

Membutuhkan pipeline untuk menjalankan eksperimen, membaca hasil, membandingkan model, dan menghasilkan visualisasi penelitian.

### 4.2 Dosen pembimbing / reviewer

Membutuhkan hasil eksperimen yang dapat ditelusuri, termasuk distribusi data, metode validasi, performa model, dan interpretasi SHAP.

### 4.3 Pengembang

Membutuhkan struktur kode modular agar preprocessing, feature engineering, training, evaluation, dan XAI dapat diuji dan dikembangkan tanpa mengubah bagian lain.

---

## 5. Dasar Metodologis yang Wajib Dipertahankan

Persyaratan berikut berasal langsung dari rancangan penelitian dan tidak boleh diubah tanpa alasan metodologis yang terdokumentasi.

### 5.1 Unit analisis

Satu baris data mewakili **satu mahasiswa**.

### 5.2 Target

```text
Y = 1 jika Final_Individu >= 75
Y = 0 jika Final_Individu < 75
```

Label harus dibuat sebelum Final Individu dihapus dari matriks fitur.

### 5.3 Leakage prevention

Kolom berikut tidak boleh masuk ke `X`:

- `Final_Individu`;
- `Total`;
- identifier seperti `NIM` dan `Nama`;
- variabel lain yang dihitung menggunakan Final Individu atau informasi setelah final.

### 5.4 Nilai 0 dan 40

- `0` dapat berarti mahasiswa tidak melakukan aktivitas atau tidak mengumpulkan tugas.
- `40` dapat merupakan penalti keterlambatan.
- Keduanya tidak boleh diubah menjadi missing value atau dihapus hanya karena rendah.
- Validasi harus menggunakan aturan penilaian akademik yang berlaku.

### 5.5 Validasi

Gunakan:

- stratified train-test split `80:20`;
- 5-fold stratified cross-validation pada training set;
- test set disimpan untuk evaluasi final.

### 5.6 Model utama

- Decision Tree = baseline.
- Random Forest = model utama/pembanding.

### 5.7 XAI

TreeSHAP diterapkan setelah model utama dipilih, terutama pada Random Forest, dengan:

- global feature importance;
- beeswarm plot;
- local explanation pada beberapa mahasiswa.

---

## 6. Asumsi Struktur Dataset

Rancangan penelitian menyebut dataset sekitar **±150 mahasiswa**, namun jumlah final harus ditentukan setelah validasi dan kriteria inklusi/exklusi.

Contoh kelompok kolom yang diharapkan:

| Kelompok          | Contoh isi                 | Peran                                         |
| ----------------- | -------------------------- | --------------------------------------------- |
| Identitas         | `NIM`, `Nama`              | Metadata, bukan fitur                         |
| Kelas             | `Kelas`                    | Metadata analisis                             |
| Kehadiran         | pertemuan 1..n             | Sumber `Attendance_Rate`                      |
| TP                | TP 1..n                    | Sumber `TP_Mean`, completion                  |
| Responsi          | Responsi 1..n              | Sumber `Respons_Mean`, completion             |
| Laporan/Asistensi | komponen laporan/asistensi | Sumber `Laporan_Mean`, completion             |
| Final Individu    | `Final_Individu`           | Pembentuk target, kemudian dikeluarkan dari X |
| Final Kelompok    | `Final_Kelompok`           | Tidak digunakan sebagai target utama          |
| Total             | `Total`                    | Dikeluarkan dari X                            |

Implementasi **tidak boleh mengasumsikan nama kolom persis seperti contoh**. Gunakan file konfigurasi mapping kolom agar struktur dataset aktual dapat disesuaikan tanpa memodifikasi kode inti.

---

## 7. Persyaratan Data Ingestion

### 7.1 Format input

Versi pertama menerima:

- CSV (`.csv`);
- Excel (`.xlsx`).

### 7.2 Konfigurasi sumber data

Gunakan file seperti `configs/data_config.yaml` untuk mendefinisikan:

- lokasi dataset;
- nama sheet Excel jika diperlukan;
- nama kolom Final Individu;
- kolom identitas;
- kolom kelas;
- daftar kolom kehadiran;
- daftar kolom TP;
- daftar kolom responsi;
- daftar kolom laporan/asistensi.

### 7.3 Validasi awal

Pipeline harus memeriksa:

1. file tersedia;
2. dataset dapat dibaca;
3. header tersedia;
4. jumlah baris > 0;
5. Final Individu tersedia;
6. tidak ada dua mahasiswa identik tanpa penjelasan;
7. tipe nilai numerik sesuai kebutuhan;
8. nilai berada pada rentang yang sesuai aturan penilaian;
9. struktur kolom aktivitas dapat dipetakan.

Jika validasi gagal, pipeline harus berhenti dengan pesan error yang jelas.

---

## 8. Data Cleaning dan Data Understanding

Cleaning harus didasarkan pada **makna akademik**, bukan hanya aturan statistik.

### 8.1 Pemeriksaan yang wajib

- jumlah mahasiswa;
- jumlah kolom;
- tipe data;
- missing value per kolom;
- distribusi Final Individu;
- distribusi label;
- distribusi nilai TP;
- distribusi responsi;
- distribusi laporan/asistensi;
- distribusi kehadiran;
- duplikasi;
- nilai di luar rentang yang diperbolehkan;
- jumlah mahasiswa per kelas.

### 8.2 Missing value

Setiap missing value harus diklasifikasikan sebagai salah satu:

- memang tidak ada data;
- aktivitas tidak dilakukan dan secara akademik direpresentasikan dengan nilai tertentu;
- kesalahan input;
- tidak berlaku untuk mahasiswa/kelas tertentu.

Jangan mengubah missing menjadi 0 tanpa dasar aturan akademik.

### 8.3 Nilai valid

Nilai `0`, `40`, `50`, dan nilai lainnya harus dipertahankan apabila memang merupakan hasil penilaian yang sah.

---

## 9. Pembentukan Label Kompetensi

Tahap ini harus dilakukan setelah validasi data Final Individu.

### 9.1 Aturan

| Final Individu | Label numerik | Label teks     |
| -------------: | ------------: | -------------- |
|          >= 75 |             1 | Kompeten       |
|           < 75 |             0 | Belum Kompeten |

### 9.2 Edge case

- Final Individu tepat `75` → **Kompeten**.
- Final Individu kosong/tidak valid → tidak dapat membentuk label; baris harus diperiksa dan dikeluarkan dari modelling jika tidak dapat diperbaiki berdasarkan sumber resmi.

### 9.3 Output

Tambahkan:

```text
Competency_Label
Competency_Name
```

Kemudian drop:

```text
Final_Individu
```

dari fitur modelling.

---

## 10. Feature Engineering

Feature engineering dibagi menjadi tiga kelompok eksperimen.

### 10.1 S1 — Basic Features

| Kode | Nama              | Formula / definisi                             |
| ---- | ----------------- | ---------------------------------------------- |
| X1   | `Attendance_Rate` | jumlah skor kehadiran / jumlah pertemuan × 100 |
| X2   | `TP_Mean`         | rata-rata seluruh nilai TP                     |
| X3   | `Respons_Mean`    | rata-rata seluruh nilai responsi               |
| X4   | `Laporan_Mean`    | rata-rata nilai laporan/asistensi              |

Catatan: bila skala kehadiran berupa `1`, `0.5`, `0`, hitung sesuai makna skala yang sudah dijelaskan pada data.

### 10.2 S2 — Behavioral Features

S2 menambahkan fitur kontinuitas aktivitas:

| Kode | Nama                      | Definisi                                                                                       |
| ---- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| X5   | `TP_Completion_Rate`      | proporsi TP yang benar-benar dikumpulkan/dikerjakan berdasarkan status yang dapat diverifikasi |
| X6   | `Respons_Completion_Rate` | proporsi responsi yang benar-benar dilakukan berdasarkan catatan                               |
| X7   | `Laporan_Completion_Rate` | proporsi komponen laporan/asistensi yang diselesaikan                                          |

#### Aturan completion rate

Completion rate **tidak boleh** disimpulkan hanya dari besar kecilnya nilai tanpa bukti bahwa nilai tersebut mewakili status pengerjaan.

Nilai `40` tidak otomatis menjadi tidak selesai. Status harus mengikuti data dan aturan akademik.

### 10.3 S3 — Relational Features

S3 menambahkan:

| Kode | Nama             | Formula                  |
| ---- | ---------------- | ------------------------ |
| X8   | `Respons_TP_Gap` | `Respons_Mean - TP_Mean` |

Fitur ini hanya merepresentasikan **selisih performa** dua bentuk evaluasi.

Dilarang menafsirkan `Respons_TP_Gap` sebagai:

- bukti kecurangan;
- bukti ketidakjujuran;
- integritas mahasiswa;
- penyebab kompetensi.

---

## 11. Feature Registry

Buat satu modul terpusat agar fitur dapat dilacak.

Contoh struktur:

```python
FEATURE_REGISTRY = {
    "S1": [
        "Attendance_Rate",
        "TP_Mean",
        "Respons_Mean",
        "Laporan_Mean",
    ],
    "S2": [
        "Attendance_Rate",
        "TP_Mean",
        "Respons_Mean",
        "Laporan_Mean",
        "TP_Completion_Rate",
        "Respons_Completion_Rate",
        "Laporan_Completion_Rate",
    ],
    "S3": [
        "Attendance_Rate",
        "TP_Mean",
        "Respons_Mean",
        "Laporan_Mean",
        "TP_Completion_Rate",
        "Respons_Completion_Rate",
        "Laporan_Completion_Rate",
        "Respons_TP_Gap",
    ],
}
```

Tujuan feature registry adalah menjamin bahwa S1/S2/S3 dapat dijalankan ulang tanpa perubahan manual pada kode training.

---

## 12. Data Splitting dan Cross-Validation

### 12.1 Train-test split

Gunakan:

```text
80% training
20% test
```

dengan stratifikasi berdasarkan `Competency_Label`.

### 12.2 Random seed

Semua eksperimen harus memiliki seed terpusat, misalnya:

```python
RANDOM_STATE = 42
```

Nilai seed boleh diganti, tetapi harus konsisten dan terdokumentasi.

### 12.3 Cross-validation

Gunakan:

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

Cross-validation dijalankan **hanya pada training set**.

### 12.4 Test set

Test set:

- tidak boleh digunakan untuk tuning;
- tidak boleh digunakan untuk memilih fitur berdasarkan hasil test;
- hanya digunakan sebagai evaluasi akhir setelah desain eksperimen ditetapkan.

---

## 13. Pipeline Anti-Leakage

Semua langkah yang belajar dari data harus berada di dalam pipeline training/CV, jika digunakan.

Contoh langkah yang harus berada di pipeline bila memang diperlukan:

- imputasi;
- scaling;
- feature selection;
- tuning;
- transformasi lain yang mempelajari parameter dari data.

Untuk Decision Tree dan Random Forest, scaling numerik umumnya tidak diperlukan, tetapi arsitektur pipeline tetap harus disiapkan agar preprocessing dapat dilakukan secara benar.

---

## 14. Model Decision Tree

Decision Tree berfungsi sebagai baseline yang mudah diinterpretasikan.

### 14.1 Tujuan

- baseline performa;
- menghasilkan aturan keputusan;
- menjadi pembanding langsung terhadap Random Forest.

### 14.2 Parameter yang dapat dituning

Minimum set yang disarankan:

- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `class_weight` bila diperlukan.

Tuning harus tetap sederhana karena ukuran dataset sekitar ±150 mahasiswa.

---

## 15. Model Random Forest

Random Forest adalah model utama karena diharapkan lebih stabil dan mampu menangkap pola non-linear serta interaksi fitur.

### 15.1 Parameter yang dapat dipertimbangkan

- `n_estimators`;
- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `max_features`;
- `class_weight`.

### 15.2 Prinsip tuning

Jangan melakukan grid search sangat luas. Dataset kecil meningkatkan risiko overfitting terhadap validation set.

Gunakan pencarian parameter yang wajar dan terdokumentasi.

Contoh pendekatan:

```text
GridSearchCV atau RandomizedSearchCV
scoring utama = F1 atau recall kelas yang diprioritaskan
cv = Stratified 5-Fold
n_jobs = -1
```

Pemilihan `scoring` utama harus ditetapkan sebelum melihat hasil akhir.

---

## 16. Penentuan Metrik Utama

Metrik minimal yang wajib dilaporkan:

- Accuracy;
- Precision;
- Recall;
- F1-Score;
- Confusion Matrix.

### 16.1 Interpretasi

**Accuracy**  
Proporsi prediksi benar terhadap seluruh observasi.

**Precision**  
Ketepatan ketika model memprediksi kelas tertentu sebagai positif.

**Recall**  
Kemampuan model menemukan anggota kelas target.

**F1-Score**  
Harmonic mean antara precision dan recall.

**Confusion Matrix**  
Menunjukkan TP, TN, FP, dan FN.

### 16.2 Prioritas kelas

Karena tujuan praktis dapat mengarah pada identifikasi mahasiswa yang berpotensi belum kompeten lebih awal, recall untuk kelas **Belum Kompeten** perlu diperhatikan secara khusus.

Implementasi harus mendukung evaluasi per kelas dan tidak hanya mengandalkan satu angka agregat.

---

## 17. Evaluasi Cross-Validation

Untuk setiap kombinasi:

- S1 + Decision Tree;
- S1 + Random Forest;
- S2 + Decision Tree;
- S2 + Random Forest;
- S3 + Decision Tree;
- S3 + Random Forest;

simpan:

```text
mean_accuracy
std_accuracy
mean_precision
std_precision
mean_recall
std_recall
mean_f1
std_f1
```

Hasil utama harus dilaporkan dalam bentuk **mean ± standard deviation** dari 5-fold CV pada training set.

---

## 18. Evaluasi Test Set

Setelah model dan konfigurasi diputuskan berdasarkan training + CV:

1. fit ulang model terbaik pada seluruh training set;
2. prediksi test set;
3. hitung metrik final;
4. simpan confusion matrix;
5. simpan classification report;
6. simpan probabilitas prediksi bila tersedia.

Jangan melakukan tuning ulang berdasarkan hasil test.

---

## 19. Desain Eksperimen

### 19.1 Matriks eksperimen utama

| Skenario | Fitur                           | Decision Tree | Random Forest |
| -------- | ------------------------------- | ------------- | ------------- |
| S1       | Basic                           | Ya            | Ya            |
| S2       | Basic + Behavioral              | Ya            | Ya            |
| S3       | Basic + Behavioral + Relational | Ya            | Ya            |

Total kombinasi utama: **6 eksperimen model**.

### 19.2 Tabel hasil yang diharapkan

| Skenario | Model         | CV Accuracy | CV Precision | CV Recall | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
| -------- | ------------- | ----------: | -----------: | --------: | ----: | ------------: | -------------: | ----------: | ------: |
| S1       | Decision Tree |           — |            — |         — |     — |             — |              — |           — |       — |
| S1       | Random Forest |           — |            — |         — |     — |             — |              — |           — |       — |
| S2       | Decision Tree |           — |            — |         — |     — |             — |              — |           — |       — |
| S2       | Random Forest |           — |            — |         — |     — |             — |              — |           — |       — |
| S3       | Decision Tree |           — |            — |         — |     — |             — |              — |           — |       — |
| S3       | Random Forest |           — |            — |         — |     — |             — |              — |           — |       — |

---

## 20. Kriteria Model Terbaik

Model terbaik tidak boleh ditentukan berdasarkan asumsi bahwa Random Forest pasti menang.

Aturan pemilihan harus ditetapkan sebelum melihat hasil akhir. Rekomendasi:

1. gunakan hasil 5-fold CV sebagai dasar utama;
2. perhatikan F1 dan recall kelas prioritas;
3. lihat mean dan standard deviation;
4. gunakan test set sebagai verifikasi akhir;
5. jika selisih sangat kecil, pertimbangkan kestabilan dan interpretabilitas;
6. dokumentasikan alasan pemilihan secara eksplisit.

Jika Decision Tree lebih baik, hasil tersebut tetap valid.

Jika feature engineering tidak meningkatkan performa, hal tersebut juga merupakan hasil penelitian.

---

## 21. Explainable AI — TreeSHAP

### 21.1 Kapan dijalankan

TreeSHAP dilakukan setelah model utama/best model ditetapkan, dengan fokus utama pada Random Forest.

### 21.2 Global explanation

Minimal hasilkan:

1. **Global SHAP feature importance**
2. **SHAP beeswarm plot**

Tujuan:

- mengetahui fitur paling berkontribusi;
- melihat arah kontribusi;
- memahami penyebaran kontribusi fitur antar mahasiswa.

### 21.3 Local explanation

Pilih beberapa kasus representatif, misalnya:

- satu mahasiswa diprediksi **Kompeten**;
- satu mahasiswa diprediksi **Belum Kompeten**.

Bila memungkinkan, tambahkan kasus:

- prediksi benar pada probabilitas tinggi;
- kasus yang sulit / near-boundary;
- false positive atau false negative yang informatif.

### 21.4 Aturan interpretasi

Gunakan istilah:

> "Fitur X memberikan kontribusi positif/negatif terhadap prediksi model pada kasus ini."

Jangan gunakan:

> "Fitur X menyebabkan mahasiswa menjadi kompeten."

SHAP bukan bukti kausalitas.

---

## 22. Analisis Heterogenitas Kelas

Kelas A–D diperlakukan sebagai metadata/konteks.

Analisis minimal:

- jumlah mahasiswa per kelas;
- proporsi Kompeten/Belum Kompeten per kelas;
- rata-rata TP per kelas;
- rata-rata responsi per kelas;
- rata-rata laporan/asistensi per kelas;
- rata-rata kehadiran per kelas;
- perbedaan materi;
- perbedaan dosen/asisten jika tersedia.

### 22.1 Informasi konteks

Menurut rancangan penelitian:

- Kelas A dan C mencakup logika, logika Excel, C++, dan Scratch/pemrograman berbasis blok.
- Kelas B dan D lebih berfokus pada logika pemrograman dan C++.
- Terdapat perbedaan dosen dan asisten.

Jangan membuat empat model terpisah sebagai eksperimen utama karena ukuran sampel per kelas dapat menjadi terlalu kecil.

---

## 23. Persyaratan Visualisasi

Pipeline minimal harus menghasilkan visual berikut:

### Data understanding

- distribusi label kompetensi;
- distribusi nilai Final Individu;
- distribusi fitur utama;
- missing value summary;
- distribusi mahasiswa per kelas.

### Model evaluation

- confusion matrix Decision Tree;
- confusion matrix Random Forest;
- perbandingan metrik S1/S2/S3;
- grafik mean ± std CV bila relevan.

### XAI

- SHAP bar plot / global importance;
- SHAP beeswarm;
- local SHAP waterfall/bar plot.

Semua gambar harus disimpan sebagai file agar dapat digunakan kembali pada skripsi/artikel.

---

## 24. Struktur Direktori Proyek

Struktur yang direkomendasikan:

```text
project-root/
├── README.md
├── PRD.md
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── data_config.yaml
│   ├── experiment_config.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/
│   │   └── dataset_asli.xlsx
│   ├── interim/
│   └── processed/
│       ├── cleaned_dataset.csv
│       ├── featured_S1.csv
│       ├── featured_S2.csv
│       └── featured_S3.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_treeshap_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── labeling.py
│   ├── feature_engineering.py
│   ├── feature_registry.py
│   ├── split.py
│   ├── models.py
│   ├── tuning.py
│   ├── evaluation.py
│   ├── experiments.py
│   ├── shap_analysis.py
│   ├── class_analysis.py
│   └── utils.py
│
├── scripts/
│   ├── validate_data.py
│   ├── build_features.py
│   ├── run_experiments.py
│   └── generate_shap.py
│
├── models/
│   ├── s1/
│   ├── s2/
│   └── s3/
│
├── results/
│   ├── metrics/
│   ├── cv/
│   ├── confusion_matrix/
│   ├── predictions/
│   ├── feature_importance/
│   └── shap/
│
├── reports/
│   ├── tables/
│   ├── figures/
│   └── logs/
│
└── tests/
    ├── test_labeling.py
    ├── test_feature_engineering.py
    ├── test_leakage.py
    └── test_evaluation.py
```

---

## 25. Pembagian Tanggung Jawab Modul

### `data_loader.py`

- membaca CSV/XLSX;
- normalisasi nama/struktur kolom jika dikonfigurasi;
- mengembalikan DataFrame.

### `data_validation.py`

- validasi kolom;
- validasi tipe;
- validasi missing;
- validasi rentang;
- validasi duplikat;
- laporan data quality.

### `labeling.py`

- membuat `Competency_Label`;
- membuat label teks;
- menghapus Final Individu dari X.

### `feature_engineering.py`

- `Attendance_Rate`;
- `TP_Mean`;
- `Respons_Mean`;
- `Laporan_Mean`;
- completion rates;
- `Respons_TP_Gap`;
- membangun S1/S2/S3.

### `models.py`

- konstruktor Decision Tree;
- konstruktor Random Forest;
- parameter default.

### `tuning.py`

- CV;
- pencarian hyperparameter;
- penyimpanan best parameters;
- penyimpanan hasil tuning.

### `evaluation.py`

- cross-validation metrics;
- test metrics;
- confusion matrix;
- classification report;
- agregasi mean ± std.

### `experiments.py`

- menjalankan enam kombinasi utama;
- memastikan split/CV identik sesuai konfigurasi;
- menyimpan hasil secara terstruktur.

### `shap_analysis.py`

- TreeSHAP explainer;
- global feature importance;
- beeswarm;
- local explanation;
- export nilai SHAP.

### `class_analysis.py`

- statistik per kelas;
- proporsi label;
- ringkasan fitur;
- output tabel untuk pembahasan.

---

## 26. Environment Anaconda

### 26.1 Tujuan

Semua eksperimen harus dijalankan dalam environment Conda khusus agar library penelitian tidak bercampur dengan environment global.

### 26.2 Nama environment

Rekomendasi:

```text
ml-kompetensi
```

### 26.3 Pembuatan environment

```bash
conda create -n ml-kompetensi python=3.12 -y
conda activate ml-kompetensi
```

### 26.4 Paket inti

Paket utama yang dibutuhkan:

```bash
conda install -c conda-forge pandas numpy scipy scikit-learn matplotlib seaborn openpyxl pyyaml joblib jupyterlab ipykernel -y
pip install shap
```

Catatan: versi final setiap library harus dikunci setelah implementasi awal stabil. `environment.yml` menjadi sumber reproduksi environment.

### 26.5 Registrasi kernel Jupyter

```bash
python -m ipykernel install --user --name ml-kompetensi --display-name "Python (ml-kompetensi)"
```

### 26.6 Verifikasi environment

```bash
python --version
conda list
python -c "import pandas, numpy, sklearn, shap; print('Environment OK')"
```

---

## 27. Contoh `environment.yml`

```yaml
name: ml-kompetensi
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.12
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - openpyxl
  - pyyaml
  - joblib
  - jupyterlab
  - ipykernel
  - pip
  - pip:
      - shap
```

Setelah environment stabil, export versi aktual:

```bash
conda env export --no-builds > environment.lock.yml
```

Untuk penelitian, simpan file lock tersebut bersama kode agar versi library dapat ditelusuri.

---

## 28. Konfigurasi Data

Contoh `configs/data_config.yaml`:

```yaml
data:
  path: data/raw/dataset_asli.xlsx
  sheet_name: Sheet1

columns:
  id:
    - NIM
    - Nama

  metadata:
    - Kelas

  final_individual: Final_Individu
  final_group: Final_Kelompok
  total: Total

  attendance:
    - Kehadiran_1
    - Kehadiran_2
    - Kehadiran_3

  tp:
    - TP_1
    - TP_2
    - TP_3

  respons:
    - Responsi_1
    - Responsi_2
    - Responsi_3

  laporan:
    - Laporan_1
    - Laporan_2
    - Laporan_3
```

Nama kolom pada contoh hanyalah template dan harus disesuaikan dengan dataset sebenarnya.

---

## 29. Konfigurasi Eksperimen

Contoh `configs/experiment_config.yaml`:

```yaml
random_state: 42

split:
  test_size: 0.20
  stratify: true

cv:
  strategy: StratifiedKFold
  n_splits: 5
  shuffle: true

scoring:
  primary: f1
  secondary:
    - recall
    - precision
    - accuracy

experiments:
  - S1
  - S2
  - S3

models:
  - DecisionTree
  - RandomForest
```

Catatan: metrik utama dapat diarahkan ke recall kelas Belum Kompeten jika strategi penelitian menetapkan identifikasi mahasiswa berisiko sebagai prioritas. Keputusan ini harus ditetapkan sebelum melihat hasil model.

---

## 30. Alur Eksekusi Sistem

Alur teknis wajib mengikuti urutan:

```text
Raw Dataset
   ↓
Load Data
   ↓
Validate Structure & Academic Rules
   ↓
Data Understanding
   ↓
Clean Data
   ↓
Create Competency Label
   ↓
Drop Final Individu / Total / Identifier from X
   ↓
Feature Engineering
   ↓
Build S1 / S2 / S3
   ↓
Train-Test Split 80:20 Stratified
   ↓
5-Fold Stratified CV on Train
   ↓
Decision Tree + Random Forest
   ↓
Model Comparison
   ↓
Final Test Evaluation
   ↓
Select Model
   ↓
TreeSHAP Global
   ↓
TreeSHAP Local
   ↓
Class Heterogeneity Analysis
   ↓
Export Tables / Figures / Logs
```

---

## 31. Alur Eksperimen yang Direkomendasikan

### Tahap A — Data validation

Output:

```text
results/data_quality/data_quality_report.json
results/data_quality/data_quality_report.csv
```

### Tahap B — Feature generation

Output:

```text
data/processed/featured_S1.csv
data/processed/featured_S2.csv
data/processed/featured_S3.csv
```

### Tahap C — Model comparison

Jalankan enam kombinasi model/skenario.

Output:

```text
results/metrics/model_comparison.csv
results/cv/cv_summary.csv
results/predictions/*.csv
```

### Tahap D — Final test evaluation

Simpan:

```text
results/metrics/final_test_metrics.csv
results/confusion_matrix/*.png
results/predictions/final_predictions.csv
```

### Tahap E — TreeSHAP

Output:

```text
results/shap/shap_values.csv
results/shap/global_importance.csv
results/shap/beeswarm.png
results/shap/local_case_*.png
```

### Tahap F — Class analysis

Output:

```text
results/class_analysis/class_summary.csv
results/class_analysis/class_distribution.png
```

---

## 32. Format Artefak Hasil

### 32.1 `model_comparison.csv`

Kolom minimal:

```text
scenario
model
cv_accuracy_mean
cv_accuracy_std
cv_precision_mean
cv_precision_std
cv_recall_mean
cv_recall_std
cv_f1_mean
cv_f1_std
test_accuracy
test_precision
test_recall
test_f1
best_params
```

### 32.2 `final_predictions.csv`

Kolom minimal:

```text
row_id
true_label
predicted_label
probability_competent
probability_not_competent
correct
```

Identifier pribadi seperti NIM/Nama tidak perlu diekspor ke hasil modelling kecuali secara khusus diperlukan untuk penelusuran internal. Bila diperlukan untuk analisis lokal, gunakan ID pseudonim.

### 32.3 SHAP result

Minimal:

```text
sample_id
feature
feature_value
shap_value
predicted_class
```

---

## 33. Logging dan Reproducibility

Setiap run harus menyimpan metadata:

```text
run_id
run_timestamp
python_version
platform
random_state
train_size
test_size
cv_folds
scenario
model
hyperparameters
feature_list
metrics
```

Contoh:

```json
{
  "run_id": "S3_RF_2026xxxx",
  "random_state": 42,
  "scenario": "S3",
  "model": "RandomForest",
  "cv_folds": 5,
  "test_size": 0.2
}
```

Tujuannya agar angka pada tabel penelitian dapat ditelusuri kembali ke konfigurasi yang menghasilkan angka tersebut.

---

## 34. Pengujian Kode

### 34.1 Unit test label

Kasus minimum:

```text
74.99 -> 0
75.00 -> 1
75.01 -> 1
```

### 34.2 Unit test leakage

Pastikan:

```text
Final_Individu not in X
Total not in X
NIM not in X
Nama not in X
```

### 34.3 Unit test feature engineering

Verifikasi formula:

- `Attendance_Rate`;
- `TP_Mean`;
- `Respons_Mean`;
- `Laporan_Mean`;
- completion rate;
- `Respons_TP_Gap`.

### 34.4 Unit test split

Pastikan:

- train dan test tidak overlap;
- proporsi label tetap masuk akal;
- test tidak dipakai pada CV.

---

## 35. Acceptance Criteria

Produk dianggap selesai untuk tahap penelitian apabila seluruh kondisi berikut terpenuhi.

### Data

- [ ] Dataset asli dapat dibaca otomatis.
- [ ] Struktur kolom tervalidasi.
- [ ] Jumlah mahasiswa final tercatat.
- [ ] Missing value dianalisis.
- [ ] Nilai 0 dan 40 ditangani sesuai makna akademik.
- [ ] Label `>=75` telah diverifikasi.

### Leakage

- [ ] Final Individu tidak masuk X.
- [ ] Total tidak masuk X.
- [ ] NIM/Nama tidak masuk X.
- [ ] Tidak ada transformasi yang fit menggunakan test set.

### Feature engineering

- [ ] S1 berhasil dibuat.
- [ ] S2 berhasil dibuat.
- [ ] S3 berhasil dibuat.
- [ ] Feature registry terdokumentasi.

### Model

- [ ] Decision Tree berhasil dilatih.
- [ ] Random Forest berhasil dilatih.
- [ ] 5-fold stratified CV berjalan.
- [ ] Hyperparameter tercatat.
- [ ] Random seed tercatat.

### Evaluasi

- [ ] Accuracy tersedia.
- [ ] Precision tersedia.
- [ ] Recall tersedia.
- [ ] F1 tersedia.
- [ ] Confusion matrix tersedia.
- [ ] Mean ± SD CV tersedia.
- [ ] Test-set result tersedia.

### XAI

- [ ] TreeSHAP global tersedia.
- [ ] Beeswarm plot tersedia.
- [ ] Local SHAP tersedia.
- [ ] Interpretasi menggunakan istilah kontribusi model.

### Analisis penelitian

- [ ] Perbandingan S1/S2/S3 tersedia.
- [ ] Perbandingan DT/RF tersedia.
- [ ] Distribusi kelas A/B/C/D tersedia.
- [ ] Konteks materi dan pengajar dicatat bila datanya tersedia.
- [ ] Keterbatasan dataset dibahas.

### Reproducibility

- [ ] `environment.yml` tersedia.
- [ ] versi library dapat ditelusuri.
- [ ] random seed tersimpan.
- [ ] konfigurasi eksperimen tersimpan.
- [ ] semua output memiliki nama file konsisten.

---

## 36. Risiko Teknis dan Mitigasi

| Risiko                    | Dampak                        | Mitigasi                                                            |
| ------------------------- | ----------------------------- | ------------------------------------------------------------------- |
| Dataset terlalu kecil     | performa tidak stabil         | stratified CV, model sederhana, tuning wajar                        |
| Class imbalance           | recall salah satu kelas buruk | laporkan distribusi, evaluasi per kelas, pertimbangkan class_weight |
| Leakage Final Individu    | performa palsu tinggi         | drop setelah label dibuat + unit test                               |
| Leakage Total             | model melihat hasil agregat   | drop Total                                                          |
| Nilai 0 dianggap missing  | informasi aktivitas hilang    | validasi makna akademik                                             |
| Nilai 40 dianggap outlier | penalti valid hilang          | pertahankan jika sesuai aturan                                      |
| Overfitting tuning        | generalisasi buruk            | CV pada training saja, search terbatas                              |
| Heterogenitas kelas       | interpretasi bias konteks     | analisis kelas sebagai metadata                                     |
| Terlalu banyak fitur      | noise                         | gunakan S1–S3 bertahap                                              |
| Terlalu banyak model      | fokus penelitian melebar      | inti hanya DT vs RF                                                 |
| Salah tafsir SHAP         | klaim kausal                  | gunakan bahasa kontribusi prediksi                                  |
| Test-set leakage          | estimasi performa tidak valid | lock test set sampai evaluasi final                                 |

---

## 37. Definition of Done untuk Penelitian

Pipeline penelitian dianggap siap untuk menghasilkan angka yang boleh dimasukkan ke skripsi/artikel apabila:

1. data telah divalidasi terhadap aturan penilaian resmi;
2. label kompetensi diverifikasi;
3. leakage test lulus;
4. S1/S2/S3 dapat dibuat ulang;
5. enam eksperimen utama dapat dijalankan dengan satu perintah;
6. CV 5-fold berjalan pada training set;
7. test set dikunci untuk evaluasi akhir;
8. semua metrik dan parameter tersimpan;
9. TreeSHAP dapat dijalankan pada model terpilih;
10. seluruh visual dan tabel dapat direproduksi dari file hasil.

---

## 38. Command Line Workflow

Setelah kode dibuat, target workflow pengembangan yang disarankan:

### Validasi data

```bash
python scripts/validate_data.py
```

### Bangun fitur

```bash
python scripts/build_features.py
```

### Jalankan seluruh eksperimen

```bash
python scripts/run_experiments.py
```

### Generate SHAP

```bash
python scripts/generate_shap.py
```

### Menjalankan test

```bash
pytest -q
```

### Menjalankan notebook

```bash
jupyter lab
```

Command tersebut adalah target interface proyek; implementasi script dapat disesuaikan selama urutan metodologi tetap sama.

---

## 39. Notebook Workflow

Notebook bukan sumber kebenaran utama untuk pipeline final. Notebook digunakan untuk eksplorasi, visualisasi, dan dokumentasi.

### `01_data_understanding.ipynb`

Isi:

- load data;
- struktur;
- statistik deskriptif;
- missing;
- distribusi label;
- distribusi kelas;
- validasi nilai 0/40;
- grafik awal.

### `02_preprocessing_feature_engineering.ipynb`

Isi:

- cleaning;
- label;
- leakage check;
- feature engineering;
- preview S1/S2/S3.

### `03_model_comparison.ipynb`

Isi:

- split;
- CV;
- training DT/RF;
- tuning;
- evaluasi;
- tabel perbandingan.

### `04_treeshap_analysis.ipynb`

Isi:

- load best model;
- SHAP global;
- beeswarm;
- local explanation;
- analisis fitur dominan.

---

## 40. Prosedur Penanganan Perubahan Data

Jika dataset berubah setelah pipeline selesai:

1. jangan mengubah hasil eksperimen lama;
2. simpan dataset baru sebagai versi baru;
3. catat timestamp/version;
4. jalankan ulang validasi;
5. jalankan ulang seluruh eksperimen;
6. buat run ID baru;
7. jangan mencampur hasil dari dataset berbeda dalam satu tabel final tanpa penjelasan.

Contoh:

```text
dataset_v1.xlsx
run_001/

dataset_v2.xlsx
run_002/
```

---

## 41. Prinsip Pelaporan Hasil

Laporan hasil harus mengikuti data aktual.

### Tidak boleh

- mengklaim Random Forest pasti terbaik sebelum eksperimen;
- memaksa kehadiran menjadi fitur paling penting;
- mengubah feature engineering hanya agar hasil sesuai hipotesis;
- menyebut SHAP sebagai sebab-akibat;
- menyimpulkan model menentukan kompetensi secara mutlak.

### Harus

- melaporkan bila Decision Tree lebih baik;
- melaporkan bila S2/S3 tidak meningkatkan performa;
- melaporkan variasi CV;
- membahas ukuran dataset;
- membahas heterogenitas kelas;
- menjelaskan keterbatasan mekanisme penilaian;
- menggunakan istilah kontribusi prediktif secara konsisten.

---

## 42. Struktur Output Akhir untuk Skripsi/Artikel

Pipeline harus memudahkan penyusunan bagian hasil menjadi:

1. **Deskripsi data**
2. **Distribusi label**
3. **Hasil preprocessing**
4. **Hasil feature engineering**
5. **Hasil Decision Tree**
6. **Hasil Random Forest**
7. **Perbandingan S1/S2/S3**
8. **Cross-validation mean ± SD**
9. **Confusion matrix**
10. **TreeSHAP global**
11. **TreeSHAP local**
12. **Analisis perbedaan kelas A/B/C/D**
13. **Pembahasan**
14. **Keterbatasan**
15. **Kesimpulan**

---

## 43. Pertanyaan Penelitian yang Harus Terjawab oleh Sistem

Sistem harus menyediakan bukti empiris untuk menjawab tepat tiga pertanyaan inti:

### RQ1

**Bagaimana membangun model prediksi kompetensi mahasiswa pada Praktikum Logika Pemrograman berdasarkan data aktivitas dan performa selama praktikum?**

Bukti minimal:

- pipeline preprocessing;
- label;
- fitur;
- split/CV;
- model;
- evaluasi.

### RQ2

**Bagaimana perbandingan performa Decision Tree dan Random Forest dalam memprediksi kompetensi mahasiswa?**

Bukti minimal:

- tabel S1–S3;
- mean ± SD CV;
- test metrics;
- confusion matrix;
- analisis kestabilan.

### RQ3

**Fitur apa yang memberikan kontribusi terbesar terhadap prediksi kompetensi berdasarkan TreeSHAP?**

Bukti minimal:

- SHAP importance;
- beeswarm;
- local explanation;
- pembahasan fitur dominan berbasis hasil aktual.

---

## 44. Roadmap Implementasi

### Phase 1 — Environment

- [ ] buat environment Conda;
- [ ] install dependency;
- [ ] buat `environment.yml`;
- [ ] verifikasi import.

### Phase 2 — Data

- [ ] masukkan dataset asli ke `data/raw/`;
- [ ] konfigurasi mapping kolom;
- [ ] jalankan data validation;
- [ ] selesaikan isu data berdasarkan aturan resmi.

### Phase 3 — Feature Engineering

- [ ] implementasi S1;
- [ ] implementasi S2;
- [ ] implementasi S3;
- [ ] tulis unit test.

### Phase 4 — Modelling

- [ ] implementasi split;
- [ ] implementasi CV;
- [ ] implementasi Decision Tree;
- [ ] implementasi Random Forest;
- [ ] implementasi tuning;
- [ ] implementasi evaluation.

### Phase 5 — Experiments

- [ ] jalankan enam kombinasi eksperimen;
- [ ] export CV result;
- [ ] pilih model berdasarkan aturan yang telah ditetapkan;
- [ ] evaluasi test.

### Phase 6 — XAI

- [ ] implementasi TreeSHAP;
- [ ] global plots;
- [ ] local plots;
- [ ] export SHAP values.

### Phase 7 — Reporting

- [ ] analisis kelas;
- [ ] tabel final;
- [ ] visual final;
- [ ] simpan run metadata;
- [ ] review leakage;
- [ ] review interpretasi kausal.

---

## 45. Prioritas Implementasi

### P0 — Wajib

- loading dataset;
- validation;
- label kompetensi;
- leakage prevention;
- S1/S2/S3;
- split 80:20;
- 5-fold stratified CV;
- Decision Tree;
- Random Forest;
- Accuracy/Precision/Recall/F1;
- confusion matrix;
- TreeSHAP global;
- TreeSHAP local;
- reproducibility.

### P1 — Sangat disarankan

- hyperparameter tuning;
- configuration YAML;
- automated experiment runner;
- result registry;
- unit tests;
- class analysis;
- prediction export.

### P2 — Pengembangan lanjutan

- experiment tracking framework;
- dashboard eksplorasi;
- model serialization yang lebih lengkap;
- API inferensi;
- web interface.

Fitur P2 bukan bagian wajib penelitian inti.

---

## 46. Catatan Implementasi Penting

1. **Jangan langsung coding berdasarkan nama kolom contoh.** Dataset aktual harus diperiksa lebih dulu.
2. **Jangan menghapus nilai 0/40 tanpa verifikasi aturan penilaian.**
3. **Jangan memasukkan Final Individu atau Total ke model.**
4. **Jangan menghitung completion rate dari asumsi yang tidak terdokumentasi.**
5. **Jangan menggunakan test set untuk memilih hyperparameter atau fitur.**
6. **Jangan memaksakan Random Forest menjadi model terbaik.**
7. **Jangan mengubah fitur hanya untuk membuat SHAP sesuai dugaan awal.**
8. **Jangan menggunakan bahasa kausal untuk menjelaskan SHAP.**
9. **Jangan membuat model A/B/C/D terpisah sebagai eksperimen utama pada dataset kecil.**
10. **Simpan setiap konfigurasi eksperimen agar hasil dapat direproduksi.**

---

## 47. Deliverables Final

Pada akhir implementasi, repository minimal menghasilkan:

```text
PRD.md
README.md
environment.yml
configs/*.yaml
src/*.py
scripts/*.py
tests/*.py

results/
├── data_quality/
├── metrics/
├── cv/
├── predictions/
├── confusion_matrix/
├── feature_importance/
├── shap/
└── class_analysis/

reports/
├── tables/
├── figures/
└── logs/
```

### Deliverable penelitian

- dataset yang telah divalidasi;
- dataset dengan label;
- S1/S2/S3;
- model DT/RF;
- hasil CV;
- hasil test;
- confusion matrix;
- TreeSHAP global;
- TreeSHAP local;
- analisis kelas;
- tabel dan grafik siap masuk skripsi/artikel;
- environment dan metadata reproducibility.

---

## 48. Kesimpulan PRD

PRD ini menerjemahkan rancangan penelitian menjadi spesifikasi implementasi yang dapat langsung dikembangkan menggunakan Anaconda dan Python. Inti sistem tetap mengikuti rancangan penelitian: **data aktivitas praktikum → validasi → label kompetensi dari Final Individu ≥75 → feature engineering S1/S2/S3 → Decision Tree vs Random Forest → evaluasi dengan stratified 5-fold CV dan test set → TreeSHAP → analisis kelas dan pelaporan**.

Arsitektur sengaja dibuat modular agar setiap keputusan metodologis dapat ditelusuri dan diuji. Fokus utama bukan membangun aplikasi produksi, melainkan membangun **pipeline penelitian machine learning yang valid, reproducible, anti-leakage, dan dapat digunakan untuk menghasilkan bukti empiris yang kuat**.

Rancangan penelitian sumber menekankan bahwa kekuatan studi berasal dari data praktikum nyata, pemahaman mekanisme penilaian, feature engineering berbasis proses, perbandingan model yang adil, dan interpretasi TreeSHAP yang tidak berlebihan. Implementasi harus mempertahankan prinsip tersebut sampai tahap hasil akhir.

---

## 49. Sumber Acuan PRD

PRD ini disusun berdasarkan dokumen rancangan penelitian yang diberikan, khususnya bagian tentang:

- ringkasan dan fokus penelitian;
- definisi label kompetensi;
- karakteristik dataset;
- feature engineering;
- metodologi split/CV;
- Decision Tree dan Random Forest;
- evaluasi;
- TreeSHAP;
- skenario eksperimen S1–S3;
- analisis heterogenitas kelas;
- risiko metodologis;
- checklist implementasi;
- interpretasi aturan nilai.

Dokumen sumber menyatakan secara eksplisit bahwa kelayakan publikasi tetap bergantung pada kualitas implementasi, novelty yang dibuktikan melalui tinjauan penelitian terdahulu, kualitas penulisan, venue/jurnal, dan hasil eksperimen aktual.
