# Prediksi Kompetensi Praktikum Logika Algoritma

Repositori ini memuat _pipeline_ penelitian Machine Learning yang dirancang secara modular dan _reproducible_ untuk memprediksi tingkat kompetensi mahasiswa pada praktikum **Logika & Algoritma**.

Tujuan utama dari proyek ini bukan sekadar membangun aplikasi produksi, melainkan **membangun eksperimen Machine Learning yang valid secara akademis, bebas dari _data leakage_, dan dapat digunakan untuk menghasilkan bukti empiris yang kuat** (sebagai basis pembahasan BAB IV pada laporan penelitian/skripsi).

---

## 📋 Daftar Isi

1. [Konteks Penelitian](#konteks-penelitian)
2. [Struktur Direktori](#struktur-direktori)
3. [Instalasi dan Konfigurasi](#instalasi-dan-konfigurasi)
4. [Cara Menjalankan Pipeline](#cara-menjalankan-pipeline)
5. [Skenario Fitur (Feature Engineering)](#skenario-fitur-feature-engineering)
6. [Metodologi Evaluasi](#metodologi-evaluasi)
7. [Hasil Eksperimen & Interpretasi (SHAP)](#hasil-eksperimen--interpretasi-shap)

---

## 🔬 Konteks Penelitian

Penelitian ini bertujuan untuk mengklasifikasikan mahasiswa ke dalam dua kelas kompetensi:

- `1` : **Kompeten** (Nilai Final Individu >= 75)
- `0` : **Belum Kompeten** (Nilai Final Individu < 75)

Untuk mencegah _data leakage_, fitur yang berhubungan langsung dengan nilai akhir (`Final_Individu`, `Final_Kelompok`, `Final_Total`, `NILAI_AKHIR`, `PREDIKAT`) **telah dihapus** dari atribut pelatihan. Selain itu, mahasiswa yang berstatus _dropout_ atau berhenti di awal praktikum (dideteksi dengan tingkat kehadiran $\le 1$) akan dibersihkan secara otomatis oleh _pipeline_.

Model yang diuji:

- **Decision Tree** (Cenderung _overfit_ pada data berjumlah kecil)
- **Random Forest** (Metode _ensemble bagging_ untuk meredam variansi pada dataset berukuran kecil)

---

## 📁 Struktur Direktori

```text
.
├── configs/            # File konfigurasi YAML (data_config, experiment_config)
├── data/               # Direktori dataset
│   ├── raw/            # Dataset asli (.xlsx) - JANGAN DIUBAH!
│   ├── interim/        # Data gabungan sementara (hasil pembersihan dropouts)
│   └── processed/      # Dataset final (sudah melalui feature engineering)
├── models/             # Tempat model Scikit-Learn (.pkl) tersimpan setelah training
├── notebooks/          # Jupyter Notebook untuk eksplorasi awal / EDA (jika ada)
├── reports/            # Laporan analisis (seperti data quality report)
├── results/            # Hasil akhir eksperimen
│   ├── metrics/        # Tabel evaluasi model (.csv)
│   ├── predictions/    # Hasil prediksi lengkap untuk Test Set
│   └── shap/           # Visualisasi plot interpretabilitas (TreeSHAP)
├── scripts/            # Skrip entry-point untuk menjalankan pipeline (run_*.py)
├── src/                # Modul inti Python (Source Code Modular)
│   ├── class_analysis.py
│   ├── data_loader.py          # Logika pembersihan (termasuk filter dropout)
│   ├── data_validation.py
│   ├── evaluation.py
│   ├── experiments.py          # Runner eksperimen & hyperparameter tuning
│   ├── feature_engineering.py  # Ekstraksi fitur dan kalkulasi mean/completion_rate
│   ├── feature_registry.py     # Registry pemetaan fitur per Skenario
│   ├── labeling.py             # Logika pembuatan target "Kompeten"
│   ├── models.py               # Pemanggilan base_estimator
│   ├── shap_analysis.py        # Logika penghasil grafik SHAP
│   ├── split.py                # Train-test split tershift/stratified 80:20
│   └── tuning.py               # GridSearchCV tuning
├── tests/              # (Opsional) Unit testing code
├── ANALYSIS_GUIDE.md   # Panduan penulisan Bab IV dari hasil eksperimen
├── PRD.md              # Product Requirements Document
└── requirements.txt    # Daftar dependensi Python
```

---

## ⚙️ Instalasi dan Konfigurasi

Pastikan Anda memiliki instalasi Python versi 3.9 atau lebih baru. Disarankan menggunakan _virtual environment_.

1. **Clone/Buka repositori ini** di IDE Anda.
2. **Buat dan aktifkan Virtual Environment** (Opsional namun direkomendasikan):
   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di Linux/Mac:
   source venv/bin/activate
   ```
3. **Install _Dependencies_**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Cara Menjalankan Pipeline

Proyek ini telah dikemas menjadi sederet skrip Python terstruktur. Untuk menjalankan eksperimen dari awal hingga akhir, eksekusi skrip berikut secara berurutan di dalam terminal pada _root directory_ proyek:

1. **Validasi & Load Data**
   Membaca data _raw_, membersihkan kolom _metadata_, serta menghapus mahasiswa yang sudah _dropout_ (kehadiran <= 1).

   ```bash
   python scripts/validate_data.py
   ```

2. **Feature Engineering**
   Mengonversi data _raw_ (nilai tugas, nilai respon, kehadiran) menjadi skor agregat yang _machine-readable_ dengan penanganan _missing values_ yang akurat (menganggap tugas tidak terkumpul sebagai nilai 0).

   ```bash
   python scripts/build_features.py
   ```

3. **Eksperimen Model (Training & Validasi)**
   Menjalankan skenario S1, S2, dan S3 menggunakan _GridSearchCV_ untuk _hyperparameter tuning_, melatih Decision Tree & Random Forest, lalu menyimpannya dalam metrik perbandingan.

   ```bash
   python scripts/run_experiments.py
   ```

4. **Interpretasi Model dengan TreeSHAP**
   Men- _generate_ grafik interpretasi secara otomatis untuk membedah seberapa besar dampak suatu tugas terhadap kelulusan praktikan.
   ```bash
   python scripts/generate_shap.py
   ```

_(Khusus pengguna PowerShell/Linux, Anda dapat menjalankan semuanya dalam satu baris):_

```bash
python scripts/validate_data.py; python scripts/build_features.py; python scripts/run_experiments.py; python scripts/generate_shap.py
```

---

## 🧬 Skenario Fitur (Feature Engineering)

Proyek ini menguji tiga (3) tingkatan fitur (diatur di `src/feature_registry.py`) untuk melihat titik optimal kompleksitas model:

- **S1 (Basic)**: Hanya menggunakan rata-rata nilai (`Attendance_Rate`, `TP_Mean`, `Respons_Mean`, `Laporan_Mean`).
- **S2 (Behavioral)**: S1 ditambah indikator kedisiplinan/penyelesaian tugas (`TP_Completion_Rate`, `Respons_Completion_Rate`, `Laporan_Completion_Rate`).
- **S3 (Relational)**: S2 ditambah gap kalkulatif (`Respons_TP_Gap`), yang berfungsi melihat korelasi selisih nilai Respon dan Tugas Pendahuluan.

---

## 🛡️ Metodologi Evaluasi

Untuk mempertahankan validitas hasil (anti-_leakage_), eksperimen dikawal dengan prinsip berikut:

- **Isolasi Test Set**: Data di-_split_ (80% Train, 20% Test) secara Stratified sebelum evaluasi.
- **Tuning Independen**: _Hyperparameter tuning_ (GridSearchCV) hanya dilakukan murni di dalam wilayah _Train Set_ dengan validasi silang (5-Fold CV).
- **Metric Fokus**: Pengambilan keputusan terbaik didasarkan pada metrik **F1-Score**, guna menangani ketidakseimbangan ukuran kelas serta kepekaan atas _False Positive/False Negative_.

---

## 📊 Hasil Eksperimen & Interpretasi (SHAP)

Setelah _pipeline_ selesai dijalankan, Anda dapat merujuk ke direktori `results/` untuk pelaporan BAB IV:

- `results/metrics/model_comparison.csv` : Berisi perbandingan silang seluruh _fold_ algoritma beserta stabilitas _Test-set_. Gunakan tabel ini sebagai hasil evaluasi utama.
- `results/shap/` : Menampung hasil visualisasi lokal dan global (_beeswarm plot_ dan _global importance bar plot_) untuk menjelaskan proses prediktif model agar transparan secara _white-box_.

Panduan spesifik untuk menuangkan output `results/` ke dalam bab analisis karya ilmiah telah terdokumentasi dengan lengkap di `ANALYSIS_GUIDE.md`.
