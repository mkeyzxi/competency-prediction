# 📚 Panduan Arsitektur & Fungsi Skrip (Developer Reference)

Dokumen ini menjelaskan rancangan arsitektur *software* dari sistem Prediksi Kompetensi Mahasiswa (*Early Warning System*). Seluruh basis kode dibagi secara modular berdasarkan prinsip *Separation of Concerns*, memisahkan logika pengolahan data murni, konfigurasi eksperimen, hingga mesin analisis prediksi.

---

## 🏗️ Hierarki dan Alur Kerja Utama

Kode dalam repositori ini dipisahkan ke dalam dua entitas utama:
1. **`src/` (Source Code / Mesin Internal)**: Berisi pustaka *functions* modular. Tidak dijalankan langsung oleh pengguna, melainkan dipanggil oleh skrip lain.
2. **`scripts/` (Eksekusi Eksperimen)**: Berisi skrip aplikatif yang mengeksekusi *pipeline* secara penuh dari hulu ke hilir. Didesain untuk dijalankan via Terminal.

---

## 📂 1. Pustaka Internal (`src/`)

### 🛠️ A. Persiapan Data (Data Preparation)
- **`src/build_dataset.py`**
  - **Fungsi**: Titik awal (*entry point*) seluruh proses data. Membaca data mentah dari dosen (`data/raw/DBNR.xlsx`).
  - **Kenapa Penting?**: Di sinilah justifikasi Target Label ditentukan. Mahasiswa dengan nilai $\ge 83$ (zona aman akademik) ditandai sebagai `Kompeten (1)`. Mengonversi *spreadsheet* yang rumit menjadi baris riwayat data terstruktur (`activities_long.csv`) untuk diproses lebih lanjut.

- **`src/features.py` & `src/feature_engineering.py`**
  - **Fungsi**: Mengubah riwayat absensi dan nilai tugas menjadi *Features* kuantitatif yang bisa dimengerti *Machine Learning*.
  - **Kenapa Penting?**: Menangani pemotongan waktu absensi (*temporal cutoffs*) untuk menjamin **Zero Temporal Leakage**. Di sinilah rumus-rumus kecerdasan EWS dirakit (misal: `Early_Performance_Composite` yang mengekstrak performa murni pada 3 minggu awal tanpa melihat masa depan).

- **`src/feature_registry.py`**
  - **Fungsi**: Papan kontrol (*registry*) yang mendaftar fitur apa saja yang masuk ke dalam Skenario tertentu (S1, S2, ..., S3_E).
  - **Kenapa Penting?**: Mencegah model menerima kolom bocor (seperti metadata *Final_Score*). Hanya fitur yang terdaftar di file ini yang diizinkan masuk ke ruang pelatihan model.

### 🧠 B. Mesin Pembelajaran (*Machine Learning Engine*)
- **`src/models.py`**
  - **Fungsi**: Mendefinisikan kerangka (*pipeline*) algoritma seperti *Decision Tree*, *Random Forest*, dan *Dummy Classifier*.
  - **Kenapa Penting?**: Mengunci proses *imputation* (pengisian nilai kosong) dan SMOTE (jika dipakai) ke dalam `imblearn.pipeline.Pipeline`. Ini menjamin tidak ada **Preprocessing Leakage** saat dilakukan validasi silang (transformasi data hanya berlaku pada data latih, bukan data tes).

- **`src/split.py`**
  - **Fungsi**: Memecah secara acak (namun proporsional / *stratified*) keseluruhan data menjadi data Latih dan data Uji (Holdout).
  - **Kenapa Penting?**: Menyisihkan 20% data secara absolut sebagai *Holdout Set* (Data Beku). Data beku ini **tidak pernah** dilihat oleh mesin saat fase penyetelan parameter model.

- **`src/tuning.py`**
  - **Fungsi**: Menangani *Hyperparameter Tuning* (pencarian konfigurasi pohon terbaik) menggunakan *RandomizedSearchCV* di dalam putaran *Inner Cross-Validation*.
  - **Kenapa Penting?**: File ini juga mencari titik *Threshold* klasifikasi secara dinamis dari data latih, memisahkan prediksi probabilitas agar *Recall BK* tercapai optimal.

### 📈 C. Evaluasi & Interpretasi
- **`src/evaluation.py`**
  - **Fungsi**: Mesin penilai (*Scorer*). Melakukan validasi silang (Outer CV) dan menghitung performa pada Data Beku.
  - **Kenapa Penting?**: Menyediakan algoritma **Bootstrap Confidence Interval (95%)** (melakukan sampel ulang 1000x) untuk menyajikan angka rentang akurasi (misal: 62% - 100%) karena sampel uji yang diolah tergolong kecil ($n=18$).

- **`src/evaluate_shap.py`**
  - **Fungsi**: Mengekstraksi logika internal "kotak hitam" (*blackbox*) model menggunakan *SHapley Additive exPlanations* (SHAP).
  - **Kenapa Penting?**: Ini adalah pilar utama keilmiahan. Membongkar fitur mana (misal `Early_Performance_Composite` vs `Laporan_Mean`) yang paling berkontribusi mendorong model untuk berteriak "Mahasiswa Ini Terancam Gagal!".

---

## 🚀 2. Skrip Eksekutor (`scripts/`)

Skrip di dalam direktori ini hanyalah *wrapper* (pembungkus) yang menggabungkan dan menjalankan modul-modul di `src/` dari awal hingga selesai.

- **`scripts/run_baseline_experiments.py`**
  - **Fungsi**: Melangsungkan kompetisi *Nested CV* antara Skenario Retrospektif (S1 - S5) dengan fitur-fitur yang merekap satu semester penuh. Bertujuan menetapkan *baseline* (Tolok Ukur Historis).

- **`scripts/run_optimized_experiment.py`**
  - **Fungsi**: Menjalankan pengujian berstandar SINTA-2 dengan memfokuskan pencarian pada *Genuine EWS* (S3_A hingga S3_EWS).
  - **Kenapa Penting?**: Menerapkan hierarki metodologi yang dikunci mati: Lakukan pencarian, temukan 1 pemenang berdasarkan CV, kunci pemenang, uji pada tes beku, cetak interval kepercayaan. Menyimpan model akhir ke `models/`.

- **`scripts/generate_shap_s3_dasar.py`** & **`scripts/generate_shap.py`**
  - **Fungsi**: Memanggil `src/evaluate_shap.py` untuk menggambar plot dan visualisasi matematis (seperti plot koloni lebah / *Beeswarm*) dari dua sisi: model retrospektif (S3) vs model EWS (S3_E). Gambar disimpan ke dalam `results/shap/`.

---

## 📋 Alur Kesimpulan (Rangkuman Eksekusi)

Bila diibaratkan pabrik:
1. `build_dataset.py` & `features.py` = **Penggilingan Bahan Baku**.
2. `tuning.py`, `models.py`, `split.py` = **Mesin Pabrikasi (Proses Inner Loop)**.
3. `evaluation.py` = **Quality Control / Pemeriksa Mutu (Proses Outer Loop)**.
4. `run_optimized_experiment.py` = **Mandor Pabrik (Tombol Utama)**.
5. `generate_shap.py` = **Juru Laporan/Jurnalis (Visualisasi Hasil)**.

Dengan pemisahan logika yang rapi di atas, pengembangan lebih lanjut (seperti menambah data tahun depan) tidak akan merusak kestabilan eksperimen karena parameter dan kebocoran telah dipagari sejak dini!
