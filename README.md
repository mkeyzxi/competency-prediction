# 🚀 Panduan Eksekusi Eksperimen Prediksi Kompetensi

Repositori ini berisi keseluruhan sistem (_pipeline_) eksperimen _Machine Learning_ untuk mendeteksi dini potensi kegagalan (Belum Kompeten) mahasiswa pada Praktikum Logika & Algoritma.

Berdasarkan hasil analisis dan standarisasi nilai predikat unggul akademik (Zona Aman A-), kita menggunakan **Batas Kompetensi = 83**. Dataset diolah dalam kondisi murni (tanpa intervensi _oversampling_ seperti SMOTE) untuk menjaga keaslian distribusi dan mencegah *optimistic bias*.

Berikut adalah panduan **LENGKAP** dan **BERURUTAN** untuk menjalankan seluruh eksperimen dari awal mula data mentah (Excel) hingga akhir (Grafik SHAP), tanpa perlu menyentuh atau mengubah satu baris kode pun!

---

## 🧹 Langkah 1: Pembersihan Data Mentah (Dari Excel)

Langkah paling awal adalah membaca data mentah dari Excel dosen (`data/raw/DBNR.xlsx`) lalu membersihkannya menjadi tabel data siap olah.

Buka terminal Anda, pastikan berada di dalam folder proyek ini (`c:\belajarku\Belajar ML\Logika-Algoritma`), lalu ketikkan perintah berikut:

```bash
python src/build_dataset.py
```

**Apa yang terjadi?**
Skrip ini akan mengambil data dari `DBNR.xlsx`, melabeli kelulusan mahasiswa berdasarkan ambang batas akademik kompetensi **83**, dan menyimpannya dalam format bersih (seperti `activities_long.csv` dan `students_master.csv`) di folder `data/processed/`.

---

## 🛠️ Langkah 2: Ekstraksi & Persiapan Fitur

Setelah data dibersihkan, kita harus mengekstrak riwayat aktivitas mahasiswa tersebut menjadi kumpulan _features_ (Skenario 1 hingga 5, dan S3 EWS). Ketikkan perintah berikut:

```bash
python src/features.py
```

**Apa yang terjadi?**
Skrip ini akan membaca `activities_long.csv` dan menghitung rata-rata, absensi, hingga pola EWS. Hasil akhirnya berupa file-file CSV (seperti `C_Full_S1.csv`, `C_Full_S3.csv`, dll) yang siap dimasukkan ke dalam model ML, tersimpan di folder `data/features/`.

---

## 📊 Langkah 3: Uji Coba Baseline Skenario (S1 - S5)

Tahap ini bertujuan untuk membuktikan secara ilmiah skenario dasar mana yang terbaik (komparasi S1, S2, S3, S4, dan S5).

### 3A. Menjalankan Komparasi (S1-S5)

Ketik perintah berikut:

```bash
python scripts/run_baseline_experiments.py
```

**Apa yang terjadi?**
Sistem akan melatih _Decision Tree_ dan _Random Forest_ untuk S1 hingga S5 menggunakan _Nested Cross Validation_. Di akhir proses, sistem akan mencetak **Leaderboard** (Klasemen) evaluasi Retrospektif di terminal Anda. Skenario 3 (S3) akan keluar sebagai konfigurasi fitur historis terbaik. Skrip ini juga otomatis menyimpan model S3 terbaik ke dalam folder `models/`.

### 3B. Menghasilkan Grafik SHAP untuk S3 Dasar

Setelah S3 terbukti menang, mari kita lihat alasan matematis di balik kemenangannya. Ketik:

```bash
python scripts/generate_shap_s3_dasar.py
```

**Apa yang terjadi?**
Skrip ini akan membedah model S3 Dasar menggunakan metode _Explainable AI_ (SHAP). Hasil grafiknya akan tersimpan di dalam folder `results/shap/` (Cari file yang berawalan `Baseline_S3`). Anda akan melihat bahwa model ini masih sangat bergantung pada _Laporan_Mean_.

---

## 🔬 Langkah 4: Uji Coba Khusus Skenario 3 (EWS)

Karena S3 Dasar memenangkan komparasi, kita melangkah ke pengujian **lanjutan khusus S3** dengan menambahkan fitur _Early Warning System_ (EWS) untuk melihat perilaku mahasiswa secara spesifik (seperti Tren Penurunan, Absen Berturut-turut, dll).

### 4A. Menjalankan Komparasi EWS (S3_A s/d S3_EWS)

Ketik perintah berikut:

```bash
python scripts/run_optimized_experiment.py
```

**Apa yang terjadi?**
Sistem akan menjalankan _Nested Cross Validation_ membandingkan kelompok fitur EWS (S3_A hingga S3_EWS). Model yang dikonstruksi benar-benar ditutup matanya dari data masa depan (hanya menggunakan performa 2-3 minggu pertama). Setelah selesai, skrip mengevaluasi metrik (serta Confidence Interval) dan memilih model *Genuine EWS* terbaik (S3_E), lalu menyimpannya sebagai **Model Final** di folder `models/`.

### 4B. Menghasilkan Grafik SHAP untuk Model Final (S3_E)

Untuk membuktikan bahwa fitur EWS (seperti adaptasi 2 minggu pertama) jauh lebih jitu dan cerdas daripada _Laporan_Mean_, jalankan:

```bash
python scripts/generate_shap.py
```

**Apa yang terjadi?**
Skrip ini akan melacak model final (S3_E) yang baru saja disimpan oleh Langkah 4A, membedahnya, dan menyimpan grafiknya di folder `results/shap/` (Cari file yang berawalan `FINAL_S3_E`).

---

## 🎉 Kesimpulan

Sekarang Anda dapat membuka folder `results/shap/` dan membandingkan _Feature Importance_ antara **S3 Dasar** (Retrospektif) dengan **S3_E Final** (Genuine EWS). Anda akan melihat bahwa Model Final kita kini berhasil mendeteksi potensi kegagalan dengan mengandalkan performa awal (_Early Performance_) sebagai **kontributor prediktif tertinggi**, yang menjadikannya sebuah _Early Warning System_ yang andal secara metodologis!

Selamat bereksperimen!
