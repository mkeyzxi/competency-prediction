# Dokumentasi Perubahan Pipeline (PRD v1.1)

Dokumen ini mencatat secara lengkap seluruh perubahan teknis dan arsitektural yang dilakukan pada *pipeline* prapemrosesan data, rekayasa fitur (feature engineering), hingga eksperimen Machine Learning untuk mendukung masuknya dua format data baru (Skema Kelas A/C dan Kelas B/D/E).

---

## 1. Konfigurasi (`configs/`)

### a. `data_config.yaml`
- **Sebelumnya**: Hanya membaca satu file `data_testing_praktikum.xlsx` dan kolom diekstrak secara kaku menggunakan penamaan yang disesuaikan secara sepihak.
- **Perubahan**: 
  - Memisahkan path untuk dua sumber data: `ac_path` dan `bde_path`.
  - Mendaftarkan sheet apa saja yang dibaca untuk AC (`A`, `C`) dan BDE (`KELAS B`, `KELAS D`, `KELAS E`).
  - Mendefinisikan struktur spesifik referensi skema penilaian (*scoring scheme weights*) untuk AC dan BDE agar sistem bisa merujuk pada bobot asli.

### b. `attendance_mapping.yaml` (File Baru)
- **Sebelumnya**: Tidak ada.
- **Perubahan**: Dibuat khusus untuk mendeklarasikan aturan rekonstruksi 10 pertemuan.
  - Skema **AC**: M1 dan M10 dianggap selalu hadir. M2-M7 membaca kolom presensi 1-6. M8 membaca ketersediaan nilai Individu. M9 membaca ketersediaan nilai Akhir.
  - Skema **BDE**: M1-M8 membaca kolom presensi langsung. M9 melihat ketersediaan *nilai flowchart*. M10 melihat ketersediaan *nilai kodingan*.

---

## 2. Ingesti Data (`src/data_loader.py`)

- **Sebelumnya**: Fungsi tunggal `load_and_clean_data()` yang langsung memotong (*skiprows*) baris secara absolut (baris pertama) tanpa mempedulikan hierarki *header* atau validitas baris tersebut.
- **Perubahan**:
  - Dibuat fungsi terpisah: `load_ac_data()`, `load_bde_data()`, dan `load_final_uas()`.
  - **Penanganan Header Dinamis**: File kelas A/C diekstrak mulai dari indeks spesifik (men-skip 3 baris awal header kosong/merged cells), sedangkan kelas B/D/E memulai dari baris ke-5.
  - **Pembersihan NIM**: Melakukan *casting* ke teks, membersihkan *whitespace*, dan membuang elemen `.0` pada NIM di awal sehingga format NIM di *activity sheets* dan *final sheet* identik sebelum di-*merge*.
  - **Pencegahan Duplikat**: `load_final_uas()` akan mengambil baris yang duplikat berdasakan NIM dan menyimpan hanya nilai *final* yang paling tinggi, sesuai kesepakatan terbaru.
  - **Penggabungan**: Menggabungkan data aktivitas dengan `PENILAIAN_UAS` (khusus untuk menyuplai metrik target UAS kelompok BDE).

---

## 3. Prapemrosesan (`src/preprocessing.py`)

- **Sebelumnya**: Pembersihan, pembuatan target, dan pembentukan label tergabung dalam `data_loader.py` dan dilakukan sesuka hati (misalnya kehadiran <= 1 otomatis dihapus tanpa rekam jejak).
- **Perubahan**: Modul diisolasi sepenuhnya dengan tugas:
  - **Rekonstruksi 10 Sesi Kehadiran (Attendance_1 sd Attendance_10)**: Berbasis aturan yang berbeda antara skema AC dan skema BDE (melihat pada nilai asli dan ketersediaan flowchart/kodingan). Nilai `0.5` diizinkan.
  - **Kalkulasi Absen & Flagging**: Menciptakan kolom audit `absence_count`, `Early_Exit_Flag` (> 1 absen), dan `Attendance_Ineligible_Flag` (> 3 absen).
  - **Penyimpanan Dataset Audit**: Mereka yang gugur atau hilang nilai finalnya akan disimpan ke `data/processed/excluded.csv`, sedangkan yang sah dipindahkan ke `data/processed/eligible.csv` sehingga tidak ada observasi yang hilang secara misterius (*silent drop*).

---

## 4. Rekayasa Fitur (`src/feature_engineering.py` & `src/feature_registry.py`)

- **Sebelumnya**: Setiap kelas dianggap memiliki fitur yang sama. Kolom `TP` dan `Respons` langsung di-*mean* meskipun di kelas B/D/E komponen tersebut tidak dipisah.
- **Perubahan**:
  - **Penghormatan Skema Data**: Kelas A/C dikalkulasi dengan memecah `TP_Mean` dan `Respons_Mean`.
  - **Gabungan TP & Respons untuk BDE**: Diciptakan fitur `TP_Respons_Mean` dan `TP_Respons_Completion_Rate` untuk Kelas B/D/E karena nilai Tugas Pendahuluan merepresentasikan gabungan.
  - **Zeroing Out yang Tidak Relevan**: Di dalam dataset universal, kelas A/C akan memiliki `TP_Respons_Mean = 0`, sedangkan kelas B/D/E akan memiliki `TP_Mean = 0`, `Respons_Mean = 0`, dan `Respons_TP_Gap = 0`. Hal ini menghindari error ML saat disatukan, tapi tetap membiarkan pohon keputusan membedakan arsitektur kelompok.
  - **Register Diperbarui**: Fitur tambahan ini dimasukkan ke dalam pendaftaran `S1`, `S2`, dan `S3` di `feature_registry.py`.

---

## 5. Flow Pelaksanaan (`scripts/build_features.py`)

- **Sebelumnya**: Mengimpor `create_labels` secara terpisah.
- **Perubahan**: Diubah untuk membaca langsung dari `data/processed/eligible.csv` (hasil dari *preprocessing* terbaru). Saat mengekspor hasil skenario S1/S2/S3, program ini diinstruksikan agar selalu membawa metadata kolom `Class` dan `Scoring_Scheme` agar berguna untuk *Context Analysis* selanjutnya.

---

## 6. Eksperimen dan Evaluasi (`src/experiments.py` & `src/evaluation.py`)

- **Sebelumnya**: Mengukur tingkat akurasi test secara gabungan (satu nilai untuk semua).
- **Perubahan**: 
  - Melakukan _tracking_ `Scoring_Scheme` di setiap ID prediksi pada `test_predictions`.
  - Menambahkan _function_ `run_context_analysis()` di akhir skrip eksperimen.
  - Otomatis membuat tabel evaluasi _robustness_ di `results/reports/context_analysis.csv` yang memecah akurasi, _precision_, _recall_, dan F1 score secara spesifik antara populasi kelas A/C dan B/D/E pada seluruh varian model yang dicobakan.
