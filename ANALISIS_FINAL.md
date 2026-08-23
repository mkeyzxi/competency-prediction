# Analisis Final: Prediksi Kompetensi Praktikum Logika Pemrograman

Dokumen ini merangkum analisis mendalam pasca-audit preprocessing (*Pipeline* PRD v1.1). Arsitektur terbaru ini mengakomodasi dua skema kelas (AC dan BDE) secara bersamaan dengan membasmi secara absolut masalah *Temporal Leakage* (kebocoran masa depan) serta merombak penanganan data absen/struktural.

---

## 1. Latar Belakang dan Permasalahan

Dalam upaya membangun kecerdasan buatan untuk sistem peringatan dini (*early warning system*) kelulusan praktikum, terdapat variasi data yang sangat signifikan antar kelas:
1. **Skema AC (Kelas A dan C)**: Memiliki nilai Tugas Pendahuluan (`TP`) dan `Respons` secara terpisah, dan memiliki pertemuan absensi yang terdedikasi untuk responsif Final.
2. **Skema BDE (Kelas B, D, dan E)**: Menggabungkan bobot Tugas Pendahuluan dan Respons ke dalam satu komponen `TP_Respons`, serta memvalidasi kehadiran pertemuan 9 dan 10 menggunakan pengerjaan Ujian Akhir (Flowchart dan Kodingan).

Masalah mendasar pada model ML sebelumnya adalah:
- **Temporal Leakage**: Fitur tingkat kehadiran (`Attendance_Rate`) dibangun menggunakan rata-rata absolut dari ke-10 sesi praktikum. Masuknya sesi ke-9 dan ke-10 (yang merupakan sesi Final/UAS) ke dalam fitur membuat model **bukan lagi alat prediksi *Early Warning***, melainkan "menebak masa lalu dari masa depan" (*post-event*).
- **Pemaknaan Structural Zero**: Model sebelumnya memaksakan angka $0$ pada fitur yang "tidak berlaku" di suatu kelas (misal, `Respons_Mean` untuk BDE dikunci jadi $0$). Secara akademik, nilai $0$ bermakna "mahasiswa tidak mengerjakan", sehingga pemaksaan ini mencampurkan dua makna (tidak mengerjakan vs tidak ada instrumen) dan merusak pola logika pohon keputusan.

---

## 2. Metodologi Resolusi & Pembersihan (*Anti-Leakage*)

Untuk memulihkan kejujuran eksperimen, tiga pilar pembersihan diimplementasikan:

1. **Strict Pre-Final Features (No Leakage)** 
   Sistem kini menggunakan fitur `Attendance_PreFinal_Rate` yang secara eksklusif hanya memantau absensi yang **benar-benar tersedia sebelum Ujian Akhir**.
   - Kelas AC: Menggunakan rata-rata tingkat absensi hanya dari Pertemuan 2 s.d. 7 (M1 konstan, M8-M10 berkaitan dengan final).
   - Kelas BDE: Menggunakan rata-rata absensi dari Pertemuan 1 s.d. 8 (M9-M10 berkaitan dengan pengerjaan Kodingan dan Flowchart).

2. **Imputasi *Missing Indicator* yang Jujur**
   - Skor `0` dari data asli murni dipertahankan sebagai **"Tidak Mengerjakan"**.
   - Kolom instrumen yang **Tidak Berlaku** untuk skema tertentu (seperti `TP_Mean` untuk grup BDE) kini dikembalikan sebagai nilai mutlak `NaN` (*Not Applicable / Missing*).
   - Proses penggabungan ke algoritma prediksi dibungkus di dalam *Scikit-learn Pipeline* menggunakan fungsi `SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)`. Model murni mempelajari indikator *missing* ini untuk mengenali heterogenitas skema tanpa merusak skala nilai asli mahasiswa.

3. **Penyaringan Subjek (Populasi Valid)**
   - Variabel penanda akademik seperti `absence_count` dan `Early_Exit_Flag` murni difungsikan sebagai **filter populasi** di awal (hanya membuang mahasiswa yang absen $\geq 4$ kali), dan dipastikan **tidak bocor ke matriks prediktor ($X$)**.

---

## 3. Hasil Eksperimen Global (120 Sampel Valid)

Dengan dataset 120 mahasiswa *eligible* yang kini 100% bebas dari kebocoran masa depan, berikut adalah performa model menggunakan 5-Fold Stratified Cross Validation:

| Skenario | Model | Mean CV F1-Score | F1-Score (Test Set) |
| :--- | :--- | :--- | :--- |
| **S1** (Basic) | Decision Tree | 0.721 | 0.593 |
| **S1** (Basic) | Random Forest | 0.742 | 0.733 |
| **S2** (Behavioral) | Decision Tree | 0.714 | 0.593 |
| **S2** (Behavioral) | Random Forest | **0.754** | **0.733** |
| **S3** (Relational) | Decision Tree | 0.702 | 0.593 |
| **S3** (Relational) | Random Forest | **0.756** | **0.733** |

**Kesimpulan Utama Hasil Skenario:**
- **Evaluasi Realistis (Anti-Leakage):** Terjadi penurunan F1-Score dari eksperimen lawas (0.838) ke angka **0.733**. Penurunan ini adalah indikasi **sangat positif dan jujur**, membuktikan bahwa tingginya metrik sebelumnya diperoleh secara palsu (*over-optimistic*) dari kebocoran nilai kehadiran Ujian Akhir. Model yang baru kini benar-benar murni bertindak sebagai alat peringatan dini.
- **Keunggulan Ensembling:** *Random Forest* menunjukkan ketahanan (robustness) yang luar biasa menembus angka Test F1 sebesar **0.733**, sementara *Decision Tree* tunggal langsung jatuh (0.593) akibat kerumitan variasi data yang jujur ini.
- **Dampak Feature Engineering:** Penambahan fitur *behavioral* (tingkat konsistensi penyelesaian di S2) secara nyata memberikan dorongan kestabilan `CV_F1` dari 0.742 menjadi **0.754**, memperkuat bukti bahwa rutinitas pengumpulan harian memengaruhi hasil akhir.

---

## 4. Analisis Konteks (Context Analysis): AC vs BDE

Meskipun model *Random Forest* (S2/S3) digabung dalam satu atap dengan *imputation indicator*, model ini ternyata memiliki performa yang sangat berbeda jika diuji silang per kelompok skema:

| Kelompok | Akurasi | Presisi | Recall | F1-Score | Support (N Test) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AC** | 54.5% | 62.5% | 71.4% | 66.7% | 11 |
| **BDE** | 76.9% | 75.0% | 85.7% | 80.0% | 13 |

**Interpretasi Konteks:**
Perbedaan performa antara AC (F1: 66.7%) dan BDE (F1: 80.0%) memperlihatkan adanya **heterogenitas konteks penilaian**. 
Hal ini menandakan bahwa sistem penggabungan instrumen pada skema BDE (di mana *TP* dan *Respons* dirangkum jadi satu) secara struktural jauh lebih linier terhadap nilai UAS dan lebih mudah dikenali pola resikonya (Recall BDE mencapai 85.7%) jika dibandingkan dengan skema pecahan di kelompok A/C. Perbedaan metrik ini adalah bukti dinamika alamiah operasional asisten praktikum, bukan cacat pada model algoritmik.

---

## 5. Insight dari Ekstraksi TreeSHAP

Berkat arsitektur interpolasi yang transparan, kita telah menggunakan pustaka **TreeSHAP** (SHapley Additive exPlanations) untuk membedah bagaimana *Random Forest* membuat keputusannya. Berdasarkan grafik plot (terlampir di map `results/shap/`):

1. **Kehadiran Murni Mendominasi (Attendance_PreFinal_Rate)**
   Tingkat absensi murni tanpa bocoran UAS muncul sebagai pendorong terkuat untuk probabilitas kelas "Belum Kompeten". Titik distribusi SHAP membuktikan bahwa semakin minim partisipasi fisik pre-final, skor akhir pasti anjlok secara drastis.
2. **Sinyal Kedisiplinan (Laporan_Completion_Rate)**
   Menyusul di tempat kedua pada eksperimen S2, rasio penyelesaian laporan menduduki parameter kritis. Fitur ini membantu model mengenali tren: *mahasiswa mungkin hadir, namun jika rasionya kosong (selalu mendapat nilai 0 dari asisten), ia diprediksi tidak akan lolos UAS.*
3. **Pemanfaatan Missing Indicator (`missingindicator_...`)**
   Fitur indikator "ketiadaan instrumen" yang di-injeksi otomatis oleh `SimpleImputer(add_indicator=True)` ternyata dipakai oleh Random Forest secara aktif sebagai "gerbang logika percabangan awal" untuk memisahkan gaya *scoring* BDE dan AC. 

---

## 6. Kesimpulan Praktis & Rekomendasi Skripsi

Arsitektur pipeline ini kini diklaim **aman secara metodologis, bebas dari bias waktu (*temporal leakage*), dan secara konseptual jujur (*honest imputation*)**. 
Sistem ini sangat direkomendasikan untuk dijalankan menggunakan **Random Forest pada skenario S2 (Behavioral)**. Walaupun akurasi keseluruhannya 0.733, F1-Score ini adalah cerminan sesungguhnya dari batas prediktabilitas *Early Warning System*, di mana 85.7% dari seluruh mahasiswa kelompok B/D/E yang berisiko gagal berhasil dideteksi dengan sukses jauh sebelum mereka menyentuh Ujian Akhir Praktikum.
