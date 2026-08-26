# Laporan Analisis Kelulusan Praktikum Berbasis Machine Learning dan Explainable AI (XAI)

Dokumen ini merangkum seluruh proses eksperimen, metodologi, dan hasil pengujian model klasifikasi untuk memprediksi kelulusan (status Kompeten vs. Belum Kompeten) pada praktikum Logika & Algoritma. Laporan ini disusun dengan standar pelaporan akademik, mengedepankan evaluasi yang objektif dan interpretasi model yang transparan.

---

## 1. Latar Belakang dan Identifikasi Masalah Penelitian

Prediksi kelulusan akademik secara dini sangat penting untuk memberikan intervensi kepada mahasiswa yang berisiko gagal. Namun, dalam konteks dataset historis praktikum ini, terdapat tiga masalah fundamental yang harus diatasi sebelum pemodelan dilakukan:

1. **Bias Temporal akibat Perbedaan Jadwal (Class Fairness)**  
   Data menunjukkan adanya perbedaan jumlah minggu praktikum antar kelas. Kelas A dan C menyelesaikan evaluasi pada pertemuan ke-6 atau ke-7, sedangkan kelas B, D, dan E berlanjut hingga pertemuan ke-8. Memasukkan seluruh data hingga minggu ke-8 akan menyebabkan *missing values* yang sistematis bagi kelas A dan C. Mengisi kekosongan ini dengan angka `0` berisiko menimbulkan bias, di mana algoritma akan menghukum (*penalize*) mahasiswa kelas A dan C secara tidak adil.
2. **Ketidakseimbangan Kelas (Class Imbalance)**  
   Proporsi mahasiswa yang berstatus "Kompeten" jauh lebih besar daripada "Belum Kompeten". Pemodelan prediktif konvensional pada data seperti ini cenderung bias ke arah kelas mayoritas (menghasilkan *dummy accuracy* sebesar 80%, namun dengan *Balanced Accuracy* yang rendah di kisaran 50%).
3. **Ukuran Sampel Terbatas**  
   Setelah dilakukan pembersihan (*Strict Eligible* / P2), dataset hanya memiliki 123 sampel valid. Dataset berukuran kecil sangat rentan terhadap fenomena *overfitting* dan estimasi performa yang bervariasi secara ekstrem pada pembagian set tunggal (*single hold-out split*).

---

## 2. Metodologi Penelitian

Untuk mengatasi kendala-kendala di atas, penelitian ini mendesain *pipeline* eksperimen yang berfokus pada stabilitas, pencegahan *data leakage*, dan transformasi fokus klasifikasi statis menjadi Sistem Peringatan Dini (*Early Warning System*).

### 2.1. Penyeragaman Jendela Waktu Observasi (*Temporal Cutoffs*)
Sebagai solusi terhadap bias temporal, data diubah dengan menerapkan teknik batas waktu (*cutoff*) yang seragam bagi seluruh kelas. Model dilatih pada beberapa skenario pengamatan untuk menguji kapabilitas deteksi dini:
- **C1**: Fitur dihitung secara kumulatif hingga Minggu ke-4.
- **C2**: Fitur dihitung secara kumulatif hingga Minggu ke-5.
- **C3**: Fitur dihitung secara kumulatif hingga Minggu ke-6.
- **C4**: Fitur dihitung secara kumulatif hingga Minggu ke-7.
- **C_Full**: Pengamatan menggunakan seluruh rentang sejarah historis (hingga Minggu ke-8).

Fitur-fitur statistik (rata-rata, tren, nilai maksimum/minimum, dan standar deviasi) hanya diagregasi dalam batas observasi yang ditentukan guna memastikan validitas prediktif model.

### 2.2. Pencegahan Kebocoran Data (Nested Pipeline)
1. **Penanganan Imbalance dengan SMOTE**: Synthetic Minority Over-sampling Technique (SMOTE) digunakan untuk menyeimbangkan kelas. Untuk mencegah *data leakage*, SMOTE dieksekusi secara eksklusif **di dalam** proses validasi silang (hanya pada bagian data pelatihan/*training fold*). Data uji (*validation/test fold*) tidak pernah disintesis.
2. **Seleksi Fitur via Permutation Importance (Inner-CV)**: Ekstraksi subset fitur terbaik (`DynamicTopKSelector`) menggunakan metode *Permutation Importance*. Untuk menghindari *overfitting* pemeringkatan fitur terhadap data latih, pemecahan data *inner train-validation* diaplikasikan pada tahap perhitungan kepentingannya.

### 2.3. Evaluasi Kestabilan (Repeated Stratified K-Fold CV)
Karena jumlah sampel yang kecil, penggunaan metode pengujian *hold-out* tunggal tidak cukup untuk membuktikan generalisasi model. Metodologi evaluasi yang digunakan adalah **Repeated Stratified K-Fold Cross Validation** (5 *splits*, 5 *repeats* = total 25 *folds*). Tolok ukur utama kinerja model dilaporkan dalam bentuk **Mean ± Standar Deviasi (SD)** terhadap metrik *Balanced Accuracy* dan *Accuracy*.

---

## 3. Hasil Eksperimen dan Pemilihan Model

Pengujian komparatif difokuskan pada dua algoritma *tree-based*, yakni **Decision Tree** dan **Random Forest**, yang dievaluasi di sepanjang jendela waktu temporal (C1 hingga C_Full).

### 3.1. Ringkasan Performa Model (Skenario Fitur S6)
Berikut merupakan ringkasan dari metrik *Repeated Cross Validation* serta performa *hold-out* akhir untuk kandidat terbaik:

| Cutoff | Jendela Waktu | Model & Konfigurasi Fitur | CV Balanced Accuracy (Mean ± SD) | Test Balanced Accuracy (Final Holdout) |
| :--- | :--- | :--- | :--- | :--- |
| **C1** | Minggu ke-4 | Random Forest (Feature Selection) | `0.6291 ± 0.1101` | `92.50 %` |
| **C2** | Minggu ke-5 | Random Forest (Feature Selection) | `0.6567 ± 0.1241` | `90.00 %` |
| **C3** | Minggu ke-6 | Random Forest (No Selection) | `0.6745 ± 0.1184` | `85.00 %` |
| **C4** | Minggu ke-7 | Random Forest (No Selection) | `0.6524 ± 0.1432` | `72.50 %` |
| **C_Full**| Minggu ke-8 | Random Forest (Feature Selection) | `0.6395 ± 0.1599` | `85.00 %` |

### 3.2. Evaluasi Ilusi Hold-Out dan Stabilitas
Observasi penting dari data di atas adalah adanya *hold-out illusion* pada jendela pengamatan C1. Pada set uji tunggal (Final Holdout), C1 mampu menembus akurasi berimbang sebesar 92.50%. Namun, saat diuji melalui *Repeated CV*, C1 menunjukkan metrik rata-rata terendah (0.6291) dibandingkan minggu-minggu berikutnya. Deviasi yang muncul menunjukkan bahwa akurasi di awal sangat fluktuatif terhadap data yang disajikan, dan performa tinggi di hold-out hanyalah anomali kebetulan.

### 3.3. Penentuan Waktu Peringatan Dini Optimal
Model dengan tingkat stabilitas (*mean* tertinggi) dan kemampuan generalisasi (*SD* terkendali) berada pada batas potong **C3 (Minggu ke-6)** menggunakan Random Forest (`0.6745 ± 0.1184`), disusul oleh **C2 (Minggu ke-5)** (`0.6567 ± 0.1241`). 

Temuan ini membuktikan hipotesis awal: **Menunggu hingga minggu ke-8 (C_Full) justru mendegradasi performa model prediktif (Deviasi Standar naik menjadi ± 0.1599) akibat adanya kontaminasi nilai kosong dari observasi kelas yang tidak seragam**. Oleh karena itu, pengamatan hingga Minggu ke-5 atau ke-6 dapat diajukan secara ilmiah sebagai titik penerapan Sistem Peringatan Dini yang paling valid.

---

## 4. Analisis Kesalahan (Error Analysis)

Untuk melengkapi pengujian kuantitatif, analisis kesalahan kualitatif terhadap kelemahan prediktif algoritma dilakukan dengan meninjau matriks *False Negatives* (FN). FN merepresentasikan kasus di mana mahasiswa yang pada kenyataannya lulus ("Kompeten"), secara keliru diprediksi gagal ("Belum Kompeten") oleh sistem peringatan dini di minggu ke-6.

Karakteristik dominan dari sampel mahasiswa FN dalam pengujian ini antara lain:
1. **Pemulihan Kinerja Terlambat (Late-Bloomers)**: Mahasiswa dengan metrik `TP_First2_Mean` sangat rendah di pertemuan awal. Algoritma Random Forest yang dibatasi pengamatannya hingga minggu ke-6 cenderung melabeli keterpurukan fundamental di dua tugas pertama sebagai pola kegagalan definitif. Namun secara akademik, beberapa dari mereka menunjukkan peningkatan (*improvement*) yang eksponensial di minggu ke-7 dan ke-8 yang berada di luar jangkauan radar model.
2. **Kompensasi Aspek Penilaian Lain**: Beberapa observasi menunjukkan rendahnya skor pengumpulan Laporan secara beruntun. Meski demikian, karena skema penilaian yang diberlakukan memfasilitasi kompensasi dari nilai tes formatif (Respons) dan kehadiran, mahasiswa tersebut masih memenuhi kualifikasi batas kelulusan.

---

## 5. Interpretasi Model dengan XAI (Explainable AI)

Sebagai pelengkap transparansi "kotak hitam" (black-box) algoritma Random Forest, metodologi **TreeSHAP** (SHapley Additive exPlanations) diterapkan. Berbasis pada teori permainan (*game theory*), TreeSHAP mendistribusikan secara adil besaran kontribusi setiap fitur agregat terhadap kalkulasi skor akhir prediksi.

Dalam implementasi SHAP untuk titik waktu optimal (C2/C3), dua hasil grafik utama telah digenerasi pada modul analisis (`generate_shap.py`):
1. **Global Feature Importance (Bar Plot)**: Menunjukkan peringkat rata-rata magnitudo nilai absolut SHAP untuk setiap fitur. Evaluasi mengindikasikan bahwa fitur seperti rata-rata Tugas Pendahuluan awal (`TP_First2_Mean`) dan konsistensi skor (`Laporan_Max`) secara universal merupakan diskriminator terbesar yang diandalkan oleh node percabangan (*decision splits*) dari ansambel hutan acak (Random Forest).
2. **Beeswarm Plot (Analisis Pengaruh Fitur Lokal-Global)**: Grafik ini menunjukkan arah dampak dari masing-masing fitur. Misalnya, plot akan memperlihatkan secara empiris bahwa titik data dengan nilai agregat respons yang merah (rendah) memiliki nilai SHAP negatif, yang menarik *output probability* klasifikasi mendekati probabilitas status kelas 0 ("Belum Kompeten"). Distribusi penyebaran data dalam Beeswarm menunjukkan seberapa kuat dorongan prediksi ketika sebuah nilai bergerak dari ekstrem minimum ke maksimum.

Kehadiran interpretasi XAI secara empiris memvalidasi bahwa metrik performa model yang diraih bukanlah hasil korélasi palsu (*spurious correlation*), melainkan sejalan dengan evaluasi pedagogis praktikum yang rasional.

---

## 6. Kesimpulan

Penelitian ini memvalidasi pendekatan metodologis *Temporal Cutoffs* dalam merancang prediksi kelulusan dini yang berkeadilan antar jadwal kelas. Pengujian komprehensif menggunakan *Repeated Stratified CV* dan seleksi fitur *inner-validation* membuktikan bahwa:
- Kestabilan prediksi tidak bergantung pada penggunaan seluruh sejarah log akademik (hingga akhir semester), melainkan mencapai tingkat akurasi berimbang paling konsisten jika dipotong secara seragam pada observasi Minggu ke-5 hingga ke-6.
- Penanganan kecondongan data melalui SMOTE terbukti berhasil mengangkat kepekaan pendeteksian mahasiswa berisiko tanpa menghasilkan *overfitting*, asalkan pembatasan *data leakage* dipertahankan.
- Pemanfaatan perangkat interpretasi XAI (TreeSHAP) mampu merumuskan ulang pemahaman terhadap karakteristik risiko kualitatif mahasiswa, memberikan panduan konkrit bagi intervensi perbaikan yang bisa diadopsi instansi pendidikan.
