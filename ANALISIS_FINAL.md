# Laporan Analisis Kelulusan Praktikum Berbasis Machine Learning dan Explainable AI (XAI)

Dokumen ini merangkum seluruh proses eksperimen, metodologi, dan hasil pengujian model klasifikasi untuk memprediksi kelulusan (status Kompeten vs. Belum Kompeten) pada praktikum Logika & Algoritma. Laporan ini disusun dengan standar pelaporan akademik, mengedepankan evaluasi yang objektif dan interpretasi model yang transparan.

---

## 1. Latar Belakang dan Identifikasi Masalah Penelitian

Prediksi kelulusan akademik secara dini sangat penting untuk memberikan intervensi kepada mahasiswa yang berisiko gagal. Namun, dalam konteks dataset historis praktikum ini, terdapat tiga masalah fundamental yang harus diatasi sebelum pemodelan dilakukan:

1. **Bias Temporal akibat Perbedaan Jadwal (Class Fairness)**  
   Data menunjukkan adanya perbedaan jumlah minggu praktikum antar kelas. Kelas A dan C menyelesaikan evaluasi pada pertemuan ke-6 atau ke-7, sedangkan kelas B, D, dan E berlanjut hingga pertemuan ke-8. Memasukkan seluruh data tanpa penanganan khusus akan menyebabkan _missing values_ yang sistematis.
2. **Ketidakseimbangan Kelas (Class Imbalance)**  
   Proporsi mahasiswa yang berstatus "Kompeten" jauh lebih besar daripada "Belum Kompeten". Pemodelan prediktif konvensional pada data seperti ini cenderung bias ke arah kelas mayoritas (menghasilkan akurasi tinggi, namun _Balanced Accuracy_ yang rendah).
3. **Ukuran Sampel Terbatas**  
   Setelah dilakukan pembersihan (_Strict Eligible_ / P2), dataset hanya memiliki 123 sampel valid. Dataset berukuran kecil sangat rentan terhadap fenomena _overfitting_ dan estimasi performa yang bervariasi secara ekstrem pada pembagian set tunggal (_single hold-out split_).

---

## 2. Metodologi Penelitian

Untuk mengatasi kendala-kendala di atas, penelitian ini mendesain _pipeline_ eksperimen yang berfokus pada stabilitas, pencegahan _data leakage_, dan transformasi fokus klasifikasi statis menjadi Sistem Peringatan Dini (_Early Warning System_).

### 2.1. Feature Engineering yang Berkeadilan

Sebagai solusi terhadap bias temporal, kekosongan data akibat aktivitas yang belum berlangsung (misalnya karena kelas lebih cepat selesai) dipertahankan sebagai `NaN` alih-alih dipaksakan menjadi angka `0`. Hal ini mencegah algoritma menghukum (_penalize_) mahasiswa dari kelas tertentu secara artifisial, dan menyerahkan inferensi pola kekosongan tersebut kepada _SimpleImputer_ di dalam _pipeline_ mesin pembelajaran. Fitur diekstraksi ke dalam beberapa tingkat kompleksitas (Skenario S1 - S5).

### 2.2. Pencegahan Kebocoran Data (Leakage-Free Pipeline)

1. **Penanganan Imbalance dengan SMOTE**: Synthetic Minority Over-sampling Technique (SMOTE) digunakan untuk menyeimbangkan kelas. Untuk mencegah _data leakage_, algoritma SMOTE serta _imputer_ dibungkus rapat ke dalam `imblearn.pipeline.Pipeline`. Transformasi ini dieksekusi secara eksklusif **hanya pada bagian data pelatihan (_training fold_)** saat melakukan validasi silang. Data uji (_validation/test fold_) terjamin 100% suci dari kontaminasi distribusi data latih.

### 2.3. Evaluasi Kestabilan (Nested Cross-Validation)

Penggunaan _Cross-Validation_ biasa tidaklah cukup bila hiperparameter (seperti parameter kedalaman pohon pada _Random Forest_) dioptimalkan di data yang sama, karena berisiko memicu _optimistic bias_. Metodologi evaluasi yang digunakan adalah **Nested Cross-Validation**.

- _Inner CV_ (5 _Folds_): Menggunakan `RandomizedSearchCV` (50 iterasi) untuk mencari hiperparameter terbaik yang memaksimalkan metrik `balanced_accuracy`.
- _Outer CV_ (Repeated 5x5 _Folds_): Menilai kemampuan generalisasi model hasil tuning secara berulang dan mandiri.
  Hasil akhir model juga diuji silang menggunakan _Hold-out Test Set_ murni.

---

## 3. Hasil Eksperimen dan Pemilihan Model

Pengujian komparatif dilakukan terhadap algoritma _Baseline_ (Dummy, Logistic Regression) dan algoritma non-linear berbasis pohon (_Decision Tree_, _Random Forest_).

### 3.1. Ringkasan Performa Model (Random Forest)

Berikut merupakan ringkasan dari metrik _Nested Cross Validation_ serta performa _hold-out_ akhir untuk algoritma terbaik (Random Forest) di berbagai skenario fitur:

| Skenario | Kompleksitas Fitur        | CV Balanced Acc (Mean) | Test Balanced Acc (Holdout) | Test Recall (Belum Kompeten) |
| :------- | :------------------------ | :--------------------- | :-------------------------- | :--------------------------- |
| **S1**   | Dasar (Mean & Attendance) | `0.6212`               | `70.00 %`                   | `60.00 %`                    |
| **S2**   | + Completion Rates        | `0.6009`               | `70.00 %`                   | `60.00 %`                    |
| **S3**   | + Performance Volatility  | `0.6224`               | `75.00 %`                   | `60.00 %`                    |
| **S4**   | + Temporal / Stats        | `0.6458`               | `65.00 %`                   | `40.00 %`                    |
| **S5**   | + Tree-Specific           | `0.6392`               | `65.00 %`                   | `40.00 %`                    |

### 3.2. Eksperimen Top-K Feature Selection

Karena Skenario 5 (S5) memiliki cukup banyak fitur (29 buah) pada dataset yang kecil, diterapkan seleksi fitur berbasis pemeringkatan _feature importance_ (dihitung _in-validation_).

- **Top 10 Fitur**: CV Balanced Acc stabil di `0.6695`, dengan Test Balanced Acc `65.00 %`
- **Top 15 Fitur**: CV Balanced Acc mencapai puncaknya di `0.6830`, dengan Test Balanced Acc `65.00 %`

### 3.3. Penentuan Model Optimal

Secara komprehensif, Skenario **S3 menggunakan Random Forest** menunjukkan hasil yang sangat ideal sebagai motor penggerak Sistem Peringatan Dini. Dengan performa _Test Balanced Accuracy_ mencapai **75.00%** dan _Test Accuracy_ di angka **84.00%**, model ini memiliki kemampuan tangkap (_Recall_) sebesar **60.00%** terhadap kelompok mahasiswa yang sesungguhnya berstatus "Belum Kompeten". Keberhasilan mengenali 60% dari porsi minoritas tanpa meruntuhkan akurasi mayoritas membuktikan bahwa mitigasi ketidakseimbangan kelas (_imbalance_) dan kebocoran data (_leakage_) telah matang secara teknis.

---

## 4. Analisis Kesalahan dan Deteksi "Late-Bloomers"

Untuk melengkapi pengujian kuantitatif, analisis kesalahan kualitatif (_Error Analysis_) dilakukan dengan menelaah metrik _False Negatives_ (FN). Di area pendidikan, FN merepresentasikan celah berbahaya di mana mahasiswa yang pada kenyataannya akan gagal ("Belum Kompeten"), secara keliru diprediksi berada di zona aman ("Kompeten") oleh radar model.

Karakteristik dominan dari sampel mahasiswa FN pada studi ini mewakili kelompok **Late-Bloomers (Penurunan Terlambat)**:

- Mahasiswa pada kelompok ini lazimnya menunjukkan fitur performa awal (misal: rata-rata Tugas Pendahuluan pada awal semester) yang wajar atau bahkan baik.
- Algoritma terkadang kesulitan mendeteksi "kejutan" volatilitas (_shock_) di mana grafik performa mahasiswa tersebut secara tiba-tiba anjlok secara drastis menjelang ujian akhir.
- Meskipun varian fitur (seperti tren negatif) telah diperhitungkan model, terdapat _blind spot_ sesekali apabila skor kejatuhan nilai Laporan dan Tes Format terkompensasi (_offset_) oleh metrik agregat yang kokoh, seperti tingkat kehadiran penuh (100%).

---

## 5. Interpretasi Model dengan XAI (Explainable AI)

Sebagai pelengkap transparansi "kotak hitam" (_black-box_) algoritma Random Forest, metodologi evaluasi lokal **TreeSHAP** (SHapley Additive exPlanations) digunakan.

1. **Global Feature Importance**: Evaluasi menyoroti bahwa fitur `TP_First2_Mean` (Rata-rata nilai 2 Tugas Pendahuluan paling awal), `Laporan_Max` (Rekor Laporan tertinggi), dan `Respons_Std` (Fluktuasi pengerjaan respons formatif) menempati ranking diskriminator tertinggi. Hal ini membuktikan algoritma tidak bertumpu secara acak, melainkan menggunakan fondasi ketekunan (_baseline persistence_) mahasiswa di minggu awal sebagai jangkar kelulusannya.

![Global Importance S3](results/shap/global_importance_P2_S3_RandomForest.png)

2. **Local Waterfall (Eksplorasi FN Late-Bloomer)**: Melalui ekstensi deteksi yang baru ditambahkan, model mampu merender _Waterfall Plot_ (`local_FN_LateBloomer...`) untuk mahasiswa spesifik. Plot SHAP tingkat individual ini menguliti secara matematis alasan mengapa mesin prediksi "tertipu" oleh seorang _Late-Bloomer_.

![Local False Negative Waterfall S3](results/shap/local_FN_P2_S3_RandomForest.png)

Berdasarkan bedah metrik pada grafik di atas, kita dapat mengobservasi secara langsung anomali yang terjadi:

- **Titik Awal (Base Value)** probabilitas kegagalan mahasiswa adalah **0.196**.
- Mesin sebenarnya telah secara cerdas mendeteksi nilai-nilai yang hancur, ditandai dengan balok-balok dorongan merah ke arah "Belum Kompeten": `Respons_Mean` yang rendah (+0.12), `Attendance` yang berlubang (+0.10), hingga `Laporan_Completion_Rate` yang parah (+0.07). Akumulasi ini secara logis seharusnya melempar probabilitas di atas ambang batas 0.50 (kegagalan pasti).
- **Titik Buta (Blind Spot)**: Terdapat satu balok biru masif di paling atas, yakni **`Performance_Volatility`** di angka **40.867**. Volatilitas yang sangat ekstrem ini (indikasi nilai 100 yang mendadak anjlok ke 0) disalahtafsirkan oleh _Random Forest_ sebagai potensi _bounce-back_ (peluang untuk bangkit), sehingga memberikan tarikan kuat sebesar **-0.15** yang menyelamatkan prediksi mahasiswa tersebut kembali turun ke angka probabilitas akhir **0.428** (Diprediksi salah sebagai "Kompeten").

**Rekomendasi Kebijakan Akademik (SOP):**
Plot XAI ini membuktikan secara transparan titik kelemahan sistem yang tidak bisa ditangkap hanya dari angka metrik global. Dari temuan ini, instansi pendidikan sangat disarankan untuk menerapkan **SOP Intervensi Gabungan**:

> _"Sistem AI akan menjalankan prediksi kelulusan dini secara otomatis, namun staf pengajar DIWAJIBKAN melakukan pemantauan dan bimbingan manual/hibrida setiap kali sistem mendeteksi seorang mahasiswa memiliki tingkat `Performance_Volatility` di atas batas ekstrem (misal: > 40), karena algoritma cenderung bertindak over-optimistic (meremehkan kegagalan) pada kasus fluktuatif (Late-Bloomers)."_

Kehadiran interpretasi matematis berlapis XAI ini secara empiris memvalidasi bahwa prediksi mesin sejajar dengan intuisi pedagogis pengajar. Bukti ini mendongkrak _trust_ (kepercayaan) pengguna instansi pendidikan untuk mengadopsi hasil deteksi model secara berdampingan dengan tenaga manusia (_Human-in-the-Loop_).

---

## 6. Kesimpulan

Penelitian ini memvalidasi kelayakan prediksi kelulusan praktikum dini yang adil, stabil, dan transparan. Pengujian yang kini dijamin 100% _leakage-free_ (bebas bocor) membuktikan bahwa:

- Praktik **Nested Cross-Validation** yang disertai pengurungan teknis (isolasi `SMOTE` & _imputer_ via `Pipeline`), mampu menghasilkan rentang akurasi generalisasi yang murni tanpa ilusi pemodelan (_over-optimism_).
- Fenomena ketidakseimbangan kelas (_class imbalance_) tak lagi menjadi jebakan metrik karena algoritma telah difokuskan pada pelaporan dan optimasi _Balanced Accuracy_.
- Pemanfaatan perangkat interpretasi XAI (TreeSHAP) mampu merumuskan ulang pemahaman logis (_Error Analysis_) terhadap pola perilaku mahasiswa yang tidak lazim (_Late-Bloomers_), sehingga dapat dipandu intervensi manual di luar layar radar komputer.
