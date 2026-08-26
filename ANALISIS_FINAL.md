# Analisis Final: Prediksi Kompetensi Praktikum Logika Pemrograman

Dokumen ini merupakan rangkuman evaluasi, analisis kendala model, penilaian kualitas data mentah, serta temuan *explainability* (XAI) dari keseluruhan eksperimen (Eksperimen Utama, Robustness, dan Feature Selection).

---

## 1. Apakah Data Mentahnya "Buruk"?

Secara objektif, data mentah Anda **tidak buruk**, tetapi **sangat terbatas (kecil) dan berisik (*noisy*)**. Ada beberapa karakteristik dari data yang mendikte seluruh perilaku model:

### A. *Small Sample Size* & Variansi Tinggi
Populasi utama Anda (P2 - Strict Eligible) hanya berisi **122 mahasiswa**. 
- Dalam pemisahan 80% Train / 20% Test, model di-*training* pada ~97 sampel dan diuji pada **~25 sampel**.
- *Dummy Classifier* (memprediksi kelas mayoritas secara membabi buta) mendapatkan akurasi 60%. Ini berarti dari 25 sampel *test set*, sekitar 15 mahasiswa adalah "Kompeten" dan 10 "Belum Kompeten".
- Pada skala 25 sampel, **1 mahasiswa yang salah tebak akan mengubah akurasi sebesar 4%**. Lonjakan akurasi dari 60% ke 64% ke 68% hanyalah perbedaan dari 1 atau 2 mahasiswa yang tebakannya kebetulan benar/salah. Ini membuat hasil *test set* memiliki variansi yang sangat tinggi dan sulit dipercaya sebagai indikator tunggal.

### B. *Label Noise* & Inkonsistensi Manusia
Target kita adalah `Competency_Label` yang diturunkan dari nilai `Final_Individu` (>= 75). 
Masalahnya, nilai akhir sering kali dipengaruhi faktor eksternal di luar 8 pertemuan praktikum awal, seperti:
- Mahasiswa rajin namun mendadak sakit atau *blank* saat ujian akhir.
- Mahasiswa malas namun berhasil "menebak" atau menyontek saat ujian akhir.
- Subjektivitas asisten praktikum dalam menilai laporan harian vs objektivitas mesin ujian.
Faktor-faktor ini menciptakan *irreducible error* (keributan yang tidak bisa dihilangkan) di mana data historis (praktikum M1-M8) tidak akan pernah bisa memprediksi ujian akhir dengan akurasi 100%.

---

## 2. Kendala Model (Mengapa Sulit Tembus 75%+)

### A. Overfitting & "The Curse of Dimensionality"
Ketika kita beralih dari **S1** (9 fitur) menuju **S4** (32 fitur) dan **S5** (29 fitur), performa CV (*Cross-Validation*) pada *Random Forest* meningkat hingga ~68-69%. Namun, performa pada *Test Set* justru stagnan di angka 60-64%.
- Ini adalah gejala klasik **Overfitting**. Model dengan pohon keputusan (*Decision Tree / Random Forest*) dengan 122 baris dan 30+ kolom akan sangat mudah untuk "menghafal" data *training*, sehingga gagal saat menghadapi 25 data *test*.
- Hal ini terbukti pada hasil S5 dengan *Top-10 Feature Selection*: Akurasi tesnya naik menjadi 64% dibanding *Top-25/Top-29* (60%). Dengan memotong fitur menjadi 10, model dipaksa untuk lebih general dan tidak menghafal *noise*.

### B. *Data Leakage* pada Evaluasi Awal (Terpecahkan)
Sebelumnya Anda melihat angka 72% pada Random Forest (S5 Top-25). Seperti yang sudah diuji pada eksperimen `run_p2_optimization.py`, angka tersebut adalah hasil dari **Data Leakage**. Saat *feature selection* dikurung dengan benar di dalam *Nested CV*, akurasi aslinya memang turun ke 64%. Jadi, akurasi "asli" dari Random Forest untuk data ini memang berkisar di 64%-68%, bukan 72%. 

### C. *Logistic Regression* Menjadi *Baseline* Terkuat
Mengejutkannya, pada Eksperimen P2 S4, **Logistic Regression** (model paling sederhana) berhasil mencapai:
- **Test Accuracy**: 68.00%
- **Test Balanced Accuracy**: 70.00%
- **Recall (Belum Kompeten)**: 80.00%
Ini membuktikan bahwa untuk data dengan ukuran *sample* sangat kecil dan dimensi tinggi, **batas linear (*linear boundary*) jauh lebih kokoh (robust)** daripada batas non-linear kompleks yang dibuat oleh Random Forest.

---

## 3. Insight dari Feature Importance & SHAP

Meskipun akurasi terbentur di angka ~65-70%, model telah berhasil mengidentifikasi pola kelulusan (*Early Warning System*) yang sangat valid dan logis. Dari *output feature importance* pada Random Forest S5, kita melihat:

1. **`Laporan_Trend` (15.5%)** & **`Laporan_Mean` (15.2%)**: Ini adalah 2 fitur absolut terpenting. Model melihat bahwa rata-rata nilai laporan sangat krusial, dan yang **lebih krusial** adalah trennya: apakah nilai laporannya membaik atau memburuk dari paruh pertama ke paruh kedua praktikum.
2. **`Respons_Trend` (10.3%)**: Tren kecepatan atau kualitas respons (Tanya Jawab / partisipasi) juga menjadi indikator sangat penting.
3. **Fitur Kehadiran Terpinggirkan**: `Attendance_PreFinal_Rate` hanya memiliki kontribusi sangat kecil (0.3%). Ini membuktikan hipotesis awal bahwa mahasiswa yang datang (*hadir*) belum tentu kompeten jika nilai laporan dan respons mereka buruk. *Quality over quantity*.

Berdasarkan *beeswarm plot* dari **SHAP**:
- Nilai SHAP tinggi (merah) pada `Laporan_Trend` akan sangat kuat mendorong prediksi mahasiswa menjadi "Kompeten" (1). 
- Sebaliknya, penurunan tren (biru) pada respons dan laporan adalah "Lampu Merah" (red flag) yang mendeteksi mahasiswa yang akan gagal di ujian akhir (mendorong prediksi ke "Belum Kompeten").

---

## 4. Kesimpulan untuk Skripsi

Kendala eksperimen ini **bukan** berarti skripsi Anda gagal. Sebaliknya, hal-hal inilah yang harus diangkat sebagai "Finding" utama dalam skripsi dan *paper* Anda:

1. **Jangan klaim akurasi artifisial tinggi**. Akui bahwa pada ukuran *sample* 122, akurasi stabil berada di rentang 65-70%.
2. **Kekuatan ada pada SHAP**. Buktikan bahwa sistem peringatan dini (*Early Warning System*) berbasis ML dapat mendeteksi mahasiswa berisiko gagal tidak hanya dari kehadirannya, melainkan dari **penurunan tren laporan dan respons**. Tunjukkan gambar SHAP `beeswarm` dan `local_TP`/`local_FN` sebagai kontribusi penelitian (XAI).
3. **Feature Selection itu Kritis**. Tunjukkan dalam metodologi Anda bahwa melakukan seleksi fitur secara ketat (S5 Top-10) atau mendesain fitur *domain-knowledge* secara manual mampu mengurangi efek *Curse of Dimensionality* pada data yang kecil.
4. **Logistic Regression dan Decision Tree adalah model yang lebih praktis**. Untuk ukuran sampel kecil, *Logistic Regression* menahan *overfitting*, sedangkan *Decision Tree* (yang sudah dituning) menghasilkan model *white-box* yang *rules*-nya bisa dibaca langsung oleh dosen pengajar untuk melakukan intervensi (misal: "Jika Laporan_Trend < -10 dan TP_Mean < 60, panggil mahasiswa").

**Next Step**:
- Gunakan grafik *SHAP Global Importance* dan *Beeswarm* untuk presentasi hasil.
- Bila ada waktu, bongkar berkas `error_analysis_P2_S5_RandomForest.csv` secara manual. Jika mahasiswa yang diprediksi salah secara konsisten ternyata memiliki anomali spesifik (seperti nilai UAS yang bertolak belakang drastis dengan nilai harian), Anda bisa memasukkan argumen "*Irreducible Human Error in Final Exam Grading*" sebagai batasan penelitian di skripsi Anda.
