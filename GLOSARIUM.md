# Glosarium & Gambaran Besar Proyek: Klasifikasi dan Sistem Peringatan Dini Kelulusan Praktikum

Dokumen ini disusun sebagai panduan menyeluruh (helikopter *view*) untuk membantu Anda dalam menyusun naskah skripsi, paper SINTA 2, ataupun menghadapi sidang pertanggungjawaban penelitian. Di sini dijelaskan konsep dasar, alasan pemilihan teknologi, kerangka solusi, metrik evaluasi mutakhir, serta istilah-istilah teknis penting sesuai dengan standar audit penelitian terbaru.

---

## 1. Gambaran Besar Proyek (*The Big Picture*)
Proyek ini adalah sebuah penelitian berbasis *Machine Learning* yang bertujuan untuk mengubah kumpulan log nilai mingguan praktikum (Tugas Pendahuluan, Laporan, Respons) menjadi sebuah **Sistem Peringatan Dini (*Early Warning System*)**. 

Bukan sekadar melakukan klasifikasi kelulusan di akhir semester, sistem ini didesain secara spesifik menggunakan pendekatan **Temporal Cutoff** untuk mendeteksi seawal mungkin (misalnya pada minggu ke-5 atau ke-6) mana mahasiswa yang berisiko "Belum Kompeten" agar pengajar dapat melakukan intervensi penyelamatan (remedial, bimbingan).

---

## 2. Metodologi dan Algoritma Utama

| Teknologi / Metode | Alasan Penggunaan |
| :--- | :--- |
| **Python & Scikit-Learn** | Standar industri dan akademik yang paling kokoh untuk perancangan *machine learning pipeline*. |
| **Random Forest & Decision Tree** | Algoritma berbasis pohon (*tree-based*) yang tahan terhadap pencilan (*outliers*) dan tidak mengharuskan data berdistribusi normal. Sifatnya *interpretable* dan mendukung pembacaan *feature importance* via **TreeSHAP**. *Decision Tree* merepresentasikan logika sederhana (sebagai *baseline* non-linear), sementara *Random Forest* menghasilkan prediksi *ensemble* yang kuat. |
| **SMOTE (*Synthetic Minority Over-sampling Technique*)** | Data mahasiswa "Kompeten" (78) terlalu mendominasi dibandingkan yang "Belum Kompeten" (11). SMOTE mensintesis data bayangan kelas minoritas agar algoritma tidak bias menebak "lulus semua". |
| **Nested Cross-Validation (CV Bersarang)** | Digunakan untuk mencegah ***Selection Bias***. Pemilihan hyperparameter, *balancing* (SMOTE), dan fitur (*feature engineering*) hanya boleh dilakukan di *Inner Loop CV*. Sedangkan laporan skor akhir hanya dievaluasi pada *Outer Loop Test Fold*. |

---

## 3. Metrik Evaluasi Spesifik (Imbalanced & EWS)

Pada set data yang sangat timpang (89 total vs 11 target minoritas), Akurasi dan ROC-AUC bisa menipu. Evaluasi sistem Anda dikunci pada metrik berikut:

1. **Precision BK (Presisi Belum Kompeten)**  
   Dari semua mahasiswa yang "diberi label peringatan/gagal" oleh radar AI, berapa persen yang *nyata-nyata* akan gagal? Presisi tinggi berarti sistem jarang mengeluarkan **Alarm Palsu** (*False Positives*). Ini penting untuk efisiensi operasional intervensi dosen.
2. **Recall BK (Sensitivitas Minoritas)**  
   Dari total seluruh mahasiswa yang memang akhirnya "Gagal", berapa persen yang berhasil diselamatkan/ditangkap oleh radar AI? Recall tinggi berarti sistem meminimalisir mahasiswa berisiko yang lolos tak terdeteksi (*False Negatives*).
3. **PR-AUC (*Precision-Recall Area Under Curve*)**  
   Standar emas baru untuk data sangat *imbalanced*. Jauh lebih representatif dan ketat dibandingkan kurva ROC-AUC karena PR-AUC berfokus langsung pada performa prediksi kelas minoritas.
4. **Balanced Accuracy (Akurasi Berimbang)**  
   Rata-rata dari Sensitivitas kelas positif dan spesifisitas kelas negatif. Model harus pintar menebak lulus sekaligus pintar menebak gagal.
5. **Mean ± SD (Standar Deviasi) / 95% CI**  
   Karena sampel sangat kecil, akurasi pada satu iterasi CV bisa melompat drastis (±33%). Skor rata-rata *telanjang* tidak valid. Wajib menuliskan skor sebagai rentang kepercayaan atau deviasi stabil (Contoh: `0.947 ± 0.07`).

---

## 4. Glosarium Istilah Teknis (Technical Terms) untuk Sidang/Jurnal

* **Early Warning System (EWS)**: Sistem Peringatan Dini. Memprediksi potensi kegagalan secara prediktif sebelum keputusan final (nilai akhir) diterbitkan.
* **Temporal Cutoff (Jendela Waktu)**: Mekanisme memotong seluruh catatan data mentah pada batas waktu tertentu (Misal C1, C2, C3). Semua prediksi AI dihitung hanya menggunakan amunisi data dari masa lalu (sebelum cutoff).
* **Data Leakage (Kebocoran Data Temporal)**: Kesalahan metodologi fatal ketika algoritma tanpa sengaja "mengintip" atau menggunakan informasi masa depan (setelah batas *cutoff* atau *test set*) selama proses pelatihan model atau perhitungan rata-rata.
* **Selection Bias / Optimistic Bias**: Kecurangan statistik yang tidak disadari, di mana peneliti menjalankan ratusan kombinasi CV (hyperparameter, balancing, *feature set*) lalu mengambil "satu nilai yang paling tinggi" sebagai klaim generalisasi. Diselesaikan menggunakan *Nested Cross-Validation*.
* **OOF (Out-Of-Fold) Prediction**: Prediksi yang dihasilkan oleh model pada *validation fold* (data yang tidak ikut dilatih) selama *Cross-Validation*. Dalam *Repeated CV*, prediksi probabilitas OOF ini diagregasikan (rata-rata) per baris mahasiswa.
* **Explainable AI (XAI) & SHAP**: Sub-bidang AI yang menjelaskan alasan *black-box* mengambil keputusan. **SHAP** mendistribusikan kontribusi setiap fitur (misal: Kehadiran) secara wajar.
* **SHAP Sensitivity**: Fenomena ketika nilai kepentingan (kontribusi) SHAP sebuah fitur anjlok (misal dari 0.049 menjadi 0.000) hanya karena 3 subjek spesifik dihilangkan. Menunjukkan bahwa fitur tersebut bukan "Universal", melainkan sensitif terhadap kondisi tertentu.
* **False Negative (FN) / *Late-Droppers***: Mahasiswa berisiko yang lolos dari peringatan dini sistem. Secara akademik, mereka seringkali memulai semester dengan cemerlang namun terjerembap di pekan-pekan terakhir (*Late-Droppers*).
* **False Positive (FP) / *Late-Bloomers***: "Alarm Palsu". Mahasiswa yang diprediksi akan gagal, tetapi membuktikan mereka sanggup lulus. Secara akademik, mereka adalah orang yang lambat beradaptasi di awal namun mampu mengejar ketertinggalan (*Late-Bloomers*).
* **No Final Attendance**: Kasus ekstrem (*error analysis*) mahasiswa yang *dropout* fungsional (tidak mengikuti praktikum final). Deteksi kasus-kasus seperti ini harus dianggap sebagai pembuktian fungsionalitas peringatan dini.
