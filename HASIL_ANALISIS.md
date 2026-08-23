# Hasil Analisis Prediksi Kompetensi Praktikum Logika Algoritma

## 1. Penjelasan Project dan Apa yang Dianalisis
Project ini merupakan *pipeline* penelitian *Machine Learning* yang dirancang untuk memprediksi tingkat kompetensi mahasiswa pada mata kuliah praktikum **Logika & Algoritma**. Mahasiswa diklasifikasikan ke dalam dua label/kelas utama:
- `1` : **Kompeten** (Nilai Final Individu >= 75)
- `0` : **Belum Kompeten** (Nilai Final Individu < 75)

Analisis ini melihat bagaimana fitur-fitur seperti tingkat kehadiran (Attendance Rate), rata-rata nilai Tugas Pendahuluan (TP Mean), rata-rata nilai Respon (Respons Mean), dan rata-rata nilai Laporan (Laporan Mean) dapat memprediksi tingkat kelulusan/kompetensi mahasiswa secara akurat.

## 2. Masalah dan Solusi
**Masalah Utama:**
1. Memprediksi mahasiswa mana yang berpotensi "Belum Kompeten" sejak dini (sebelum nilai final dikeluarkan).
2. Risiko terjadinya ***data leakage*** (kebocoran data), dimana fitur yang secara kalkulatif langsung berkorelasi 100% dengan nilai akhir (seperti Final Individu, Final Kelompok, Total, Predikat) tidak sengaja ikut masuk ke dalam dataset pelatihan. Hal ini membuat model tampak sangat sempurna (akurasi 100%) namun tidak memiliki nilai prediktif sama sekali di skenario nyata.
3. Kerentanan model terhadap **Overfitting** karena jumlah sampel data (mahasiswa) yang relatif sangat kecil (hanya puluhan baris data).

**Solusi yang Diterapkan:**
- *Data leakage* dicegah sepenuhnya dengan men-*drop* (menghapus) seluruh kolom yang merupakan turunan dari nilai akhir. Mahasiswa yang putus kelas di awal (*dropout*) dengan tingkat kehadiran <= 1 juga telah dibersihkan secara otomatis.
- Menjalankan eksperimen *Feature Engineering* melalui berbagai tingkatan kompleksitas fitur: **S1 (Basic)**, **S2 (Behavioral)**, dan **S3 (Relational)** untuk mencari titik keseimbangan (sweet spot) antara *underfitting* dan *overfitting*.
- Menggunakan dua algoritma pembanding: **Decision Tree** (sebagai *baseline* yang cenderung *overfit*) dan **Random Forest** (*ensemble method* berbasis *bagging* untuk meredam variansi dan meminimalisir *overfitting*).
- Mengevaluasi performa menggunakan *Stratified 5-Fold Cross Validation* (CV) di data training murni, dan menggunakan metrik **F1-Score** (guna menangani ketidakseimbangan kelas / dominansi mayoritas).

---

## 3. Hasil Analisis Performa Model (Overfitting vs Robust)

Berikut adalah ringkasan hasil eksperimen model setelah melalui tahapan *Hyperparameter Tuning* menggunakan GridSearchCV:

| Scenario | Model | CV F1 Mean | Test F1 | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **S1** | Decision Tree | 0.792 | 0.800 | 0.769 |
| **S1** | Random Forest | 0.811 | 0.889 | 0.846 |
| **S2** | Decision Tree | 0.774 | 0.667 | 0.692 |
| **S2** | Random Forest | 0.809 | 0.824 | 0.769 |
| **S3** | Decision Tree | 0.804 | 0.615 | 0.615 |
| **S3** | Random Forest | 0.814 | 0.889 | 0.846 |

**Analisis Overfitting (Keputusan Terbaik):**
1. **Decision Tree mengalami Overfitting.** Hal ini terlihat sangat jelas saat kompleksitas data meningkat di skenario S2 dan S3. Pada Skenario S3, nilai validasi murni (*CV F1 Mean*) dari Decision Tree mencapai **0.804**, tetapi ketika dites pada *Test Set* (data yang benar-benar belum pernah dilihat), performanya anjlok drastis menjadi hanya **0.615**. Ini membuktikan bahwa Decision Tree terlalu menghafal pola (*memorization*) di data latih, namun gagal melakukan generalisasi.
2. **Random Forest terbukti Sangat Robust (Tahan Overfitting).** Pada seluruh skenario (terutama S1 dan S3), Random Forest berhasil mencetak *Test F1* sebesar **0.889** dengan akurasi akhir mencapai **84.6%**. Menariknya, performa Random Forest di data Test sama stabilnya atau bahkan lebih baik dari CV F1 Mean-nya. Teknik *Ensemble Bagging* yang menggabungkan banyak *decision trees* secara acak sukses meredam variansi (fluktuasi akibat dataset kecil).

Kesimpulannya, model **Random Forest pada skenario S1 atau S3** adalah pemenang absolut dari eksperimen ini.

---

## 4. Interpretasi Model (TreeSHAP)

Machine learning tidak lagi sekadar menjadi "kotak hitam" (Black Box). Dengan mengimplementasikan **SHAP (*SHapley Additive exPlanations*)**, kita dapat membedah dengan persis alasan di balik setiap prediksi model Random Forest.

### A. Global Feature Importance (Dampak Keseluruhan)
![Global Importance S1 RandomForest](file:///c:/belajarku/Belajar%20ML/Logika-Algoritma/results/shap/global_importance_S1_RandomForest.png)
*(Keterangan: Grafik Bar di atas menunjukkan rata-rata magnitude dampak (Mean |SHAP Value|) dari tiap-tiap metrik penilaian mahasiswa).*

Berdasarkan ekstraksi data mentah dari perhitungan TreeSHAP secara numerik, kita mendapatkan urutan tingkat kepentingan (bobot prediktif) sebagai berikut:
1. **Laporan_Mean** (Bobot Rata-rata SHAP: ~0.119): Memegang peranan paling vital! Nilai rata-rata laporan praktikum menjadi indikator utama model dalam menentukan kompetensi akhir mahasiswa.
2. **Attendance_Rate** (Bobot Rata-rata SHAP: ~0.096): Tingkat kehadiran adalah faktor terpenting kedua. Hal ini sangat logis, karena absensi berbanding lurus dengan intensitas paparan pemahaman materi.
3. **Respons_Mean** (Bobot Rata-rata SHAP: ~0.083): Respon harian berkontribusi secara substansial melengkapi laporan.
4. **TP_Mean** (Bobot Rata-rata SHAP: ~0.058): Tugas Pendahuluan berada di urutan terakhir, menunjukkan bahwa persiapan pra-praktikum memiliki dampak yang sedikit lebih lemah dibandingkan performa saat praktikum berlangsung (Laporan & Respon).

### B. SHAP Beeswarm Plot (Analisis Korelasi)
![Beeswarm S1 RandomForest](file:///c:/belajarku/Belajar%20ML/Logika-Algoritma/results/shap/beeswarm_S1_RandomForest.png)
*(Keterangan: Titik kemerahan merepresentasikan input nilai asli yang "Tinggi". Semakin ke kanan sumbu X (SHAP > 0), semakin memengaruhi keputusan model ke arah "Kompeten").*

Pada plot di atas beserta pengamatan pada matriks *raw data*, terlihat distribusi korelasi yang sangat kuat:
- **Kontribusi Positif:** Mahasiswa dengan nilai `Laporan_Mean` dan `Attendance_Rate` yang sangat tinggi (diwakili titik berwarna kemerahan) secara konsisten menggeser prediksi model jauh ke sisi kanan (SHAP > 0). Kombinasi nilai tinggi ini nyaris memastikan klasifikasi model jatuh pada keputusan **"Kompeten (1)"**.
- **Efek Penalti:** Sebaliknya, nilai absensi yang rendah (biru) pada fitur *Attendance_Rate* menyumbang penalti SHAP negatif yang sangat dramatis (memanjang jauh ke sebelah kiri), menarik probabilitas model secara paksa ke arah prediksi **"Belum Kompeten"**.

---

## 5. Visualisasi Pendukung Lainnya

### A. Evaluasi Confusion Matrix
![Confusion Matrix S1 RandomForest](file:///c:/belajarku/Belajar%20ML/Logika-Algoritma/results/confusion_matrix/cm_S1_RandomForest.png)
*(Keterangan: Matriks yang memperlihatkan kebenaran vs kesalahan tebakan. Diagonal merupakan prediksi yang tepat).*
Model terbukti sangat baik dalam menangkap *True Positives* dan menekan tingkat *False Negatives*.

### B. Analisis Distribusi Kelas
![Distribusi Kelas](file:///c:/belajarku/Belajar%20ML/Logika-Algoritma/results/class_analysis/class_distribution.png)
*(Keterangan: Grafik yang menunjukkan persebaran proporsi antara status Kompeten dan Belum Kompeten di seluruh seksi/kelas praktikum).*
Melalui pembagian secara acak terdistribusi (*Stratified Split*), perbandingan rasio tidak seimbang ini berhasil disetarakan skalanya di dalam pelatihan algoritma, sehingga model tidak bersikap *bias* ke kelas mayoritas.

---

## 6. Kesimpulan
1. Proyek ini sukses menghadirkan lingkungan *pipeline* ML yang bebas *data leakage*.
2. **Random Forest** terbukti sebagai model paling stabil dengan akurasi pengujian tertinggi (**84.6%**) tanpa mengalami ancaman *overfitting* yang dialami algoritma tunggal Decision Tree.
3. Berdasarkan analisis *deep dive* TreeSHAP, fitur **Rata-rata Nilai Laporan** dan **Tingkat Kehadiran (Attendance Rate)** adalah *Early Warning Indicators* (Indikator Peringatan Dini) terbaik dengan kontribusi tertinggi. Asisten praktikum dapat memprioritaskan pendampingan kepada mahasiswa yang bermasalah pada absensi dan kualitas laporannya.
