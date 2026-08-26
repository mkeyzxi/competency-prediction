# Glosarium & Gambaran Besar Proyek: Klasifikasi dan Sistem Peringatan Dini Kelulusan Praktikum

Dokumen ini disusun sebagai panduan menyeluruh (helikopter *view*) untuk membantu Anda dalam menyusun naskah skripsi, paper SINTA 2, ataupun menghadapi sidang pertanggungjawaban penelitian. Di sini dijelaskan konsep dasar, alasan pemilihan teknologi, kerangka solusi, metrik evaluasi, serta istilah-istilah teknis penting.

---

## 1. Gambaran Besar Proyek (The Big Picture)
Proyek ini adalah sebuah penelitian berbasis *Machine Learning* yang bertujuan untuk mengubah kumpulan log nilai mingguan praktikum (Tugas Pendahuluan, Laporan, Respons) menjadi sebuah **Sistem Peringatan Dini (Early Warning System)**. 

Bukan sekadar melakukan klasifikasi di akhir semester, sistem ini didesain untuk mendeteksi seawal mungkin (pada minggu ke-5 atau ke-6) mana mahasiswa yang berisiko "Belum Kompeten" (gagal) agar instansi pendidikan dapat melakukan intervensi penyelamatan (remedial, bimbingan).

---

## 2. Apa yang Digunakan & Kenapa Digunakan?

| Teknologi / Algoritma | Kenapa Digunakan? |
| :--- | :--- |
| **Python & Scikit-Learn** | Standar industri dan akademik yang paling kokoh untuk perancangan jalur pipa data (*pipeline*) dan klasifikasi pembelajaran mesin. |
| **Random Forest & Decision Tree** | Dipilih karena merupakan algoritma berbasis pohon (*tree-based*). Algoritma ini tidak mengharuskan data berdistribusi normal, tahan terhadap pencilan (*outliers*), dan cara pengambilan keputusannya sangat selaras (kompatibel) dengan ekstraksi transparansi nilai **TreeSHAP** (Explainable AI). *Random Forest* berfungsi menghasilkan kestabilan prediksi, sementara *Decision Tree* berfungsi merepresentasikan logika sederhana. |
| **SMOTE** *(Synthetic Minority Over-sampling Technique)* | Data lulus ("Kompeten") terlalu mendominasi dibandingkan data gagal. Jika dibiarkan, model akan menebak "lulus semua" demi akurasi tinggi (*dummy trap*). SMOTE digunakan untuk mensintesis data bayangan pada kelas minoritas sehingga mesin belajar mengenali pola mahasiswa gagal dengan seimbang. |
| **Repeated Stratified K-Fold CV** | Digunakan karena ukuran data kita kecil (hanya 123 sampel valid). Melakukan satu kali pemisahan (*hold-out split*) rentan memberikan estimasi akurasi yang "beruntung tinggi" (*Hold-out illusion*). Pengujian berulang (contoh: 25 kali diacak) memastikan model kita benar-benar stabil. |
| **Permutation Importance** *(Nested Selector)* | Mengidentifikasi fitur terpenting dengan mengacak isi suatu fitur, lalu melihat seberapa besar akurasi hancur. Fitur dengan daya hancur tertinggi berarti fitur tersebut sangat esensial. |

---

## 3. Solusi Kunci (*The Core Solution*)

Penelitian ini memecahkan masalah mendasar yang kerap diremehkan oleh peneliti lain, yaitu **Bias Durasi Kelas**. Kelas A/C selesai di minggu ke-6, sementara B/D/E di minggu ke-8.

**Solusi Ilmiah yang Diterapkan:**
Menggunakan mekanisme **Temporal Cutoff** (*Common Window*). Seluruh data mentah diseragamkan potongannya. Semua mahasiswa dinilai setara HANYA sampai batas minggu tertentu (misalnya `C2` = Minggu ke-5, `C3` = Minggu ke-6). 
Metode ini secara elegan mengubah masalah "*missing value* sistematis" menjadi sebuah eksperimen pembuktian *Sistem Peringatan Dini*: *"Buktikan pada minggu ke berapakah prediksinya paling stabil?"*

---

## 4. Metrik Evaluasi

Pada proyek dengan ketidakseimbangan kelas (*imbalanced data*), metrik akurasi biasa sangat menyesatkan. Berikut metrik utama yang digunakan:

1. **Balanced Accuracy (Akurasi Berimbang)**  
   *Metrik utama (Utara/North Star) dalam proyek ini*. Dihitung dari rata-rata Sensitivitas (kemampuan mendeteksi status Kompeten) dan Spesifisitas (kemampuan mendeteksi status Belum Kompeten). Sebuah model tidak akan mendapat nilai *Balanced Accuracy* tinggi jika ia hanya pintar menebak lulus tapi buta dalam menebak mahasiswa gagal.
2. **Mean ± SD (Standar Deviasi)**  
   Simbol kestabilan. Jika model mencetak *Test Accuracy* 95% namun memiliki SD ± 0.17 (sangat lebar deviasinya), berarti model tersebut rapuh secara generalisasi (*overfitting/hold-out illusion*). Model yang tangguh diincar pada SD yang lebih sempit (misal ± 0.11).
3. **Precision & Recall**  
   - *Precision*: Jika sistem memprediksi mahasiswa "Gagal", seberapa yakin tebakan tersebut benar?
   - *Recall*: Dari seluruh mahasiswa yang nyatanya Gagal, berapa persen yang berhasil tertangkap sistem radar peringatan dini kita?

---

## 5. Glosarium Istilah Teknis (Technical Terms) untuk Sidang/Jurnal

* **Explainable AI (XAI)**: Sebuah sub-bidang AI yang bertujuan membuat "kotak hitam" (*black box*) algoritma peramal menjadi transparan dan bisa dijelaskan secara logis kepada manusia (dosen/praktisi).
* **SHAP (SHapley Additive exPlanations)**: Metode interpretasi yang didasarkan pada Teori Permainan Koperasi (*Cooperative Game Theory*). SHAP membagi-bagikan (mendistribusikan) kontribusi setiap fitur (misal: Rata-rata Laporan) terhadap prediksi akhir (Lulus/Gagal) secara sangat adil.
* **Beeswarm Plot**: Grafik utama SHAP yang menggabungkan sebaran distribusi data dan magnitudo dampak (warna merah tinggi, warna biru rendah). Sangat kuat untuk memvisualisasikan korelasi arah variabel terhadap keputusan akhir.
* **Data Leakage (Kebocoran Data)**: Kesalahan metodologi terfatal dalam AI, yaitu ketika algoritma tanpa sengaja mempelajari informasi dari set tes (data masa depan/kunci jawaban) selama fase pelatihan. Dalam penelitian ini, dicegah melalui eksekusi SMOTE secara murni di dalam *inner CV fold*.
* **False Negative (FN) Error Analysis**: Dalam eksperimen kita, ini adalah kelompok mahasiswa yang oleh komputer diprediksi "Akan Gagal", namun kenyataannya mereka "Berhasil Lulus" (*Late-bloomers* atau telat beradaptasi).
* **Hold-Out Illusion**: Terjadi ketika hasil evaluasi pada satu set tes tertentu sangat bagus, seolah-olah model tersebut sempurna. Namun ketika diuji secara komprehensif, performanya runtuh.
* **Feature Engineering (Rekayasa Fitur)**: Proses mendaur ulang data mentah mingguan (M1-M8) menjadi agregat bermakna, seperti mencari Nilai Maksimal, Rata-Rata Awal, hingga Tren Deviasi Standar, guna menyuapi model algoritma secara lebih komprehensif.
