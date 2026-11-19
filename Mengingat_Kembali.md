# 🧠 Mengingat Kembali: Logika Proyek EWS Praktikum

Dokumen ini berisi rangkuman diskusi untuk menyegarkan ingatan mengenai komponen-komponen kunci dalam proyek prediksi kelulusan praktikum (_Early Warning System_ / EWS).

---

## 1. Apa itu S1 sampai S5? (Skenario Fitur)
**S1 hingga S5** adalah skenario tingkat kerumitan fitur yang dimasukkan ke dalam model *Machine Learning*. Semakin tinggi angkanya, semakin banyak detail yang diberikan ke model.
*   **S1 (Paling Dasar):** Hanya melihat rata-rata kehadiran, rata-rata nilai Tugas Pendahuluan (TP), dan rata-rata Laporan.
*   **S2:** S1 + jumlah bolos dan persentase kelengkapan pengumpulan tugas.
*   **S3:** S2 + **Fluktuasi Nilai** (`Performance_Std`). Model mulai melihat apakah nilai mahasiswa stabil atau naik-turun.
*   **S4 & S5:** Paling rumit, mencakup fitur seperti rata-rata 2 tugas pertama, 2 tugas terakhir, tren nilai (menanjak/menurun), nilai minimum, maksimum, dll.

Dari hasil baseline, **S3** terbukti sebagai kombinasi yang paling pas dan seimbang.

---

## 2. Isi dari S3 (Skenario 3 Dasar)
Skenario S3 murni belum mengandung fitur kecerdasan EWS. Ia hanya berisi 6 fitur rekapitulasi standar:
1.  `Attendance_PreFinal_Rate` (Persentase Kehadiran)
2.  `TP_Mean` (Rata-rata nilai Tugas Pendahuluan)
3.  `Laporan_Mean` (Rata-rata nilai Laporan)
4.  `TP_Completion_Rate` (Persentase TP yang dikumpulkan)
5.  `Laporan_Completion_Rate` (Persentase Laporan yang dikumpulkan)
6.  `Performance_Std` (Standar Deviasi / kestabilan nilai)

---

## 3. Apa itu EWS (S3_A sampai S3_EWS)?
Karena S3 adalah yang terbaik, pengujian dilanjutkan dengan menambahkan 1 "Fitur Cerdas" (EWS) ke dalam rumusan S3 untuk mendeteksi tanda bahaya seawal mungkin.
*   **S3_A:** S3 + `Absence_Streak_Max` (Rekor bolos berturut-turut terpanjang).
*   **S3_B:** S3 + `Performance_Decline_Flag` (Deteksi apakah nilai awal bagus lalu akhir-akhir mendadak anjlok).
*   **S3_C:** S3 + `Zero_Score_Count` (Menghitung seberapa sering mendapat nilai 0 mutlak).
*   **S3_D:** S3 + `Cumulative_Completion_Trend` (Perbandingan rajin ngumpul tugas di paruh awal vs paruh akhir).
*   **S3_E (Pemenang Final):** S3 + `Early_Performance_Composite` (Hanya melihat rata-rata **2 tugas & laporan paling awal**, mengabaikan sisanya).
*   **S3_EWS:** S3 digabung dengan semua fitur cerdas di atas sekaligus.

---

## 4. Jendela Waktu (Temporal Cutoffs): W1, W2, W3, dan C_Full
Jendela waktu digunakan untuk mencegah "Kebocoran Masa Depan" dan menguji seberapa dini model bisa memprediksi.
*   **`W1` (Week 1):** Model hanya boleh melihat data dari 1 pertemuan pertama.
*   **`W2` (Week 2):** Model hanya boleh melihat data dari 2 pertemuan pertama.
*   **`W3` (Week 3):** Model hanya boleh melihat data dari 3 pertemuan pertama.
*   **`C_Full` (Full Semester):** Model melihat **seluruh data satu semester penuh**.

---

## 5. Kapan Pengujian W1, W2, W3 dilakukan?
Pengujian di berbagai jendela waktu (W1, W2, W3, C_Full) **tidak dilakukan satu per satu secara manual**. Semuanya berjalan secara otomatis (di-_loop_) ketika Anda menjalankan skrip:
```bash
python scripts/run_optimized_experiment.py
```
Skrip ini akan mengambil pemenang EWS (Decision Tree - S3_E), lalu memotong datanya di W1 dan mengujinya, kemudian lanjut memotong di W2 lalu mengujinya, dan seterusnya. Hasilnya dibandingkan untuk membuktikan bahwa di W2/W3 saja, sistem sudah sangat akurat.

---

## 6. Kenapa `generate_shap_s3_dasar.py` tidak menggunakan W1, W2, W3?
Saat skrip `generate_shap_s3_dasar.py` (atau `run_baseline_experiments.py`) dijalankan, W1, W2, dan W3 **sama sekali belum digunakan**. 

Skrip S3 Dasar memuat data `data/features/C_Full_S3.csv`. Artinya, model "Baseline" ini bermain dengan aturan retrospektif (menggunakan data seluruh semester dari awal sampai akhir). Karena ia menggunakan data seluruh semester (`C_Full`), SHAP akan menunjukkan bahwa ia sangat bergantung pada nilai akhir, yang membuatnya **kurang ideal sebagai Peringatan Dini**. 

Kelemahan inilah yang kemudian diselesaikan di langkah optimasi EWS (skrip `run_optimized_experiment.py`) yang menerapkan batasan W1, W2, dan W3.
