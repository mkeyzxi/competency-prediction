Yang sudah bisa dipertahankan

Target:
Final Individu >= 75 → Kompeten, < 75 → Belum Kompeten.

Model:
Decision Tree sebagai baseline, Random Forest sebagai pembanding utama.

Skenario:
S1 Basic → S2 Behavioral → S3 Relational.

Hasil sementara:
RF S2/S3 dengan Test F1 0,838 adalah kandidat terbaik, karena S2 meningkatkan S1 dari 0,812 → 0,838, sedangkan S3 tidak menambah peningkatan.

Dataset:
120 mahasiswa eligible setelah aturan absensi diterapkan.

Ini sudah konsisten dengan arah PRD-mu yang memang menempatkan feature engineering, perbandingan DT/RF, dan TreeSHAP sebagai inti penelitian.

Tetapi saya akan mengubah 3 hal sebelum kamu menjalankan eksperimen final

1. Pisahkan aturan "eligible" dan "early warning"

Aturanmu sekarang:

absence >= 4 → tidak memenuhi syarat melanjutkan/lolos praktikum.
absence > 7 → early exit.

Itu sebaiknya dipakai untuk menentukan populasi analisis, bukan sebagai fitur model.

Jangan sampai:

Attendance_Rate
Absence_Count
Early_Exit_Flag

digunakan sebagai predictor apabila Early_Exit_Flag sendiri secara definisi sudah mengetahui mahasiswa berhenti dari praktikum.

Lebih aman:

RAW DATA
↓
Eligibility filtering
↓
120 mahasiswa
↓
feature engineering
↓
model 2. Untuk early warning, jangan masukkan sesi final/UAS ke fitur

Ini yang paling penting.

BDE:

Pertemuan 9 = nilai flowchart
Pertemuan 10 = nilai kodingan

Kalau dua sesi itu merupakan pelaksanaan Final/UAS, maka keduanya memang valid untuk menentukan riwayat kehadiran, tetapi tidak cocok sebagai fitur early-warning sebelum final.

Jadi saya akan membuat dua konsep:

Attendance_All
= seluruh 10 sesi

dan

Attendance_PreFinal
= hanya sesi yang tersedia sebelum Final

Untuk model early warning, gunakan:

Attendance_PreFinal

Bukan Attendance_All.

Kalau tujuan penelitianmu ternyata bukan prediksi dini sebelum UAS, melainkan prediksi berdasarkan seluruh aktivitas praktikum sampai akhir, maka Attendance_All bisa digunakan. Tetapi jangan mencampur dua klaim tersebut.

3. Hentikan penggunaan "structural zero" sebagai solusi utama

Ini yang menurut saya perlu diubah dari implementasi sekarang.

Sekarang:

AC
TP_Mean ≠ 0
Respons_Mean ≠ 0
TP_Respons_Mean = 0

BDE
TP_Mean = 0
Respons_Mean = 0
TP_Respons_Mean ≠ 0

Saya tidak menyarankan ini untuk versi final.

Karena 0 sudah punya makna akademik:

mahasiswa tidak mengerjakan.

Sedangkan di sini 0 juga berarti:

fitur tidak berlaku untuk skema tersebut.

Ini membuat dua makna berbeda menjadi satu nilai.

Lebih baik:

AC:
TP_Mean = nilai
Respons_Mean = nilai
TP_Respons_Mean = NaN

BDE:
TP_Mean = NaN
Respons_Mean = NaN
TP_Respons_Mean = nilai

Kemudian tambahkan:

Assessment_Group = AC / BDE

sebagai metadata.

Kalau model membutuhkan imputasi, lakukan imputasi di dalam pipeline CV, bukan sebelum split.

Jadi eksperimen final yang saya rekomendasikan

Saya akan gunakan desain ini:

DATA MENTAH
↓
Validasi NIM + duplicate check
↓
Gabungkan sumber data berdasarkan NIM
↓
Tentukan kelas:
AC / BDE
↓
Rekonstruksi attendance
↓
Eligibility filtering
├── absence >= 4 → exclude
└── absence > 7 → tandai Early Exit
↓
Bentuk target:
Final >= 75
↓
Hapus seluruh informasi final/post-final
↓
Feature engineering
├── S1 Basic
├── S2 Behavioral
└── S3 Relational
↓
80:20 stratified split
↓
5-fold CV di training
↓
DT vs RF
↓
Test set
↓
TreeSHAP
↓
Analisis:
Overall
AC
BDE
Dan hasilmu nanti sebaiknya dilaporkan seperti ini

Jangan mengatakan:

"Random Forest S2 sangat efektif."

Gunakan:

"Pada dataset 120 mahasiswa yang memenuhi kriteria analisis, Random Forest pada skenario S2 menghasilkan Test F1-Score sebesar 0,838, lebih tinggi dibandingkan S1 sebesar 0,812. Penambahan fitur behavioral meningkatkan performa model, sedangkan penambahan fitur relational pada S3 tidak memberikan peningkatan tambahan."

Kemudian:

"Analisis berdasarkan kelompok skema penilaian menunjukkan perbedaan performa antara AC dan BDE. Perbedaan tersebut diperlakukan sebagai indikasi heterogenitas konteks penilaian, bukan sebagai bukti bahwa skema penilaian atau kelompok asisten menyebabkan perbedaan kompetensi."

Itu jauh lebih aman untuk skripsi.

Kesimpulan praktis

Jadi tidak perlu membuang penelitianmu dan tidak perlu mengganti Random Forest.

Yang perlu dilakukan sekarang adalah:

Perbaiki preprocessing → jalankan ulang 6 eksperimen → cek CV per fold → evaluasi AC/BDE → baru pilih model final → jalankan TreeSHAP pada model yang benar-benar terpilih.

Dan saya justru menyarankan jangan mengubah S1/S2/S3 lagi sebelum audit preprocessing selesai. Masalah terbesar sekarang ada di definisi fitur dan temporal leakage, bukan pada algoritmanya.
