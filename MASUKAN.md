1. Pertama, nilai 0 belum tentu "merusak" model

Dalam PRD-mu, sudah ditetapkan bahwa:

nilai 0 berarti mahasiswa tidak melaksanakan aktivitas dan harus dipertahankan sebagai 0, bukan diubah menjadi missing atau dihapus.

Jadi kalau mahasiswa memang tidak mengumpulkan laporan dan nilainya memang 0, secara metodologis 0 adalah data yang valid.

Bahkan secara intuitif:

Tidak mengerjakan laporan → 0 → kemungkinan kompetensi rendah

itu justru merupakan pola yang mungkin ingin ditemukan model.

Masalah yang lebih penting adalah berapa banyak mahasiswa dengan 0, bagaimana distribusinya, dan apakah 0 tersebut berasal dari mekanisme yang sama.

2. Dari hasilmu, masalah utamanya bukan sekadar "ada nilai 0"

Hasil yang kamu dapat:

Skenario Model Test Accuracy Test F1
S1 Decision Tree 0.56 0.557
S1 Random Forest 0.60 0.599
S2 Decision Tree 0.60 0.599
S2 Random Forest 0.60 0.594
S3 Decision Tree 0.60 0.599
S3 Random Forest 0.52 0.519

Ini menarik karena:

S2 dan S3 tidak otomatis memperbaiki model.

Bahkan S3 + Random Forest turun menjadi 0.52.

Itu justru temuan penelitian yang sah, karena PRD-mu sendiri mengatakan jangan mengasumsikan feature engineering pasti meningkatkan akurasi.

Jadi jangan memaksakan dataset supaya angka naik.

3. Yang lebih saya khawatirkan: definisi populasi penelitian

PRD-mu sebenarnya sudah punya mekanisme yang jauh lebih tepat daripada mengutak-atik nilai.

Ada:

Early_Exit_Flag ketika absensi > 1
Attendance_Ineligible_Flag ketika absensi > 3
dataset eligible
dataset excluded

dan PRD secara eksplisit menyarankan agar eksklusi dilaporkan, bukan dihapus secara diam-diam.

Ini menurut saya titik yang perlu kamu eksplorasi.

Misalnya dari 158 mahasiswa:

158 raw students
↓
missing/duplicate validation
↓
eligible students
↓
model

Jangan:

158 mahasiswa
↓
"yang bikin model jelek dibuang"
↓
akurasi naik

Tetapi:

158 mahasiswa
↓
berdasarkan aturan akademik
↓
30 early-exit
20 attendance-ineligible
108 eligible
↓
model pada 108 eligible

dan 30 + 20 tadi tetap dilaporkan.

Itu jauh lebih defensible secara ilmiah.

4. Untuk kasus "tidak hadir / tidak mengumpulkan / Final individu tidak ada"

Ini harus dipisahkan satu per satu.

A. Tidak mengumpulkan laporan

Kalau sistem akademik memang memberi:

tidak mengumpulkan → 0

maka:

pertahankan 0.

Jangan ubah menjadi:

0 → 20
0 → 40
0 → median
0 → missing

karena itu mengubah observasi asli.

B. Tidak hadir

PRD sudah menentukan bahwa kehadiran direkonstruksi menjadi:

1
0.5
0

dengan 0 = tidak hadir dan 0.5 = hadir parsial.

Jadi lagi-lagi:

0 jangan diganti.

Yang boleh kamu lakukan adalah menggunakan informasi tersebut untuk membuat:

Attendance_Rate
absence_count
partial_count
Early_Exit_Flag
Attendance_Ineligible_Flag

Itu memang sesuai desain penelitianmu.

C. Final Individu tidak ada

Nah, ini berbeda.

Karena Final Individu adalah target:

Final >= 75 → Kompeten
Final < 75 → Belum Kompeten

maka kalau Final benar-benar missing, target tidak bisa dibentuk. PRD secara eksplisit menyatakan data tersebut harus diperbaiki dari sumber resmi atau dikeluarkan dari modelling.

Jadi jangan melakukan:

Final missing → 0

kecuali sumber akademik memang mengatakan mahasiswa tersebut memperoleh nilai 0.

Kalau tidak diketahui:

Final missing → exclude
reason = MISSING_FINAL

Itu benar secara metodologi.

5. Apakah aman "rekayasa sedikit"?

Kalau yang dimaksud:

NIM dan Nama tetap asli, tetapi nilai kehadiran/laporan saya ubah sedikit supaya model lebih bagus

Tidak saya rekomendasikan.

Itu bukan preprocessing lagi. Itu sudah menjadi modifikasi observasi penelitian.

Untuk publikasi, pertanyaan yang kemungkinan muncul:

"Apakah data merupakan data observasi asli?"

Kalau jawabannya ternyata nilai telah diubah agar performa meningkat, itu bisa menjadi masalah serius.

6. Tapi ada satu jenis "rekayasa" yang aman

Ada perbedaan besar antara:

Manipulasi data asli ❌
Laporan asli = 0
↓
diubah menjadi 30

dengan:

Transformasi data yang terdokumentasi ✅

misalnya:

Laporan_1 = 0
Laporan_2 = 80
Laporan_3 = 90

→ Laporan_Mean = 56.67

atau:

Attendance:
1,1,0,1,0.5...

→ Attendance_Rate

Ini bukan manipulasi. Itu feature engineering, dan memang sudah menjadi bagian metodologi PRD-mu.

7. Bahkan kamu boleh membuat dataset anonim

Ini justru sangat saya sarankan kalau datanya mahasiswa.

Misalnya:

NIM Nama
220001 Ahmad
220002 Budi

di dataset penelitian bisa dibuat:

STU_001
STU_002

dan:

Nama → dihapus
NIM → pseudonymized

sementara mapping asli disimpan terpisah dan tidak masuk ke data modelling.

PRD sendiri melarang NIM dan Nama menjadi fitur model.

Jadi kamu bisa menjaga privasi tanpa mengubah nilai akademik.

8. Saya justru menyarankan eksperimen tambahan

Daripada "memperbaiki" data agar accuracy naik, buat penelitianmu lebih kuat dengan sensitivity analysis.

Misalnya:

Eksperimen A

Raw-valid data

Semua nilai 0 yang sah dipertahankan.

Eksperimen B

Eligible population

Keluarkan mahasiswa yang:

Early_Exit_Flag = 1

berdasarkan aturan PRD.

Eksperimen C

Strict eligible

Keluarkan:

Attendance_Ineligible_Flag = 1
Eksperimen D

Bandingkan model:

Original 0
vs
excluded ineligible

Bukan mengubah 0 menjadi nilai lain.

Kemudian laporkan:

## Dataset N Accuracy F1

All valid students 158 0.xx 0.xx
Eligible students xxx 0.xx 0.xx
Strict eligible xxx 0.xx 0.xx

Kalau akurasinya naik setelah exclusion berdasarkan aturan akademik, kamu punya alasan metodologis, bukan karena mengutak-atik angka.

PRD memang mengharuskan populasi eligible/excluded dan alasan eksklusi dicatat.

9. Ada masalah lain yang menurut saya lebih penting daripada 0

Dataset-mu hanya 122 observasi yang masuk ke feature dataset, sedangkan data awal 158.

Itu berarti ada sekitar:

36 mahasiswa yang sudah tidak masuk ke featured dataset.

Nah, ini justru perlu kamu bedah.

Pertanyaan pentingnya:

158
↓
berapa missing?
↓
berapa early exit?
↓
berapa attendance ineligible?
↓
berapa missing final?
↓
berapa duplicate?
↓
berapa eligible?
↓
122?

Jangan langsung fokus:

"kenapa accuracy cuma 60%?"

Sebelum itu kita harus tahu:

"122 mahasiswa itu sebenarnya siapa dan 36 mahasiswa yang hilang itu kenapa?"

Karena kalau 36 mahasiswa tersebut keluar karena suatu aturan seleksi, hal itu bisa menghasilkan selection bias. PRD sendiri sudah mengidentifikasi early-exit selection bias sebagai salah satu risiko metodologis.

10. Ada satu hal lagi yang cukup menarik dari S3

S3 memasukkan:

Respons_TP_Gap = Respons_Mean - TP_Mean

Tetapi untuk BDE:

TP = Respons

sehingga:

Respons_TP_Gap = 0

untuk seluruh observasi BDE. PRD sendiri sudah mendokumentasikan konsekuensi ini.

Jadi kalau S3 tidak bagus, itu bukan berarti dataset buruk.

Bisa saja memang fitur tambahan tersebut tidak memberikan informasi baru.

Kesimpulan saya

Jangan mengubah nilai asli mahasiswa untuk menaikkan akurasi.

Yang aman dan justru lebih kuat untuk penelitian:

DATA ASLI
↓
VALIDASI
↓
PERTAHANKAN 0 YANG VALID
↓
REKONSTRUKSI KEHADIRAN
↓
BUAT FLAG EARLY EXIT / INELIGIBLE
↓
EXCLUDE BERDASARKAN ATURAN YANG SUDAH DITETAPKAN
↓
FEATURE ENGINEERING
↓
CV + TEST
↓
SENSITIVITY ANALYSIS

Kalau targetmu SINTA 2, akurasi 60% bukan otomatis berarti penelitianmu gagal. Yang jauh lebih berbahaya adalah mendapatkan 85% dari data yang sudah "dibantu" supaya cocok dengan model.

Dan ada hal yang cukup menarik: PRD-mu sudah cukup matang untuk membangun argumen metodologis bahwa data 0 tidak dibuang, tetapi mahasiswa yang memang tidak memenuhi populasi analisis dikelola melalui eligibility/exclusion yang eksplisit.

Menurut saya, langkah berikut yang paling bernilai adalah mengaudit 158 → 122 baris itu secara rinci. Dari sana kita bisa melihat apakah accuracy rendah memang karena distribusi label/0, karena exclusion, class imbalance, atau justru ada masalah pada feature engineering dan split.
