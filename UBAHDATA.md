# UBAHDATA.md
# Migrasi Penelitian Early Warning Kompetensi Praktikum ke Dataset Basis Data Non Relasional

> Dokumen ini adalah spesifikasi migrasi penuh penelitian **Sistem Peringatan Dini Kompetensi Mahasiswa Praktikum** dari dataset Praktikum Logika & Algoritma ke dataset mata kuliah **Basis Data Non Relasional** dengan total **89 mahasiswa dari 5 kelas (A, B, C, D, E)**.
>
> Struktur metodologis penelitian lama dipertahankan sejauh masih sesuai: **target kompetensi → temporal cutoff → feature engineering bertingkat → preprocessing CV-aware → Decision Tree/Random Forest → evaluasi → TreeSHAP → error analysis**.
>
> Perubahan utama penelitian baru adalah:
>
> 1. jumlah mahasiswa berubah dari 123 menjadi 89;
> 2. struktur aktivitas per kelas tidak sepenuhnya identik;
> 3. terdapat nilai E;
> 4. data saat ini sudah bersih sehingga fokus berpindah dari pembersihan data ke audit konsistensi dan pembentukan dataset analitik;
> 5. **Logistic Regression dikeluarkan dari rancangan eksperimen**;
> 6. fitur `Respons_*` pada penelitian lama hanya dipakai apabila memang tersedia pada data baru; jika tidak tersedia, jangan dibuat secara artifisial;
> 7. data pada sheet `FINAL` diperlakukan sebagai sumber **ground truth/post-final**, bukan sebagai predictor early warning.

---

# 1. Tujuan Migrasi

Penelitian baru tetap menjawab pertanyaan inti:

> **Seberapa dini pola aktivitas mahasiswa selama proses pembelajaran/praktikum dapat digunakan untuk memprediksi status kompetensi sebelum evaluasi final, dan fitur apa yang paling berkontribusi terhadap prediksi tersebut?**

Fokus penelitian **bukan** sekadar mencari model dengan Accuracy tertinggi.

Fokus utama adalah:

- apakah sinyal risiko sudah muncul sejak fase awal;
- apakah informasi kehadiran, laporan, dan TP cukup informatif;
- apakah pola nilai awal, nilai terbaru, konsistensi, volatilitas, dan trend membantu prediksi;
- apakah model mampu mendeteksi mahasiswa **Belum Kompeten**;
- kapan cutoff temporal memberikan trade-off terbaik antara **kedinian warning** dan **kestabilan prediksi**;
- fitur apa yang berkontribusi terhadap output model berdasarkan TreeSHAP;
- kesalahan apa yang masih terjadi pada mahasiswa tertentu.

Bahasa hasil penelitian harus tetap bersifat **prediktif**, bukan kausal.

Gunakan:

- "berkontribusi terhadap prediksi";
- "mendorong output model";
- "berasosiasi dengan keputusan model";
- "menjadi signal prediktif".

Hindari:

- "menyebabkan mahasiswa tidak kompeten";
- "kehadiran menyebabkan kelulusan";
- "fitur X menyebabkan kompetensi".

---

# 2. Perubahan Dataset

## 2.1 Populasi

Dataset baru:

| Komponen | Dataset lama | Dataset baru |
|---|---:|---:|
| Mata kuliah | Praktikum Logika & Algoritma | Basis Data Non Relasional |
| Jumlah mahasiswa | 123 P2 Strict Eligible | **89** |
| Jumlah kelas | 5 kelas | **5 kelas A-E** |
| Kondisi data | melalui eligibility/quality checks | **sudah bersih** |
| Model utama | Random Forest | **Random Forest** |
| Model pembanding | Dummy, Logistic Regression, Decision Tree | **Dummy, Decision Tree, Random Forest** |
| XAI | TreeSHAP | **TreeSHAP** |
| Validasi | Nested CV + hold-out | **tetap dipertahankan** |
| Temporal cutoff | C1-C4/C_Full | **C1-C4/C_Full atau cutoff adaptif berdasarkan aktivitas aktual** |

Jumlah 89 adalah **small-sample dataset**. Konsekuensinya, setiap observasi memiliki pengaruh relatif besar terhadap metrik. Oleh sebab itu, validasi berulang dan pelaporan variasi performa harus diberi bobot besar.

---

# 3. Struktur Sumber Data

Dataset baru terdiri dari:

- Sheet `A`
- Sheet `B`
- Sheet `C`
- Sheet `D`
- Sheet `E`
- Sheet `FINAL`

## 3.1 Sheet kelas A-E

Setiap sheet kelas berisi secara umum:

### Identitas

- `NIM`
- `NAMA`

Identifier hanya digunakan untuk:

- pencocokan;
- audit;
- error analysis individual;
- pelacakan sumber data.

Identifier **tidak boleh menjadi predictor**.

### Kehadiran

Terdiri atas:

- `Pertemuan 1`
- `Pertemuan 2`
- ...
- `Total`
- `TOTAL NILAI HADIR`

Makna:

- `1` = hadir;
- `0` = tidak hadir.

### Laporan

Terdiri atas:

- `Laporan 1`
- `Laporan 2`
- ...
- `TOTAL NILAI LAPORAN`

Nilai `0` yang memang berarti tidak mengerjakan harus dipertahankan sebagai **0**.

### TP

Terdiri atas:

- `TP 1`
- `TP 2`
- ...
- `TOTAL NILAI TP`

Nilai `0` yang memang berarti tidak mengerjakan harus dipertahankan sebagai **0**.

### Final

Sheet kelas biasanya memiliki:

- `FINAL`
- `TOTAL NILAI`
- `BOBOT`

Kolom ini harus diperlakukan hati-hati karena berada pada atau setelah tahap evaluasi akhir.

---

# 4. Aturan Penting untuk Nilai 0, Nilai E, dan Aktivitas Tidak Tersedia

## 4.1 Nilai 0

Pada dataset baru, nilai `0` harus dipertahankan apabila memang memiliki makna akademik:

- tidak hadir;
- tidak mengumpulkan TP;
- tidak mengumpulkan laporan;
- aktivitas dikerjakan tetapi nilainya benar-benar 0.

Jangan mengubah 0 menjadi NaN.

Alasannya:

> `0` = mahasiswa gagal/tidak melakukan aktivitas yang tersedia.

Sedangkan:

> `NaN` = aktivitas tersebut memang belum tersedia pada cutoff atau tidak ada karena perbedaan struktur pembelajaran.

Kedua kondisi tersebut secara akademik berbeda.

---

## 4.2 Nilai E

Nilai `E` pada dataset baru adalah **hasil/label kategorikal akhir**, bukan otomatis predictor.

Nilai E harus diaudit untuk mengetahui:

1. apakah E berasal dari `TOTAL NILAI`;
2. apakah E identik dengan kondisi `Belum Kompeten`;
3. apakah terdapat E walaupun sebagian aktivitas masih bernilai tinggi;
4. apakah batas konversi huruf konsisten untuk seluruh kelas.

Jangan membuat aturan:

> `E = Belum Kompeten`

sebelum hubungan E dengan nilai numerik final diverifikasi.

Untuk machine learning, target utama tetap harus dibuat dari **nilai numerik ground truth yang didefinisikan secara eksplisit**.

---

# 5. Bagaimana Memperlakukan Sheet FINAL?

Ini merupakan bagian paling penting dalam migrasi.

Sheet `FINAL` berisi:

| Kolom | Makna |
|---|---|
| Nama | identitas |
| NIM | identitas |
| Kelas | kelompok |
| CRUD & DB | komponen penilaian final |
| Fitur dan Inovasi | komponen penilaian final |
| Kerapihan | komponen penilaian final |
| Individu | komponen penilaian individual |
| Nilai Final Total | agregat penilaian final |

## 5.1 Jangan memasukkan seluruh sheet FINAL sebagai predictor

Untuk penelitian **early warning**, nilai pada sheet `FINAL` merupakan informasi yang tersedia pada tahap evaluasi final.

Jika model dimaksudkan memberi warning **sebelum final**, maka:

```text
CRUD & DB            -> JANGAN masuk X
Fitur dan Inovasi    -> JANGAN masuk X
Kerapihan            -> JANGAN masuk X
Individu             -> JANGAN masuk X
Nilai Final Total    -> JANGAN masuk X
BOBOT/HURUF FINAL    -> JANGAN masuk X
```

Semua komponen tersebut dapat digunakan untuk:

- membentuk target;
- audit ground truth;
- analisis deskriptif final;
- error analysis;

tetapi tidak sebagai predictor awal.

Jika komponen final dimasukkan ke X, penelitian berubah menjadi **prediksi dari nilai final menggunakan nilai final**, sehingga terjadi target leakage.

---

# 6. Keputusan Ground Truth / Target

Ini harus dibekukan sebelum modeling.

Penelitian lama menggunakan prinsip:

```text
Final_Individu >= 75  -> Kompeten
Final_Individu < 75   -> Belum Kompeten
```

Migrasi yang paling konsisten dengan penelitian lama adalah mempertahankan **Individu** sebagai target utama apabila kolom tersebut memang merupakan penilaian pemahaman individual mahasiswa.

## 6.1 Target utama yang direkomendasikan

```text
Competency_Label =
    1 jika Individu >= 75
    0 jika Individu < 75
```

Dengan:

- `1` = Kompeten
- `0` = Belum Kompeten

Ini mempertahankan semangat ground truth penelitian sebelumnya: kompetensi individual ditentukan dari penilaian akhir individual.

---

## 6.2 Bagaimana dengan Nilai Final Total?

`Nilai Final Total` tetap diambil dan disimpan, tetapi **bukan predictor**.

Gunakan sebagai:

- variabel audit;
- deskripsi distribusi nilai akhir;
- pemeriksaan konsistensi;
- analisis sensitivitas target;
- pembandingan dengan target individual.

Buat dua kolom terpisah:

```text
Final_Individu
Final_Total
Competency_Label
```

Jangan mengganti target secara diam-diam.

---

## 6.3 Sensitivity Analysis target

Karena dataset baru memiliki struktur final yang lebih jelas, penelitian dapat menambahkan analisis sensitivitas:

### Target Utama

```text
Competency_Label_Individual
= 1 jika Individu >= 75
= 0 jika Individu < 75
```

### Target Sensitivitas

```text
Competency_Label_Total
= 1 jika Nilai Final Total >= 75
= 0 jika Nilai Final Total < 75
```

Target sensitivitas **tidak digunakan untuk mengganti target utama**.

Tujuannya hanya menjawab:

> Apakah kesimpulan model berubah jika kompetensi didefinisikan berdasarkan nilai final keseluruhan, bukan komponen penilaian individu?

Jika kedua target menghasilkan pola yang serupa, robustness penelitian menjadi lebih kuat.

---

# 7. Dataset Analitik Akhir

Dataset modeling sebaiknya memiliki satu baris per mahasiswa.

Contoh struktur:

```text
NIM
Nama
Kelas

Attendance_P1
Attendance_P2
...
Attendance_Pk

Laporan_1
Laporan_2
...
Laporan_m

TP_1
TP_2
...
TP_n

TP_Mean
Laporan_Mean
Attendance_PreFinal_Rate

TP_Completion_Rate
Laporan_Completion_Rate

TP_Std
Laporan_Std
Performance_Std

TP_First2_Mean
Laporan_First2_Mean

TP_Last2_Mean
Laporan_Last2_Mean

TP_Trend
Laporan_Trend

Performance_Late_Mean
Laporan_Max
Absence_Count

Final_Individu
Final_Total
Competency_Label
```

Perlu dibedakan antara:

### Raw activity features

Nilai per pertemuan/aktivitas.

### Engineered features

Mean, std, completion, trend, first2, last2, dll.

### Ground truth

`Final_Individu`, `Final_Total`, `Competency_Label`.

---

# 8. Feature Engineering Baru

Karena data baru tidak menyebut adanya fitur `Respons`, jangan memaksakan fitur tersebut.

Feature engineering disusun berdasarkan aktivitas yang benar-benar tersedia.

## 8.1 Feature family 1 — Attendance

### `Attendance_PreFinal_Rate`

```text
jumlah hadir / jumlah pertemuan yang tersedia
```

Nilai 0 tetap valid.

### `Absence_Count`

```text
jumlah pertemuan dengan nilai 0
```

### `Attendance_First2_Rate`

Proporsi kehadiran pada dua pertemuan awal.

### `Attendance_Last2_Rate`

Proporsi kehadiran pada dua pertemuan terakhir yang masuk cutoff.

---

# 9. Feature Family Laporan

## 9.1 Level

### `Laporan_Mean`

Rata-rata nilai laporan pada cutoff.

### `Laporan_Max`

Nilai laporan tertinggi.

### `Laporan_Min`

Nilai laporan terendah.

### `Laporan_Median`

Median nilai laporan bila jumlah observasi mencukupi.

---

## 9.2 Completion

### `Laporan_Completion_Rate`

Definisi:

```text
jumlah laporan yang dikerjakan
/
jumlah laporan yang seharusnya tersedia pada cutoff
```

Jangan menghitung aktivitas yang belum tersedia sebagai gagal.

---

## 9.3 Variability

### `Laporan_Std`

Standar deviasi nilai laporan.

Interpretasi:

- kecil = relatif konsisten;
- besar = performa tidak stabil.

---

## 9.4 Temporal Features

### `Laporan_First2_Mean`

Rata-rata dua laporan awal.

### `Laporan_Last2_Mean`

Rata-rata dua laporan terbaru.

### `Laporan_Trend`

Arah perubahan performa laporan.

Implementasi yang direkomendasikan:

```python
numpy.polyfit(time_index, score, 1)[0]
```

Interpretasi:

- positif = kecenderungan meningkat;
- negatif = kecenderungan menurun;
- mendekati 0 = relatif datar.

### `Laporan_Max`

Tetap dipertahankan karena sudah terbukti berguna dalam eksperimen penelitian lama.

---

# 10. Feature Family TP

Gunakan struktur paralel.

## 10.1 Level

- `TP_Mean`
- `TP_Max`
- `TP_Min`
- `TP_Median`

## 10.2 Completion

`TP_Completion_Rate`

```text
jumlah TP dikerjakan / jumlah TP tersedia pada cutoff
```

## 10.3 Variability

`TP_Std`

## 10.4 Early Performance

`TP_First2_Mean`

## 10.5 Recent Performance

`TP_Last2_Mean`

## 10.6 Trend

`TP_Trend`

---

# 11. Combined Performance Features

Gabungkan sumber performa yang benar-benar tersedia.

## 11.1 `Performance_Mean`

Rata-rata performa aktivitas akademik yang diizinkan pada cutoff.

Contoh:

```text
Performance_Mean =
mean(TP values + Laporan values)
```

Jangan memasukkan final.

## 11.2 `Performance_Std`

Variabilitas keseluruhan aktivitas.

## 11.3 `Performance_First2_Mean`

Signal performa awal.

## 11.4 `Performance_Late_Mean`

Rata-rata performa pada fase akhir cutoff.

## 11.5 `Performance_Trend`

Trend gabungan jika jumlah titik waktu memungkinkan.

---

# 12. Perlakuan Perbedaan Jumlah Aktivitas Antar Kelas

Jika kelas A-E memiliki jumlah laporan/TP/pertemuan yang berbeda, **jangan memaksa semua kolom kosong menjadi 0**.

Contoh:

```text
Kelas A: TP1 TP2 TP3
Kelas E: TP1 TP2 TP3 TP4
```

Maka:

```text
TP4 pada A = NaN
TP4 pada E = nilai sebenarnya
```

Bukan:

```text
TP4 pada A = 0
```

kecuali memang mahasiswa kelas A memiliki kewajiban TP4 tetapi tidak mengerjakannya.

Ini merupakan implementasi langsung dari prinsip **common temporal window** pada penelitian lama.

---

# 13. Common Temporal Window

Perbedaan struktur antar kelas merupakan bagian penting dari migrasi.

Tujuannya adalah:

> membandingkan informasi yang memang tersedia pada waktu yang setara.

Gunakan cutoff:

```text
C1 = sangat awal
C2 = awal-menengah
C3 = menengah
C4 = lebih lanjut
C_Full = seluruh aktivitas pra-final
```

Tetapi cutoff harus ditentukan berdasarkan **aktivitas yang benar-benar tersedia**, bukan berdasarkan jumlah kolom semata.

---

# 14. Cara Menentukan Cutoff

Misalnya aktivitas efektif adalah:

```text
Pertemuan 1-6
TP 1-4
Laporan 1-4
```

Maka dapat dibuat:

### C1

Hanya aktivitas fase awal:

```text
Pertemuan awal
TP awal
Laporan awal
```

### C2

Menambahkan aktivitas berikutnya.

### C3

Memasukkan sebagian besar aktivitas tengah.

### C4

Memasukkan aktivitas yang lebih mendekati final.

### C_Full

Seluruh aktivitas sebelum penilaian final.

Tidak boleh menggunakan:

- CRUD & DB final;
- Fitur & Inovasi final;
- Kerapihan final;
- Individu final;
- Nilai Final Total.

---

# 15. Rekomendasi Pembentukan Cutoff

Karena jumlah mahasiswa hanya 89 dan kelas memiliki struktur yang mungkin tidak identik, jangan membuat terlalu banyak cutoff.

Rekomendasi:

```text
C1
C2
C3
C_Full
```

`C4` boleh dibuat jika memang ada cukup aktivitas berbeda secara substantif.

Alasannya adalah setiap kombinasi:

```text
cutoff × feature set × model
```

menambah jumlah eksperimen.

Pada small sample, terlalu banyak konfigurasi meningkatkan risiko cherry-picking.

---

# 16. Feature Set Bertingkat

Migrasi S1-S6 tidak perlu dipertahankan secara identik.

Gunakan:

## S1 — Basic

Fitur dasar:

```text
TP_Mean
Laporan_Mean
Attendance_PreFinal_Rate
```

Tujuan:

> Apakah agregat sederhana sudah memiliki signal?

---

## S2 — Completion

S1 +

```text
TP_Completion_Rate
Laporan_Completion_Rate
Absence_Count
```

Tujuan:

> Apakah konsistensi partisipasi menambah informasi?

---

## S3 — Variability

S2 +

```text
TP_Std
Laporan_Std
Performance_Std
```

Tujuan:

> Apakah kestabilan/volatilitas performa menambah kemampuan diskriminasi?

---

## S4 — Temporal

S3 +

```text
TP_First2_Mean
Laporan_First2_Mean
TP_Last2_Mean
Laporan_Last2_Mean
TP_Trend
Laporan_Trend
Performance_Late_Mean
```

Tujuan:

> Apakah trajectory mahasiswa memberikan informasi tambahan?

---

## S5 — Combined Statistical

S4 +

```text
TP_Min
TP_Max
Laporan_Min
Laporan_Max
Performance_Mean
Performance_Trend
```

Tujuan:

> Apakah representasi statistik yang lebih lengkap memberikan peningkatan yang bertahan pada validasi?

---

## S6 — Temporal Selected

S6 bukan berarti "semakin banyak fitur".

S6 sebaiknya merupakan:

> feature set temporal yang paling relevan dan dapat dipertanggungjawabkan setelah evaluasi S1-S5.

Contoh kandidat:

```text
TP_First2_Mean
Laporan_First2_Mean
TP_Last2_Mean
Laporan_Last2_Mean
TP_Trend
Laporan_Trend
Performance_Std
Attendance_PreFinal_Rate
Absence_Count
TP_Completion_Rate
Laporan_Completion_Rate
Performance_Late_Mean
```

Jumlah final S6 harus dibekukan berdasarkan kode.

---

# 17. Menghindari Ledakan Dimensi

Pada n=89, jumlah fitur jangan dibiarkan berkembang tanpa kontrol.

Prinsip:

```text
89 mahasiswa
↓
fitur relatif sedikit
↓
trajectory meaningful
↓
validasi ketat
```

Jangan membuat ratusan fitur dari satu kolom.

Hindari transformasi yang tidak memiliki makna akademik.

---

# 18. Model yang Digunakan

Sesuai keputusan migrasi:

## 18.1 Dummy Classifier

Tetap digunakan sebagai baseline.

Tujuan:

> mengetahui apakah model ML benar-benar mengungguli prediksi kelas mayoritas.

---

## 18.2 Decision Tree

Digunakan karena:

- non-linear;
- relatif mudah dijelaskan;
- dapat menjadi pembanding yang transparan;
- kompatibel dengan TreeSHAP.

---

## 18.3 Random Forest

Menjadi model utama karena:

- dapat menangkap relasi non-linear;
- lebih stabil daripada single tree;
- cocok untuk feature interaction;
- kompatibel dengan TreeSHAP.

---

## 18.4 Logistic Regression

**DIHAPUS.**

Tidak perlu dibuat dalam:

- benchmark;
- tabel model;
- eksperimen utama;
- pembahasan hasil.

Dengan demikian model comparison menjadi:

```text
Dummy
Decision Tree
Random Forest
```

---

# 19. Penanganan Class Imbalance

Pertama hitung:

```text
jumlah Kompeten
jumlah Belum Kompeten
proporsi masing-masing kelas
```

Sebelum memutuskan menggunakan SMOTE.

Karena n=89, SMOTE tidak boleh digunakan secara otomatis hanya karena terdapat imbalance.

Jika kelas minoritas terlalu kecil, synthetic oversampling dapat menjadi tidak stabil.

Rekomendasi:

1. ukur distribusi kelas;
2. bandingkan baseline tanpa balancing;
3. jika diperlukan, uji `class_weight="balanced"` terlebih dahulu;
4. SMOTE hanya diuji jika jumlah kasus minoritas memadai;
5. semua balancing harus berada di dalam training fold.

---

# 20. Pipeline CV-Aware

Urutan yang benar:

```text
Raw Dataset
    ↓
Target Separation
    ↓
Train/Test Split atau Outer CV
    ↓
Inner Training
    ↓
Imputation
    ↓
Balancing / SMOTE jika digunakan
    ↓
Feature Selection jika digunakan
    ↓
Hyperparameter Tuning
    ↓
Model Fit
    ↓
Validation
    ↓
Outer Evaluation
    ↓
Final Hold-out
```

Jangan:

```text
Impute seluruh dataset
↓
SMOTE seluruh dataset
↓
Split
```

Karena dapat menyebabkan leakage.

---

# 21. Imputation

Karena data mentah sudah bersih, tidak berarti semua nilai harus dipaksa menjadi angka.

NaN yang muncul akibat **aktivitas belum tersedia** boleh dipertahankan sampai pipeline.

Imputer dipasang di dalam pipeline.

Untuk fitur numerik dapat menggunakan:

```text
median
```

Imputer hanya fit pada training fold.

---

# 22. Validasi

Dengan n=89, satu hold-out dapat sangat sensitif.

Desain utama:

```text
Stratified Repeated K-Fold
```

dan:

```text
Nested Cross-Validation
```

Rekomendasi awal:

```text
Outer: Stratified 5-Fold
Repeats: 3 atau 5
Inner: Stratified 4/5-Fold
```

Namun jumlah fold harus disesuaikan dengan **jumlah minimum kelas minoritas**.

Jika kelas minoritas sangat sedikit, jumlah fold harus diturunkan.

---

# 23. Hold-out Test

Hold-out tetap diperbolehkan sebagai evaluasi akhir.

Namun:

> hold-out tidak boleh digunakan untuk memilih S1/S2/S3/S4/S5, threshold, hyperparameter, atau feature selection.

Prosedur:

```text
Data 89
↓
Lock final test set
↓
Training/validation pada data lainnya
↓
Pilih model berdasarkan CV
↓
Freeze model
↓
Test sekali
```

Jika split test terlalu kecil, jangan mengangkat angka test sebagai bukti utama generalisasi.

---

# 24. Metrik Utama

Metrik tidak boleh hanya Accuracy.

Gunakan:

## 24.1 Accuracy

```text
(TP + TN) / N
```

## 24.2 Balanced Accuracy

```text
(Recall Kompeten + Recall Belum Kompeten) / 2
```

Ini menjadi metrik utama perbandingan.

## 24.3 Recall Belum Kompeten

Sangat penting untuk early warning.

```text
Recall_Belum_Kompeten =
TN / (TN + FP)
```

dengan konvensi:

```text
0 = Belum Kompeten
1 = Kompeten
```

Catatan: implementasi metric dari sklearn harus memastikan `pos_label` dan confusion matrix konsisten.

## 24.4 Precision

Menilai kualitas alarm.

## 24.5 F1 Macro

Menghindari dominasi kelas mayoritas dalam ringkasan.

## 24.6 Confusion Matrix

Wajib disimpan.

## 24.7 Mean ± SD

Untuk repeated CV:

```text
mean metric ± standard deviation
```

---

# 25. Cutoff Temporal sebagai Pertanyaan Utama

Eksperimen temporal harus membandingkan:

```text
C1
C2
C3
C_Full
```

dan untuk setiap cutoff:

```text
S1
S2
S3
S4
S5
```

Tidak semua kombinasi harus diuji jika bukti menunjukkan feature set tertentu sudah cukup.

Tujuan utama bukan:

> "cutoff mana yang paling akurat?"

Tetapi:

> "seberapa dini signal kompetensi muncul dan pada cutoff mana performanya cukup stabil?"

---

# 26. Aturan Memilih Cutoff Warning

Cutoff final tidak dipilih berdasarkan Accuracy test tertinggi.

Gunakan kombinasi:

```text
Balanced Accuracy
Recall Belum Kompeten
CV mean
CV SD
kedinian cutoff
kompleksitas fitur
```

Prinsip keputusan:

> Cutoff yang sedikit lebih awal dengan performa yang hampir sama tetapi stabilitas lebih baik dapat lebih berguna sebagai early warning daripada cutoff paling akhir.

---

# 27. Feature Selection Top-K

Feature selection tetap boleh dilakukan.

Kandidat:

```text
Top-10
Top-15
Top-20
```

Tetapi selector harus fit hanya pada training/inner CV.

Jangan menghitung ranking seluruh dataset lalu melakukan split.

Tujuan feature selection:

- melihat apakah fitur lebih sedikit mempertahankan performa;
- mengurangi redundansi;
- memeriksa stabilitas fitur.

Bukan untuk "memaksa" jumlah fitur tertentu.

---

# 28. TreeSHAP

TreeSHAP digunakan setelah model kandidat final terkunci.

Analisis:

## Global

- mean absolute SHAP;
- ranking fitur;
- beeswarm.

## Directional

- apakah nilai feature tinggi/rendah cenderung mendorong prediksi menuju Kompeten atau Belum Kompeten.

## Local

Pilih beberapa mahasiswa:

- True Positive;
- True Negative;
- False Positive;
- False Negative.

Jangan hanya mengambil satu kasus yang menarik.

---

# 29. Interpretasi Feature Importance

Misalnya:

```text
TP_First2_Mean
```

menjadi fitur tertinggi.

Pernyataan yang diperbolehkan:

> TP_First2_Mean memiliki kontribusi prediktif tinggi terhadap output Random Forest pada dataset penelitian.

Pernyataan yang tidak diperbolehkan:

> Nilai TP awal menyebabkan mahasiswa menjadi kompeten.

---

# 30. Error Analysis

Pada early warning, dua kesalahan penting:

## False Negative

```text
Actual = Belum Kompeten
Prediction = Kompeten
```

Ini paling berbahaya secara operasional karena mahasiswa berisiko tidak terdeteksi.

## False Positive

```text
Actual = Kompeten
Prediction = Belum Kompeten
```

Ini berarti model memberi alarm kepada mahasiswa yang akhirnya kompeten.

---

# 31. Analisis Pola Late-Dropper

Tetap gunakan konsep penelitian lama.

Definisi analitis:

> Mahasiswa yang pada cutoff awal terlihat aman tetapi kemudian memiliki target akhir Belum Kompeten.

Periksa:

```text
First2
Last2
Trend
Std
Attendance
Completion
```

Tujuan:

> mengetahui blind spot model terkait penurunan performa yang terjadi setelah cutoff.

---

# 32. Analisis Late-Bloomer

Definisi analitis:

> Mahasiswa yang pada cutoff awal diprediksi berisiko tetapi target akhirnya Kompeten.

Ini penting untuk melihat:

- false alarm;
- pemulihan performa;
- keterbatasan cutoff terlalu dini.

---

# 33. Penggunaan Sheet Individu

Ini menjawab pertanyaan:

> "Sheet Individu langsung ambil nilai keseluruhan atau baca pola pada sheet FINAL?"

Keputusan metodologis yang disarankan:

### Untuk target

Gunakan:

```text
FINAL.Individu
```

sebagai ground truth utama bila kolom `Individu` memang merupakan penilaian kemampuan/pemahaman individual saat final.

### Untuk predictor

Jangan gunakan:

```text
FINAL.Individu
FINAL.CRUD_DB
FINAL.Fitur_Inovasi
FINAL.Kerapihan
FINAL.Nilai_Final_Total
```

### Untuk audit

Ambil semuanya:

```text
Final_Individu
Final_CRUD_DB
Final_Fitur_Inovasi
Final_Kerapihan
Final_Total
```

Tujuannya agar setiap target dapat diaudit kembali.

---

# 34. Apakah `Nilai Final Total` Perlu Diambil?

**Ya, ambil.**

Tetapi statusnya:

```text
Ground truth audit / sensitivity analysis
```

bukan:

```text
Predictor
```

Dataset akhir sebaiknya menyimpan:

```text
Final_Individu
Final_CRUD_DB
Final_Fitur_Inovasi
Final_Kerapihan
Final_Total
Competency_Label
```

Namun dalam matriks X:

```text
Final_* = excluded
```

---

# 35. Mengapa Tidak Langsung Mengambil Nilai Keseluruhan Sebagai Fitur?

Karena nilai keseluruhan final mengandung informasi masa depan.

Contoh:

```text
Laporan + TP + Kehadiran
            ↓
       proses semester
            ↓
      FINAL PROJECT
            ↓
       Individu
            ↓
     Final Total
```

Model early warning hanya boleh membaca bagian:

```text
Laporan + TP + Kehadiran
```

Target baru diketahui kemudian.

Jika `Final Total` dimasukkan ke X:

```text
Final Total -> Model -> Competency Label
```

maka model mendapatkan hampir langsung jawaban target.

Itu adalah leakage.

---

# 36. Struktur Dataset Final yang Disarankan

Pisahkan dataset menjadi tiga lapisan.

## `students_master.csv`

```text
NIM
Nama
Kelas
```

## `activities_long.csv`

Format:

```text
NIM
Kelas
Activity_Type
Activity_ID
Time_Index
Score
Available
```

Contoh:

```text
12345
A
TP
TP1
1
80
1
```

```text
12345
A
Laporan
Laporan1
1
75
1
```

```text
12345
A
Attendance
P1
1
1
```

Format long sangat direkomendasikan untuk membangun cutoff secara konsisten.

## `final_ground_truth.csv`

```text
NIM
Kelas
Final_CRUD_DB
Final_Fitur_Inovasi
Final_Kerapihan
Final_Individu
Final_Total
Competency_Label
```

---

# 37. Data Dictionary

| Variabel | Tipe | Dipakai sebagai X? | Keterangan |
|---|---|---:|---|
| NIM | identifier | Tidak | ID |
| Nama | identifier | Tidak | identitas |
| Kelas | categorical | Opsional | sebaiknya untuk audit/stratification, bukan predictor utama |
| Attendance_P* | numeric | Ya | aktivitas kehadiran |
| Laporan_* | numeric | Ya | nilai laporan |
| TP_* | numeric | Ya | nilai TP |
| TP_Mean | numeric | Ya | agregat |
| Laporan_Mean | numeric | Ya | agregat |
| Attendance_PreFinal_Rate | numeric | Ya | partisipasi |
| TP_Completion_Rate | numeric | Ya | konsistensi |
| Laporan_Completion_Rate | numeric | Ya | konsistensi |
| TP_Std | numeric | Ya | variabilitas |
| Laporan_Std | numeric | Ya | variabilitas |
| TP_First2_Mean | numeric | Ya | performa awal |
| Laporan_First2_Mean | numeric | Ya | performa awal |
| TP_Last2_Mean | numeric | Ya | performa terbaru |
| Laporan_Last2_Mean | numeric | Ya | performa terbaru |
| TP_Trend | numeric | Ya | trajectory |
| Laporan_Trend | numeric | Ya | trajectory |
| Performance_Std | numeric | Ya | volatilitas gabungan |
| Performance_Late_Mean | numeric | Ya | performa akhir cutoff |
| Absence_Count | numeric | Ya | jumlah absen |
| Final_Individu | numeric | **Tidak** | ground truth |
| Final_CRUD_DB | numeric | **Tidak** | komponen final |
| Final_Fitur_Inovasi | numeric | **Tidak** | komponen final |
| Final_Kerapihan | numeric | **Tidak** | komponen final |
| Final_Total | numeric | **Tidak** | agregat final |
| Competency_Label | binary | target | 1/0 |

---

# 38. Apakah Kelas Boleh Menjadi Fitur?

`Kelas` tidak disarankan menjadi predictor utama dalam model pertama.

Alasannya:

- hanya ada 5 kelas;
- kelas dapat menangkap perbedaan pengajar/jadwal/struktur;
- dapat mengurangi interpretabilitas individu;
- generalisasi ke kelas baru menjadi lebih lemah.

Gunakan `Kelas` untuk:

- stratifikasi;
- audit;
- analisis distribusi;
- subgroup analysis.

Eksperimen sensitivitas dengan `Kelas` sebagai predictor boleh dilakukan kemudian, tetapi jangan jadikan model utama secara otomatis.

---

# 39. Analisis Distribusi Sebelum Modeling

Wajib dilakukan terlebih dahulu.

Hitung:

```text
N total = 89
N Kompeten
N Belum Kompeten
Persentase setiap kelas
```

Lalu:

```text
Kelas A: kompeten / belum kompeten
Kelas B: ...
Kelas C: ...
Kelas D: ...
Kelas E: ...
```

Periksa juga:

- jumlah nilai E;
- minimum;
- maksimum;
- mean;
- median;
- SD;
- distribusi Final_Individu;
- distribusi Final_Total.

---

# 40. Audit Ground Truth

Sebelum modeling, buat tabel:

| NIM | Kelas | Individu | Final Total | Label | Nilai Huruf |
|---|---|---:|---:|---|---|

Periksa:

1. duplikasi NIM;
2. NIM yang tidak cocok;
3. nama yang berbeda antar sheet;
4. final kosong;
5. Individu kosong;
6. label tidak konsisten;
7. nilai E tetapi angka final tinggi;
8. angka final rendah tetapi huruf bukan E.

Semua koreksi harus dicatat.

---

# 41. Join Data

Join utama menggunakan:

```text
NIM
```

Bukan `Nama`.

Nama hanya sebagai fallback audit.

Jika NIM tidak tersedia pada salah satu sheet, lakukan:

```text
NIM -> primary key
Nama + Kelas -> secondary validation
```

Jangan melakukan fuzzy matching nama secara diam-diam.

---

# 42. Data Quality Report

Meskipun data disebut sudah bersih, tetap buat laporan:

```text
duplicate_count
missing_id_count
missing_activity_count
invalid_numeric_count
negative_score_count
score_above_100_count
final_missing_count
label_missing_count
```

Output contoh:

```text
Total student: 89
Duplicate NIM: 0
Missing target: 0
Invalid score: 0
```

Angka final harus berasal dari hasil preprocessing nyata, bukan ditulis manual.

---

# 43. Penanganan Perbedaan Jumlah Pertemuan

Data lama menunjukkan variasi pertemuan.

Untuk dataset baru, jumlah pertemuan per kelas harus dihitung langsung dari workbook.

Jangan menggunakan angka 6 atau 7 secara hard-code sebelum diperiksa.

Buat tabel:

| Kelas | Pertemuan | Laporan | TP |
|---|---:|---:|---:|
| A | hasil audit | hasil audit | hasil audit |
| B | hasil audit | hasil audit | hasil audit |
| C | hasil audit | hasil audit | hasil audit |
| D | hasil audit | hasil audit | hasil audit |
| E | hasil audit | hasil audit | hasil audit |

Jika struktur benar-benar sama, tabel ini menjadi bukti bahwa common window dapat dibuat lebih sederhana.

---

# 44. Rule untuk Mahasiswa Jarang Hadir

Untuk mahasiswa yang jarang hadir:

```text
Attendance = 0
```

jangan dihapus.

Untuk mahasiswa yang tidak mengerjakan TP:

```text
TP = 0
```

jangan dihapus.

Untuk mahasiswa yang tidak mengerjakan laporan:

```text
Laporan = 0
```

jangan dihapus.

Ini justru merupakan **signal akademik** yang ingin dipelajari model.

Namun:

> nilai 0 bukan berarti target otomatis Belum Kompeten.

Target tetap berasal dari ground truth final.

Dengan demikian penelitian dapat menguji:

> apakah pola ketidakhadiran dan ketidakpengerjaan aktivitas berkontribusi terhadap prediksi kompetensi.

---

# 45. Jangan Membuat Label dari Attendance

Jangan melakukan:

```text
Absence tinggi -> Belum Kompeten
```

sebelum melihat nilai final.

Attendance adalah predictor.

Competency_Label adalah ground truth.

Pemisahan ini harus dijaga.

---

# 46. Contoh Transformasi Mahasiswa

Misalnya:

```text
Attendance:
1, 1, 0, 1, 0, 1

TP:
80, 75, 0, 70

Laporan:
85, 90, 65, 0
```

Maka:

```text
Absence_Count = 2
Attendance_PreFinal_Rate = 4/6
TP_Completion_Rate = 3/4
Laporan_Completion_Rate = 3/4
TP_Mean = mean(80,75,0,70)
Laporan_Mean = mean(85,90,65,0)
```

Nilai 0 tetap masuk ke mean apabila 0 memang berarti tidak mengerjakan.

---

# 47. Hal yang Tidak Boleh Dilakukan

Jangan:

```text
0 -> NaN
```

bila 0 bermakna tidak hadir/tidak mengerjakan.

Jangan:

```text
NaN karena aktivitas belum tersedia -> 0
```

Jangan:

```text
FINAL -> predictor
```

Jangan:

```text
feature selection sebelum split
```

Jangan:

```text
SMOTE sebelum split
```

Jangan:

```text
imputation sebelum split
```

Jangan:

```text
memilih model dari hold-out test
```

Jangan:

```text
mengatakan SHAP = sebab-akibat
```

Jangan:

```text
menambahkan Logistic Regression
```

---

# 48. Urutan Eksperimen

## Eksperimen 1 — Baseline

Model:

```text
Dummy
Decision Tree
Random Forest
```

Feature set:

```text
S1
```

Cutoff:

```text
C_Full
```

Tujuan:

> mendapatkan baseline performa.

---

## Eksperimen 2 — Feature Expansion

Bandingkan:

```text
S1
S2
S3
S4
S5
```

Tujuan:

> menemukan titik kompleksitas yang paling informatif.

---

## Eksperimen 3 — Temporal

Bandingkan:

```text
C1
C2
C3
C_Full
```

dengan feature set yang sudah dipilih.

Tujuan:

> menemukan kapan signal sudah cukup stabil.

---

## Eksperimen 4 — Top-K

Uji:

```text
Top-10
Top-15
Top-20
```

Tujuan:

> menguji apakah pengurangan fitur meningkatkan stabilitas.

---

## Eksperimen 5 — XAI

Model final:

```text
Random Forest
```

Analisis:

```text
SHAP Global
SHAP Beeswarm
SHAP Dependence
SHAP Local
```

---

# 49. Model Selection

Model final **tidak dipilih karena satu angka Accuracy test tertinggi**.

Gunakan urutan:

1. Balanced Accuracy CV;
2. Recall Belum Kompeten;
3. F1 Macro;
4. SD/variabilitas CV;
5. kedinian cutoff;
6. jumlah fitur;
7. performa hold-out sebagai pemeriksaan akhir.

Model final harus dibekukan sebelum melihat test secara substantif.

---

# 50. Tabel Hasil yang Wajib Dibuat

## Tabel A — Distribusi Target

| Kelas | Kompeten | Belum Kompeten | Total |
|---|---:|---:|---:|
| A | - | - | - |
| B | - | - | - |
| C | - | - | - |
| D | - | - | - |
| E | - | - | - |
| Total | - | - | **89** |

---

## Tabel B — Distribusi Nilai

| Variabel | Min | Max | Mean | Median | SD |
|---|---:|---:|---:|---:|---:|
| Individu | - | - | - | - | - |
| Final Total | - | - | - | - | - |

---

## Tabel C — Perbandingan Model

| Feature Set | Model | CV BalAcc | Test Acc | Test BalAcc | Recall BK | F1 Macro |
|---|---|---:|---:|---:|---:|---:|
| S1 | Dummy | - | - | - | - | - |
| S1 | DT | - | - | - | - | - |
| S1 | RF | - | - | - | - | - |
| ... | ... | ... | ... | ... | ... | ... |

---

## Tabel D — Temporal

| Cutoff | Fitur | CV BalAcc | Recall BK | Mean ± SD | Status |
|---|---|---:|---:|---:|---|
| C1 | S* | - | - | - | - |
| C2 | S* | - | - | - | - |
| C3 | S* | - | - | - | - |
| C_Full | S* | - | - | - | - |

---

## Tabel E — Feature Importance

| Rank | Feature | Mean Abs SHAP | Makna |
|---:|---|---:|---|
| 1 | - | - | - |
| 2 | - | - | - |
| 3 | - | - | - |
| ... | ... | ... | ... |

---

# 51. Notebook/Script Structure

Struktur implementasi yang disarankan:

```text
project/
├── data/
│   ├── raw/
│   │   └── basis_data_non_relational.xlsx
│   ├── processed/
│   │   ├── students_master.csv
│   │   ├── activities_long.csv
│   │   └── final_ground_truth.csv
│   └── features/
│       ├── C1_S1.csv
│       ├── C2_S1.csv
│       ├── C3_S1.csv
│       └── C_Full_S1.csv
│
├── src/
│   ├── load_data.py
│   ├── audit_data.py
│   ├── build_long.py
│   ├── target.py
│   ├── temporal.py
│   ├── features.py
│   ├── pipeline.py
│   ├── train.py
│   ├── evaluate.py
│   ├── shap_analysis.py
│   └── error_analysis.py
│
├── outputs/
│   ├── data_quality_report.csv
│   ├── class_distribution.csv
│   ├── model_results.csv
│   ├── temporal_results.csv
│   ├── confusion_matrices/
│   ├── shap/
│   └── error_analysis.csv
│
└── UBAHDATA.md
```

---

# 52. Checklist Implementasi

## Data

- [ ] Import workbook.
- [ ] Baca sheet A-E.
- [ ] Baca sheet FINAL.
- [ ] Normalisasi nama kolom.
- [ ] Audit NIM.
- [ ] Audit duplikasi.
- [ ] Pastikan total mahasiswa = 89.
- [ ] Hitung jumlah mahasiswa tiap kelas.
- [ ] Audit nilai E.
- [ ] Audit nilai 0.
- [ ] Audit missing.

## Target

- [ ] Ambil `Individu`.
- [ ] Ambil `Nilai Final Total`.
- [ ] Bentuk `Competency_Label` dari `Individu`.
- [ ] Simpan `Final_Total` sebagai sensitivity ground truth.
- [ ] Jangan masukkan final components ke X.

## Temporal

- [ ] Tentukan aktivitas yang tersedia pada setiap cutoff.
- [ ] Pisahkan unavailable dari true zero.
- [ ] Bekukan C1-C4/C_Full.
- [ ] Dokumentasikan jumlah aktivitas pada setiap cutoff.

## Feature engineering

- [ ] Attendance rate.
- [ ] Absence count.
- [ ] TP mean.
- [ ] TP completion.
- [ ] TP std.
- [ ] TP first2.
- [ ] TP last2.
- [ ] TP trend.
- [ ] Laporan mean.
- [ ] Laporan completion.
- [ ] Laporan std.
- [ ] Laporan first2.
- [ ] Laporan last2.
- [ ] Laporan trend.
- [ ] Performance std.
- [ ] Performance late mean.

## Modeling

- [ ] Dummy.
- [ ] Decision Tree.
- [ ] Random Forest.
- [ ] Tidak menggunakan Logistic Regression.
- [ ] Imputation dalam pipeline.
- [ ] Balancing dalam pipeline.
- [ ] Selection dalam inner CV.
- [ ] Tuning dalam inner CV.
- [ ] Repeated stratified CV.
- [ ] Nested CV.
- [ ] Hold-out dikunci.

## Evaluation

- [ ] Accuracy.
- [ ] Balanced Accuracy.
- [ ] Recall Belum Kompeten.
- [ ] Precision.
- [ ] F1 Macro.
- [ ] Confusion matrix.
- [ ] Mean ± SD.

## Explainability

- [ ] SHAP global.
- [ ] SHAP ranking.
- [ ] Beeswarm.
- [ ] Dependence.
- [ ] Local TP/TN.
- [ ] Local FP/FN.

## Error analysis

- [ ] False Negative.
- [ ] False Positive.
- [ ] Late-Dropper.
- [ ] Late-Bloomer.
- [ ] Pola attendance.
- [ ] Pola completion.
- [ ] Pola trend.
- [ ] Pola volatility.

---

# 53. Hipotesis Eksperimental

Penelitian tidak perlu membuat klaim kausal. Hipotesis dibuat dalam bentuk empiris.

## H1

> Penambahan fitur completion dari S1 ke S2 dapat meningkatkan kemampuan diskriminasi model terhadap status kompetensi.

## H2

> Penambahan fitur variability dari S2 ke S3 dapat meningkatkan Balanced Accuracy atau Recall Belum Kompeten dibanding feature set dasar.

## H3

> Fitur temporal dan trajectory pada S4/S5 dapat memberikan informasi tambahan mengenai mahasiswa yang mengalami perubahan performa.

## H4

> Terdapat cutoff temporal sebelum final pada saat signal kompetensi sudah cukup stabil untuk early warning.

## H5

> Random Forest memberikan performa diskriminasi yang lebih stabil dibandingkan Decision Tree pada sebagian konfigurasi feature set.

## H6

> Fitur performa awal, performa terbaru, attendance, completion, dan/atau volatility dapat menunjukkan kontribusi prediktif tinggi menurut TreeSHAP.

Hasil dapat mendukung atau menolak hipotesis tersebut.

---

# 54. Research Questions Baru

### RQ1

Apakah Random Forest memberikan performa yang lebih baik dibandingkan Decision Tree pada ruang fitur yang sama?

### RQ2

Apakah penambahan completion dan attendance meningkatkan kemampuan deteksi mahasiswa Belum Kompeten?

### RQ3

Apakah fitur variability dan trajectory menghasilkan peningkatan performa yang konsisten pada dataset small-sample?

### RQ4

Pada cutoff temporal mana signal kompetensi mulai cukup stabil untuk digunakan sebagai early warning?

### RQ5

Fitur apa yang paling berkontribusi terhadap prediksi Random Forest berdasarkan TreeSHAP?

### RQ6

Apa karakteristik False Negative dan False Positive pada data mahasiswa Basis Data Non Relasional?

### RQ7

Apakah kesimpulan model relatif konsisten ketika target dibandingkan antara `Final_Individu` dan `Final_Total` melalui sensitivity analysis?

---

# 55. Novelty Setelah Migrasi

Jangan menyatakan novelty:

> "Random Forest belum pernah digunakan untuk prediksi kompetensi mahasiswa."

Itu terlalu umum.

Novelty lebih tepat diarahkan pada integrasi:

```text
common temporal window
+
activity-based feature engineering
+
small-sample validation
+
imbalance-aware evaluation
+
TreeSHAP
+
error analysis
```

dalam konteks **Basis Data Non Relasional**.

Kontribusi empiris yang dapat dicari:

1. menemukan titik kompleksitas feature set yang paling stabil;
2. menentukan seberapa dini signal kompetensi muncul;
3. mengidentifikasi pola aktivitas yang berkontribusi terhadap prediksi;
4. menemukan blind spot model pada FN/FP;
5. menguji sensitivitas definisi kompetensi individual versus total final.

---

# 56. Hal yang Harus Dikunci Sebelum Eksperimen

Sebelum menjalankan model, freeze:

```text
1. Definisi target
2. Threshold kompetensi
3. Daftar predictor
4. Daftar fitur final
5. Common temporal window
6. Feature set S1-S5/S6
7. Prosedur split
8. Inner CV
9. Outer CV
10. Random state
11. Metric utama
12. Aturan model selection
13. Aturan threshold selection
14. Aturan feature selection
```

Setelah ini hold-out tidak boleh digunakan untuk mengubah desain.

---

# 57. Alur Penelitian Final

```text
WORKBOOK BASIS DATA NON RELASIONAL
                 │
                 ▼
          Data Audit Awal
                 │
                 ├── A
                 ├── B
                 ├── C
                 ├── D
                 ├── E
                 └── FINAL
                 │
                 ▼
           Join berdasarkan NIM
                 │
                 ▼
        Pisahkan Data Final
                 │
                 ├── Ground Truth
                 │      └── Final_Individu
                 │
                 └── Audit/Sensitivity
                        └── Final_Total
                 │
                 ▼
        Common Temporal Window
                 │
            ┌────┼────┐
            ▼    ▼    ▼
           C1   C2   C3 ... C_Full
            │
            ▼
      Feature Engineering
            │
            ├── S1 Basic
            ├── S2 Completion
            ├── S3 Variability
            ├── S4 Temporal
            └── S5 Combined
            │
            ▼
       CV-aware Pipeline
            │
            ├── Imputation
            ├── Balancing
            ├── Feature Selection
            └── Hyperparameter Tuning
            │
            ▼
       Model Comparison
            │
            ├── Dummy
            ├── Decision Tree
            └── Random Forest
            │
            ▼
       Repeated / Nested CV
            │
            ▼
         Model Selection
            │
            ▼
        Final Hold-out
            │
            ├── Metrics
            ├── Confusion Matrix
            ├── TreeSHAP
            └── Error Analysis
                    │
                    ▼
           Early Warning Finding
```

---

# 58. Kesimpulan Keputusan Migrasi

Keputusan utama untuk dataset **89 mahasiswa / 5 kelas** adalah:

### Data aktivitas

Gunakan:

```text
Attendance
Laporan
TP
```

sebagai sumber predictor.

### Nilai 0

Pertahankan sebagai nilai nyata ketika berarti:

```text
tidak hadir
tidak mengerjakan
nilai sebenarnya 0
```

### Nilai belum tersedia

Gunakan:

```text
NaN
```

bila aktivitas memang belum tersedia karena temporal/class structure.

### Sheet FINAL

Baca **seluruh kolom** untuk audit:

```text
CRUD & DB
Fitur dan Inovasi
Kerapihan
Individu
Nilai Final Total
```

tetapi jangan menjadikannya predictor.

### Target utama

Pertahankan prinsip penelitian lama:

```text
Final_Individu >= 75 -> Kompeten
Final_Individu < 75  -> Belum Kompeten
```

### Sensitivity target

Tambahkan:

```text
Final_Total >= 75 -> Kompeten
Final_Total < 75  -> Belum Kompeten
```

hanya untuk robustness/sensitivity analysis.

### Model

Gunakan:

```text
Dummy
Decision Tree
Random Forest
```

**Tanpa Logistic Regression.**

### Feature engineering

Prioritaskan:

```text
level
completion
attendance
variability
early performance
recent performance
trend
trajectory
```

### Early warning

Gunakan:

```text
C1
C2
C3
C_Full
```

dan tambahkan C4 hanya jika data aktivitas memang mendukung cutoff tersebut.

### XAI

Gunakan:

```text
TreeSHAP
```

untuk menjawab:

> fitur mana yang paling berkontribusi terhadap prediksi model?

Bukan:

> fitur mana yang menyebabkan mahasiswa kompeten?

---

# 59. Checklist Keputusan Final untuk Sheet FINAL

Sebelum coding, gunakan aturan ini sebagai keputusan tetap:

| Kolom FINAL | Diambil? | Menjadi X? | Menjadi target? | Untuk audit? |
|---|---:|---:|---:|---:|
| Nama | Ya | Tidak | Tidak | Ya |
| NIM | Ya | Tidak | Tidak | Ya |
| Kelas | Ya | Tidak pada model utama | Tidak | Ya |
| CRUD & DB | Ya | **Tidak** | Tidak | Ya |
| Fitur dan Inovasi | Ya | **Tidak** | Tidak | Ya |
| Kerapihan | Ya | **Tidak** | Tidak | Ya |
| Individu | **Ya** | **Tidak** | **Ya — utama** | Ya |
| Nilai Final Total | **Ya** | **Tidak** | **Tidak — sensitivity** | Ya |
| Nilai Huruf/E | Ya bila tersedia | **Tidak** | Tidak | Ya |

---

# 60. Kalimat Metodologi yang Dapat Dipakai di Manuskrip

> Dataset penelitian terdiri atas 89 mahasiswa pada lima kelas mata kuliah Basis Data Non Relasional. Data aktivitas pembelajaran mencakup kehadiran, nilai laporan, dan tugas pendahuluan (TP), sedangkan nilai evaluasi akhir diperoleh dari lembar penilaian final. Status kompetensi dibentuk berdasarkan nilai penilaian individu pada evaluasi akhir, dengan mahasiswa dikategorikan Kompeten apabila memperoleh nilai ≥75 dan Belum Kompeten apabila memperoleh nilai <75. Seluruh komponen penilaian final digunakan sebagai sumber ground truth dan audit, tetapi tidak dimasukkan sebagai predictor agar tidak terjadi target leakage. Nilai nol yang merepresentasikan ketidakhadiran atau ketidakmengerjaan aktivitas dipertahankan sebagai observasi valid, sedangkan aktivitas yang belum tersedia pada suatu cutoff temporal diperlakukan sebagai nilai yang belum teramati. Eksperimen membandingkan Dummy Classifier, Decision Tree, dan Random Forest pada beberapa tingkat feature engineering dan temporal cutoff. Evaluasi dilakukan menggunakan stratified repeated/nested cross-validation dan hold-out test dengan Balanced Accuracy, Recall kelas Belum Kompeten, F1 Macro, Accuracy, Precision, dan confusion matrix. Interpretasi model dilakukan menggunakan TreeSHAP serta error analysis terhadap False Positive dan False Negative.

---

# 61. Ringkasan Arsitektur Data

Secara konseptual:

```text
                    ┌────────────────────────┐
                    │  SHEET A-E             │
                    │  Attendance            │
                    │  Laporan               │
                    │  TP                    │
                    └───────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Temporal Feature Space  │
                   │ C1 / C2 / C3 / Full     │
                   └────────────┬────────────┘
                                │
                                ▼
                  ┌──────────────────────────┐
                  │ S1 → S2 → S3 → S4 → S5  │
                  └────────────┬─────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │ DT / RF / Dummy              │
                └─────────────┬────────────────┘
                              │
                ┌─────────────┴────────────┐
                ▼                          ▼
        Performance Metrics           TreeSHAP
                │                          │
                └─────────────┬────────────┘
                              ▼
                      Error Analysis
                              │
                              ▼
                     Early Warning
```

Sedangkan ground truth berada di jalur terpisah:

```text
FINAL
  │
  ├── CRUD & DB
  ├── Fitur & Inovasi
  ├── Kerapihan
  ├── Individu ───────────────► Competency_Label
  └── Nilai Final Total ──────► Sensitivity / Audit
```

Dengan struktur tersebut, informasi masa depan tetap terpisah dari ruang fitur early warning.

---

# 62. Keputusan Utama Penelitian Baru

**Jangan menjadikan `Nilai Final Total` sebagai predictor.**

**Jangan membuang nilai 0 yang memang berarti tidak hadir/tidak mengerjakan.**

**Jangan mengubah aktivitas yang belum tersedia menjadi 0.**

**Gunakan `Individu` sebagai target utama agar migrasi tetap selaras dengan definisi target penelitian sebelumnya, dengan `Nilai Final Total` sebagai target sensitivitas/audit.**

**Gunakan semua komponen pada sheet FINAL untuk audit, tetapi pisahkan tegas dari X.**

**Gunakan Dummy, Decision Tree, dan Random Forest; Logistic Regression dihapus.**

**Pertahankan temporal cutoff karena nilai penelitian tetap berada pada konsep early warning, bukan klasifikasi akhir semester.**

**Dengan n=89, validasi berulang dan nested CV menjadi semakin penting karena varians estimasi performa akan lebih besar daripada dataset sebelumnya.**

---

# 63. Status Migrasi

Dokumen ini menjadi baseline spesifikasi eksperimen.

Yang **sudah diputuskan**:

```text
Dataset                 = Basis Data Non Relasional
N                       = 89
Kelas                   = A-E
Predictor utama         = Attendance + Laporan + TP
Target utama            = Final_Individu >= 75
Target sensitivity      = Final_Total >= 75
Nilai 0 nyata           = dipertahankan
Unavailable activity    = NaN
Logistic Regression     = dihapus
Model                   = Dummy + DT + RF
XAI                     = TreeSHAP
Temporal                = C1/C2/C3/C_Full
Evaluasi                = Repeated/Nested CV + Hold-out
Metrik utama            = Balanced Accuracy + Recall Belum Kompeten
```

Yang **belum boleh diisi dengan asumsi** dan harus dihitung dari workbook:

```text
jumlah tiap kelas
jumlah pertemuan tiap kelas
jumlah TP tiap kelas
jumlah laporan tiap kelas
jumlah Kompeten
jumlah Belum Kompeten
jumlah E
distribusi nilai
cutoff persis C1/C2/C3/C_Full
jumlah fitur setiap S1-S5
hyperparameter terbaik
hasil model
hasil SHAP
jumlah FN/FP
```

Semua angka tersebut harus dihasilkan dari data aktual, bukan diambil dari dataset penelitian lama.

---

# 64. Prioritas Implementasi

Urutan implementasi paling aman:

```text
1. Baca Excel
2. Audit 89 mahasiswa
3. Join A-E dengan FINAL berdasarkan NIM
4. Bentuk final_ground_truth
5. Verifikasi Individu dan Final Total
6. Bentuk Competency_Label
7. Ubah activity table menjadi long format
8. Tentukan common temporal window
9. Generate S1-S5
10. Build CV-aware pipelines
11. Benchmark Dummy/DT/RF
12. Repeated CV
13. Nested CV
14. Lock kandidat final
15. Hold-out test
16. TreeSHAP
17. FN/FP analysis
18. Sensitivity target Final_Total
19. Freeze hasil
20. Tulis manuscript
```

---

# 65. Prinsip Akhir

Penelitian baru jangan diperlakukan sebagai sekadar:

> "ganti mata kuliah dan masukkan Excel baru."

Yang dilakukan adalah **replikasi terkontrol dengan domain baru**.

Struktur metodologis lama dipertahankan, tetapi feature space, jumlah observasi, target audit, temporal window, distribusi kelas, dan hasil eksperimen harus dihitung ulang dari awal.

Dengan demikian penelitian tetap memiliki rantai metodologis:

```text
data aktivitas nyata
        ↓
temporal representation
        ↓
feature engineering
        ↓
ML early warning
        ↓
validasi leakage-aware
        ↓
prediksi mahasiswa berisiko
        ↓
TreeSHAP
        ↓
error analysis
        ↓
interpretasi kontribusi fitur
```

dan bukan:

```text
nilai final
↓
predict nilai final
↓
klaim early warning
```

Itu adalah batas metodologis utama yang harus dijaga dalam seluruh migrasi penelitian ini.
