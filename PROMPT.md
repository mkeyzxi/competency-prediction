# PROMPT — BUAT FILE `ANALYSIS_GUIDE.md`

Saya ingin Anda membuat satu file dokumentasi bernama:

```text
ANALYSIS_GUIDE.md
```

File ini harus menjadi **panduan audit, analisis, dan interpretasi hasil eksperimen machine learning** dari project penelitian saya.

Jangan hanya menjelaskan teori. Anda harus **membaca dan memeriksa implementasi project yang sudah ada**, terutama source code, konfigurasi, hasil eksperimen, CSV, JSON, log, confusion matrix, SHAP, dan seluruh output yang tersedia.

## 1. KONTEKS PENELITIAN

Penelitian ini berjudul:

> Prediksi Kompetensi Mahasiswa pada Praktikum Logika Pemrograman Menggunakan Decision Tree dan Random Forest dengan Interpretasi TreeSHAP.

Tujuan penelitian:

1. Membangun model prediksi kompetensi mahasiswa berdasarkan aktivitas/performa praktikum.
2. Membandingkan Decision Tree dan Random Forest.
3. Menentukan pengaruh/kontribusi feature engineering terhadap performa.
4. Menginterpretasikan model terbaik menggunakan TreeSHAP.

Label kompetensi:

```text
Final Individu >= 75 → Kompeten (1)
Final Individu < 75  → Belum Kompeten (0)
```

Ketentuan penting:

- `Final Individu` digunakan untuk membentuk target.
- Setelah label dibuat, `Final Individu` TIDAK BOLEH menjadi fitur.
- `Total` tidak boleh digunakan sebagai fitur utama karena merupakan agregasi nilai dan berpotensi menyebabkan data leakage.
- Nilai `0` dan `40` harus diperlakukan berdasarkan makna akademiknya dan tidak boleh otomatis dianggap missing/outlier.
- Kelas A–D digunakan sebagai konteks analisis, bukan otomatis sebagai fitur utama.
- Model utama yang dibandingkan adalah Decision Tree dan Random Forest.
- TreeSHAP digunakan untuk menjelaskan kontribusi fitur pada model pohon, terutama Random Forest.
- Jangan membuat klaim hubungan sebab-akibat dari SHAP.

## 2. TUJUAN UTAMA FILE

`ANALYSIS_GUIDE.md` harus membantu saya menjawab secara akademis:

- Mengapa nilai hyperparameter tertentu digunakan?
- Bagaimana `max_depth` dipilih?
- Apa dampak `max_depth` terhadap Decision Tree dan Random Forest?
- Parameter Random Forest apa saja yang digunakan?
- Parameter Decision Tree apa saja yang digunakan?
- Apakah hyperparameter diperoleh dari tuning atau ditentukan manual?
- Apakah tuning dilakukan dengan benar tanpa data leakage?
- Mengapa model tertentu menjadi model terbaik?
- Apa arti nilai CV F1?
- Apa arti Test F1?
- Mengapa CV F1 dan Test F1 bisa berbeda?
- Bagaimana menilai kestabilan model?
- Apa arti confusion matrix?
- Apa arti TP, TN, FP, FN dalam konteks penelitian ini?
- Apa arti Precision, Recall, F1-Score, dan Accuracy?
- Kelas mana yang menjadi perhatian utama?
- Apakah model bagus dalam menemukan mahasiswa yang belum kompeten?
- Apa arti setiap grafik yang dihasilkan pipeline?
- Apa output setiap file pada `results/`?
- Bagaimana membaca grafik TreeSHAP?
- Apa arti SHAP value positif dan negatif?
- Apa arti besar kecilnya SHAP value?
- Apa fitur paling berkontribusi?
- Apakah fitur tersebut benar-benar menyebabkan kompetensi?
- Bagaimana menjelaskan SHAP secara akademis?
- Apakah S1, S2, atau S3 lebih baik?
- Apakah feature engineering benar-benar meningkatkan performa?
- Apakah Random Forest benar-benar lebih baik dari Decision Tree?
- Apakah hasil eksperimen cukup stabil?
- Apakah ada indikasi overfitting?
- Apakah ada indikasi underfitting?
- Apakah hasil test dapat dipercaya?
- Apakah ada data leakage?
- Apakah pipeline sesuai dengan metodologi penelitian?
- Apakah hasil sudah cukup untuk digunakan dalam BAB IV?
- Apa saja kekurangan hasil eksperimen?
- Eksperimen tambahan apa yang masih diperlukan?

## 3. WAJIB MEMBACA IMPLEMENTASI AKTUAL

Sebelum menulis `ANALYSIS_GUIDE.md`, periksa seluruh project.

Minimal periksa:

```text
.
├── scripts/
├── src/
├── config/
├── data/
├── results/
├── tests/
├── environment.yml
├── requirements.txt
└── file konfigurasi lain yang digunakan
```

Jika struktur folder berbeda, cari lokasi sebenarnya.

Jangan mengarang nama file atau parameter.

Identifikasi secara aktual:

- preprocessing
- feature engineering
- train-test split
- cross-validation
- hyperparameter tuning
- Decision Tree
- Random Forest
- scoring
- model selection
- SHAP
- output metrics
- output plots
- random seed
- class weighting
- imputation
- scaling
- feature selection
- pipeline
- serialization model
- configuration

## 4. ANALISIS HYPERPARAMETER

Buat bagian khusus:

```markdown
## Hyperparameter Audit
```

Jelaskan seluruh hyperparameter yang digunakan.

Minimal:

### Decision Tree

Periksa:

```text
criterion
splitter
max_depth
min_samples_split
min_samples_leaf
max_features
class_weight
random_state
```

### Random Forest

Periksa:

```text
n_estimators
criterion
max_depth
min_samples_split
min_samples_leaf
max_features
class_weight
bootstrap
random_state
```

Jangan mengasumsikan semua parameter digunakan. Tuliskan hanya parameter yang benar-benar ditemukan dalam source code/configuration.

## 5. ANALISIS KHUSUS `max_depth`

Buat bagian:

```markdown
## Analisis max_depth
```

Jelaskan dengan bahasa mudah tetapi akademis:

1. Apa itu `max_depth`.
2. Apa yang terjadi jika `max_depth` kecil.
3. Apa yang terjadi jika `max_depth` besar.
4. Hubungan `max_depth` dengan:
   - underfitting
   - overfitting
   - kompleksitas model
   - interpretabilitas

5. Perbedaan dampak `max_depth` pada:
   - Decision Tree
   - Random Forest

6. Bagaimana `max_depth` dipilih dalam project.
7. Apakah nilainya:
   - manual
   - default
   - hasil GridSearchCV
   - RandomizedSearchCV
   - atau metode lain.

8. Tunjukkan source code yang menjadi bukti.
9. Jelaskan parameter kandidat yang diuji jika tersedia.
10. Jelaskan mengapa konfigurasi terpilih menjadi pemenang berdasarkan metric yang digunakan.

Buat tabel seperti:

| Model         | Parameter | Kandidat | Nilai Terpilih | Metode Pemilihan | Scoring |
| ------------- | --------- | -------- | -------------- | ---------------- | ------- |
| Decision Tree | max_depth | ...      | ...            | ...              | ...     |
| Random Forest | max_depth | ...      | ...            | ...              | ...     |

Jangan mengisi `...` dengan asumsi.

## 6. AUDIT HYPERPARAMETER TUNING

Cari dan jelaskan:

```text
GridSearchCV
RandomizedSearchCV
cross_validate
cross_val_score
StratifiedKFold
StratifiedShuffleSplit
```

atau mekanisme tuning lain.

Pastikan menjawab:

- Dataset mana yang digunakan untuk tuning?
- Apakah test set sudah dipisahkan sebelum tuning?
- Apakah CV dilakukan hanya pada training set?
- Apakah preprocessing dilakukan dalam Pipeline?
- Apakah imputasi dipelajari hanya dari training?
- Apakah feature engineering mengandung informasi test?
- Apakah model terbaik dipilih berdasarkan CV?
- Apakah test set benar-benar hanya digunakan sekali sebagai evaluasi akhir?

Berikan status:

```text
PASS
WARNING
FAIL
```

untuk masing-masing aspek.

## 7. AUDIT DATA LEAKAGE

Buat bagian:

```markdown
## Data Leakage Audit
```

Minimal periksa:

```text
Final Individu
Total
target/label
fitur agregat
preprocessing
imputation
feature selection
hyperparameter tuning
test set
cross-validation
SHAP
```

Cari kemungkinan leakage langsung maupun tidak langsung.

Untuk setiap temuan gunakan tabel:

| Komponen | Status | Risiko | Bukti Source Code | Rekomendasi |
| -------- | ------ | ------ | ----------------- | ----------- |

Jangan memberikan status PASS tanpa bukti.

## 8. ANALISIS DATASET

Jelaskan:

- jumlah mahasiswa
- jumlah fitur
- fitur yang digunakan
- kelas target
- distribusi kelas
- jumlah data train
- jumlah data test
- distribusi label train
- distribusi label test
- missing value
- nilai unik yang penting
- potensi imbalance
- kelas A/B/C/D jika tersedia

Jelaskan juga apakah ukuran dataset sesuai untuk kompleksitas model.

## 9. ANALISIS S1, S2, S3

Buat bagian:

```markdown
## Feature Engineering Analysis
```

Gunakan definisi:

### S1 — Basic

```text
Attendance_Rate
TP_Mean
Respons_Mean
Laporan_Mean
```

### S2 — Behavioral

S1 +:

```text
TP_Completion_Rate
Respons_Completion_Rate
Laporan_Completion_Rate
```

### S3 — Relational

S2 +:

```text
Respons_TP_Gap
```

Periksa implementasi aktual apakah benar-benar sesuai.

Kemudian analisis:

- apakah S2 meningkatkan performa dari S1?
- apakah S3 meningkatkan performa dari S2?
- model mana yang paling diuntungkan?
- apakah penambahan fitur benar-benar berguna?
- apakah ada fitur yang tidak memberikan manfaat?

Jangan menganggap S2/S3 lebih bagus sebelum melihat hasil aktual.

## 10. ANALISIS SEMUA MODEL

Baca file:

```text
results/metrics/
```

dan seluruh file metrics yang tersedia.

Buat tabel perbandingan aktual:

| Skenario | Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV Std | Test Accuracy | Test Precision | Test Recall | Test F1 |
| -------- | ----- | ----------: | -----------: | --------: | ----: | -----: | ------------: | -------------: | ----------: | ------: |

Gunakan angka aktual.

Jika suatu metrik tidak tersedia, tuliskan:

```text
N/A — tidak tersedia pada output pipeline
```

Jangan menghitung ulang kecuali data yang diperlukan memang tersedia.

## 11. ANALISIS PERBEDAAN CV VS TEST

Jelaskan secara konkret untuk setiap model:

- CV F1
- Test F1
- selisihnya
- interpretasi

Contoh format:

```text
CV F1       = ...
Test F1     = ...
Difference  = ...
Assessment  = ...
```

Kemudian klasifikasikan:

```text
Excellent / Reasonable / Warning
```

Tetapi jangan membuat threshold sembarangan. Jelaskan bahwa kategori tersebut merupakan interpretasi eksploratif, bukan standar universal.

## 12. ANALISIS OVERFITTING

Periksa indikasi:

```text
CV tinggi + Test jauh lebih rendah
Train tinggi + Validation rendah
Train tinggi + Test rendah
```

Jika train score tersedia, analisis:

```text
Train → CV → Test
```

Jangan menyatakan overfitting hanya karena CV dan Test berbeda sedikit.

## 13. CONFUSION MATRIX

Cari semua file di:

```text
results/confusion_matrix/
```

Untuk setiap gambar:

1. sebutkan nama file;
2. model;
3. skenario;
4. split/evaluasi;
5. TP;
6. TN;
7. FP;
8. FN;
9. interpretasi;
10. konsekuensi akademis.

Pastikan definisi kelas mengikuti project:

```text
1 = Kompeten
0 = Belum Kompeten
```

Beri perhatian khusus pada:

```text
False Negative untuk kelas Belum Kompeten
```

karena secara praktis mahasiswa yang sebenarnya belum kompeten tetapi diprediksi kompeten dapat terlewat untuk intervensi.

Jika confusion matrix hanya ditampilkan dalam bentuk gambar dan angka tidak dapat dibaca dari metadata/output CSV, jangan mengarang angkanya.

Tuliskan:

```text
Angka harus diverifikasi dari file asli.
```

## 14. ANALISIS PRECISION, RECALL, F1

Jelaskan masing-masing dalam konteks penelitian.

Jangan hanya memberikan definisi matematika.

Gunakan contoh konteks:

```text
Precision:
Dari mahasiswa yang diprediksi Kompeten, berapa yang benar-benar Kompeten?

Recall:
Dari mahasiswa yang sebenarnya Kompeten, berapa yang berhasil ditemukan model?
```

Kemudian lakukan hal yang sama untuk kelas `Belum Kompeten`.

Jika scoring menggunakan binary/macro/weighted, jelaskan scoring yang benar-benar digunakan oleh pipeline.

## 15. ANALISIS TREE SHAP

Cari:

```text
results/shap/
```

Periksa seluruh gambar dan data SHAP.

Untuk setiap output:

- nama file
- model
- skenario
- tipe plot
- tujuan plot
- cara membaca
- temuan aktual

Minimal identifikasi:

```text
Feature Importance
Summary Plot
Beeswarm
Bar Plot
Dependence Plot
Waterfall Plot
Force Plot
Local Explanation
```

Jika jenis plot tertentu tidak ada, jangan mengarang bahwa plot tersebut ada.

## 16. PENJELASAN SHAP VALUE

Berikan penjelasan khusus:

### SHAP value positif

Jelaskan apakah mendorong prediksi menuju kelas tertentu berdasarkan konfigurasi output model.

### SHAP value negatif

Jelaskan arah pengaruh terhadap output yang dianalisis.

### SHAP value besar

Jelaskan bahwa fitur memberikan kontribusi besar terhadap prediksi model.

### SHAP value mendekati nol

Jelaskan bahwa kontribusinya relatif kecil pada observasi tersebut.

JANGAN menyimpulkan:

```text
fitur X menyebabkan mahasiswa kompeten
```

Gunakan:

```text
fitur X memberikan kontribusi terhadap prediksi model
```

## 17. ANALISIS SETIAP GAMBAR

Ini merupakan bagian paling penting.

Cari seluruh:

```text
*.png
*.jpg
*.jpeg
*.svg
```

di:

```text
results/
```

Buat daftar:

| File | Jenis Visualisasi | Model | Skenario | Tujuan | Temuan Aktual | Digunakan di BAB IV? |
| ---- | ----------------- | ----- | -------- | ------ | ------------- | -------------------- |

Untuk SETIAP gambar jelaskan:

```text
1. Gambar ini menunjukkan apa?
2. Sumbu X menunjukkan apa?
3. Sumbu Y menunjukkan apa?
4. Warna menunjukkan apa?
5. Titik/batang menunjukkan apa?
6. Bagaimana cara membacanya?
7. Apa temuan aktualnya?
8. Apa kesimpulan yang boleh dibuat?
9. Apa kesimpulan yang tidak boleh dibuat?
10. Apakah gambar layak masuk BAB IV?
```

Jangan hanya melihat nama file. Bila memungkinkan, baca metadata atau source code pembuat visualisasi untuk memastikan maknanya.

## 18. ANALISIS TREE / DECISION TREE OUTPUT

Jika terdapat plot pohon Decision Tree:

jelaskan:

- root node
- split
- feature
- threshold
- gini/entropy
- samples
- value
- class
- leaf
- depth

Kemudian jelaskan bagaimana `max_depth` terlihat pada visualisasi.

Jika tidak terdapat visualisasi tree, nyatakan bahwa output tersebut tidak tersedia.

## 19. MODEL TERBAIK

Tentukan kandidat model terbaik berdasarkan data aktual.

Jangan otomatis memilih Random Forest.

Jelaskan:

```text
Model terbaik =
alasan =
metric utama =
metric pendukung =
CV =
Test =
stabilitas =
confusion matrix =
```

Jika terdapat konflik antara CV dan Test, jelaskan konflik tersebut.

## 20. MODEL SELECTION RULE

Cari di source code bagaimana model dipilih.

Contoh kemungkinan:

```text
best_params_
best_score_
GridSearchCV.best_estimator_
max(test_f1)
max(cv_f1)
```

Dokumentasikan secara eksplisit.

Jika implementasi memilih model berdasarkan test set untuk tuning/model selection, tandai:

```text
FAIL — berpotensi test-set leakage
```

## 21. REPRODUCIBILITY AUDIT

Periksa:

```text
random_state
numpy seed
python seed
library version
environment.yml
requirements.txt
dataset version
configuration
```

Berikan status:

```text
PASS
WARNING
FAIL
```

## 22. OUTPUT FILE AUDIT

Buat inventaris seluruh output:

```text
results/
```

Contohnya:

```text
results/
├── metrics/
├── confusion_matrix/
├── shap/
├── models/
├── predictions/
└── ...
```

Untuk setiap file jelaskan:

| File | Format | Sumber Script | Isi | Kegunaan |
| ---- | ------ | ------------- | --- | -------- |

Jika ada file hasil eksperimen yang redundant, tandai.

## 23. KESALAHAN DAN ANOMALI

Cari secara aktif:

- angka yang tidak konsisten
- nama model yang salah
- nama skenario tidak konsisten
- metric tidak cocok
- file kosong
- output duplikat
- test score yang mencurigakan
- CV score yang identik secara tidak wajar
- hasil S2/S3 identik
- parameter tuning yang tidak digunakan
- konfigurasi yang tidak dibaca
- random state yang tidak konsisten
- preprocessing di luar pipeline
- leakage

Buat bagian:

```markdown
## Findings / Anomalies
```

dengan severity:

```text
CRITICAL
HIGH
MEDIUM
LOW
INFO
```

## 24. JAWABAN SIAP UNTUK DOSEN/PENGUJI

Buat bagian:

```markdown
## Pertanyaan yang Kemungkinan Ditanyakan Penguji
```

Minimal jawab:

### Mengapa menggunakan Random Forest?

### Mengapa dibandingkan dengan Decision Tree?

### Mengapa menggunakan `max_depth`?

### Bagaimana menentukan `max_depth`?

### Mengapa menggunakan 5-Fold CV?

### Mengapa menggunakan Stratified CV?

### Mengapa split 80:20?

### Mengapa Final Individu tidak digunakan sebagai fitur?

### Mengapa Total tidak digunakan?

### Mengapa nilai 0 dan 40 tidak dihapus?

### Mengapa S1, S2, S3 dibuat?

### Mengapa S2 lebih baik/buruk dari S1?

### Mengapa S3 lebih baik/buruk dari S2?

### Mengapa Random Forest lebih baik/buruk dari Decision Tree?

### Apa arti F1 = ...?

### Apa arti CV F1 = ...?

### Mengapa Test F1 berbeda dari CV F1?

### Apa arti confusion matrix?

### Apa arti False Negative?

### Apa arti SHAP?

### Apakah SHAP membuktikan sebab-akibat?

### Apa fitur paling berkontribusi?

### Mengapa fitur tersebut penting?

### Apakah model dapat digunakan untuk memvonis kompetensi mahasiswa?

Jawaban harus berdasarkan implementasi dan hasil aktual project.

## 25. JAWABAN BAB IV

Buat bagian:

```markdown
## Draft Struktur Analisis BAB IV
```

Berikan struktur:

```text
4.1 Deskripsi Dataset
4.2 Preprocessing
4.3 Pembentukan Label
4.4 Feature Engineering
4.5 Pembagian Data
4.6 Hyperparameter Tuning
4.7 Hasil Decision Tree
4.8 Hasil Random Forest
4.9 Perbandingan S1, S2, S3
4.10 Evaluasi Model
4.11 Confusion Matrix
4.12 TreeSHAP
4.13 Analisis Global
4.14 Analisis Lokal
4.15 Analisis Kelas
4.16 Pembahasan
```

Untuk setiap bagian berikan:

- data apa yang harus dimasukkan;
- tabel/gambar apa yang harus dimasukkan;
- sumber file;
- poin yang harus dijelaskan;
- kesimpulan yang boleh dibuat.

## 26. JANGAN MENGARANG

Ini adalah aturan terpenting.

Jika informasi tidak ditemukan:

```text
NOT FOUND
```

Jika informasi tidak bisa dipastikan:

```text
NEEDS VERIFICATION
```

Jika hasil berasal dari interpretasi:

```text
INTERPRETATION
```

Jika berasal langsung dari source code:

```text
SOURCE-CODE EVIDENCE
```

Jika berasal langsung dari hasil eksperimen:

```text
EXPERIMENTAL RESULT
```

Bedakan dengan jelas antara:

```text
FACT
INTERPRETATION
ASSUMPTION
```

Jangan pernah mengarang nilai metric, parameter, angka confusion matrix, fitur, atau hasil SHAP.

## 27. FORMAT AKHIR

`ANALYSIS_GUIDE.md` harus profesional, sistematis, dan mudah dibaca.

Gunakan:

- Markdown heading
- tabel
- code block
- checklist
- warning
- PASS/WARNING/FAIL
- ringkasan temuan

Pada bagian paling awal buat:

```markdown
# Analysis Guide

## Executive Summary
```

Executive Summary harus menjawab:

1. Model terbaik saat ini.
2. Skenario terbaik.
3. CV F1 terbaik.
4. Test F1 terbaik.
5. Parameter penting.
6. Risiko metodologis utama.
7. Apakah hasil siap dianalisis di BAB IV.
8. Apa yang masih harus diverifikasi.

## 28. FINAL VERDICT

Pada bagian paling akhir buat:

```markdown
# Final Verdict
```

Gunakan format:

```text
Pipeline Status:
Model Status:
Data Leakage Status:
Cross-Validation Status:
Hyperparameter Status:
Feature Engineering Status:
Confusion Matrix Status:
SHAP Status:
Reproducibility Status:
BAB IV Readiness:
Research Readiness:
```

Kemudian:

```markdown
## Immediate Actions
```

Berikan prioritas:

```text
P0 — wajib diperbaiki
P1 — sangat disarankan
P2 — penyempurnaan
```

## 29. SANGAT PENTING

Jangan memodifikasi source code penelitian.

Tugas Anda pada tahap ini hanya:

1. membaca;
2. mengaudit;
3. menganalisis;
4. mendokumentasikan;
5. memberikan rekomendasi.

Jangan mengubah hasil eksperimen.

Jangan menjalankan eksperimen tambahan yang mengubah output project kecuali benar-benar diperlukan untuk memverifikasi sesuatu; jika perlu menjalankan kode, jelaskan apa yang dijalankan dan jangan mengubah source code.

Output akhir yang wajib dibuat:

```text
ANALYSIS_GUIDE.md
```

Pastikan file tersebut benar-benar dibuat di root directory project.
