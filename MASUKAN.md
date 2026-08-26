# MASUKAN IMPLEMENTASI — Peningkatan Performa Model Prediksi Kompetensi Mahasiswa

## 0. Tujuan Revisi

Tujuan revisi ini adalah meningkatkan performa prediksi secara **sah secara metodologis**, terutama Accuracy, tanpa mengubah nilai akademik asli mahasiswa dan tanpa memasukkan informasi yang baru diketahui setelah Final sebagai fitur prediktor.

Penelitian tetap menggunakan:

- Target: `Competency_Label`
- `1 = Kompeten`
- `0 = Belum Kompeten`
- Skenario fitur: S1, S2, S3
- Dataset gabungan kelas A, B, C, D, E
- Format fitur dibuat konsisten lintas seluruh kelas

### Prinsip utama

> **Kejar performa model, bukan angka hasil rekayasa data.**

Nilai 0 yang terbukti sebagai nilai akademik sah harus dipertahankan. Perbaikan performa dilakukan melalui definisi populasi yang benar, pencegahan leakage, feature engineering yang valid, tuning, repeated cross-validation, baseline comparison, dan pemilihan model yang terukur.

---

# 1. Struktur Data Gabungan Kelas A–E

Mulai revisi ini, kelas A, B, C, D, dan E dianggap sebagai satu dataset analitik selama seluruh kelas telah memiliki struktur fitur yang sama.

Format fitur utama:

```text
Attendance_PreFinal_Rate
TP_Mean
Respons_Mean
Laporan_Mean
TP_Completion_Rate
Respons_Completion_Rate
Laporan_Completion_Rate
Respons_TP_Gap
Competency_Label
```

Metadata yang tetap disimpan tetapi **tidak masuk X pada eksperimen utama**:

```text
NIM
Nama
Class
Scoring_Scheme
Competency_Name
```

Jika tersedia metadata tambahan seperti semester, kelompok, atau sesi, metadata tersebut hanya boleh digunakan jika secara metodologis memang tersedia sebelum prediksi dan dinyatakan sebagai eksperimen tambahan.

---

# 2. DEFINISI TARGET

Gunakan definisi target yang konsisten:

```python
Competency_Label = (Final_Individu >= 75).astype(int)
```

Dengan:

```text
Final_Individu >= 75 → Kompeten (1)
Final_Individu < 75  → Belum Kompeten (0)
```

`Final_Individu` hanya digunakan untuk membuat target dan **tidak boleh berada di X**.

Kolom berikut dilarang menjadi predictor:

```text
Final_Individu
Final_UAS
Final_Kelompok
Nilai_Flowchart
Nilai_Kodingan
NILAI_AKHIR
Predikat
NIM
Nama
Competency_Name
kolom yang dihitung dari Final
kolom yang baru tersedia setelah Final
```

---

# 3. PERUBAHAN PALING PENTING: AUDIT DATA LEAKAGE

Sebelum tuning model, lakukan audit leakage secara eksplisit.

## 3.1 Attendance_PreFinal_Rate

Nama fitur harus benar-benar berarti **kehadiran sebelum Final**.

Jangan membentuk `Attendance_PreFinal_Rate` menggunakan bukti kehadiran yang berasal dari kegiatan Final atau informasi yang baru tersedia saat Final.

Contoh yang tidak boleh:

```text
Final tersedia → dianggap hadir → masuk Attendance_PreFinal_Rate
```

Contoh yang benar:

```text
Attendance_PreFinal_Rate
= kehadiran sesi yang memang terjadi sebelum Final
```

Jika sumber lama memakai sesi Final untuk merekonstruksi kehadiran, pisahkan fitur tersebut dari fitur pre-final atau jangan masukkan ke eksperimen prediksi pre-final.

### ATURAN

> Semua predictor harus dapat diketahui **sebelum mahasiswa menjalani Final Individu**.

Ini lebih penting daripada sekadar menaikkan Accuracy.

---

# 4. VALIDITAS NILAI 0

Nilai 0 tetap dipertahankan apabila 0 memang merupakan nilai akademik sah.

Contoh:

```text
Tidak mengumpulkan TP → 0
Tidak mengumpulkan laporan → 0
Tidak mengikuti respons → 0
```

Tetap:

```python
0 -> 0
```

Jangan melakukan:

```text
0 → median
0 → mean
0 → nilai acak
0 → nilai minimum non-zero
```

Dan jangan menghapus mahasiswa hanya karena memiliki banyak nilai 0 kecuali ada **aturan eligibility akademik yang telah ditentukan dan dicatat**.

---

# 5. TIGA POPULASI UNTUK SENSITIVITY ANALYSIS

Pipeline harus benar-benar menghasilkan minimal tiga populasi analitik yang dapat dibandingkan.

## P0 — RAW VALID

Semua mahasiswa yang:

1. memiliki identitas valid,
2. memiliki target `Final_Individu` valid,
3. memiliki predictor valid,
4. tidak memiliki duplicate unresolved.

P0 tidak melakukan exclusion berdasarkan Early Exit atau Attendance Ineligible.

## P1 — ELIGIBLE

```python
P1 = P0[P0["Early_Exit_Flag"] == False]
```

## P2 — STRICT ELIGIBLE

```python
P2 = P0[P0["Attendance_Ineligible_Flag"] == False]
```

Catatan:

`Attendance_Ineligible_Flag` biasanya merupakan subset dari Early Exit. Jangan menjumlahkan jumlah eksklusi P1 dan P2 seolah-olah keduanya independen.

Semua exclusion harus dicatat dalam audit log.

---

# 6. AUDIT POPULASI 158 → DATA ANALITIK

Buat tabel otomatis:

```text
Raw students
Valid Final
Duplicate unresolved
Missing Final
Early Exit
Attendance Ineligible
P0 Raw-Valid
P1 Eligible
P2 Strict-Eligible
```

Pipeline wajib menyimpan:

```text
data/processed/population_audit.csv
data/processed/excluded.csv
```

Dengan minimal kolom:

```text
NIM
Class
reason_code
Early_Exit_Flag
Attendance_Ineligible_Flag
Final_Valid
Included_P0
Included_P1
Included_P2
```

Tidak boleh ada penghapusan baris yang tidak dapat dijelaskan.

---

# 7. FEATURE ENGINEERING BARU

## 7.1 S1 — BASIC

```text
S1 = [
    Attendance_PreFinal_Rate,
    TP_Mean,
    Respons_Mean,
    Laporan_Mean
]
```

S1 adalah baseline feature space yang sederhana dan mudah diinterpretasikan.

---

## 7.2 S2 — BEHAVIORAL

```text
S2 = S1 + [
    TP_Completion_Rate,
    Respons_Completion_Rate,
    Laporan_Completion_Rate
]
```

Completion rate dihitung dari status pengerjaan yang benar-benar tersedia.

Contoh:

```text
TP_Completion_Rate = jumlah TP yang benar-benar dikerjakan / jumlah TP yang seharusnya dikerjakan
```

Jangan menganggap nilai rendah sebagai tidak mengumpulkan jika sumber tidak mendukung interpretasi tersebut.

---

## 7.3 S3 — RELATIONAL

```text
S3 = S2 + [Respons_TP_Gap]
```

```python
Respons_TP_Gap = Respons_Mean - TP_Mean
```

Tujuan fitur adalah menangkap perbedaan antara performa tugas pendahuluan dan evaluasi respons.

### CATATAN PENTING UNTUK DATA GABUNGAN A–E

Karena sekarang seluruh kelas menggunakan format yang sama, tetap simpan metadata `Scoring_Scheme` untuk audit.

Jika pada data tertentu TP dan Respons sebenarnya berasal dari satu skor gabungan yang sama, jangan mengklaim kedua fitur tersebut sebagai dua pengukuran independen.

Tambahkan:

```text
TP_Response_Source
```

dengan contoh:

```text
SEPARATE
COMBINED
```

Jika `COMBINED`, dokumentasikan bahwa `TP_Mean == Respons_Mean` dapat terjadi secara struktural.

---

# 8. FEATURE TAMBAHAN YANG BOLEH DIUJI

Tujuannya bukan memperbanyak fitur sebanyak mungkin, tetapi mencari fitur yang memiliki makna prediktif dan tersedia sebelum Final.

Uji sebagai eksperimen tambahan terpisah:

## 8.1 Weighted Activity Score

Jika seluruh komponen sudah dinormalisasi ke skala yang sama, buat:

```python
Activity_Score = (
    0.25 * TP_Mean +
    0.25 * Respons_Mean +
    0.25 * Laporan_Mean +
    0.25 * Attendance_PreFinal_Rate * 100
)
```

**JANGAN langsung memakai bobot akademik sebagai kebenaran penelitian** jika bobot tersebut berbeda antar kelas. Bobot ini hanya boleh diuji sebagai feature engineering tambahan dan harus dibandingkan dengan S1/S2/S3.

Alternatif yang lebih aman adalah membuat fitur tanpa bobot tetap:

```text
Activity_Mean
Activity_Median
Activity_Min
Activity_Max
```

tetapi hanya apabila interpretasinya jelas dan tidak menyebabkan redundancy berlebihan.

---

# 9. Fitur TREND / TEMPORAL: PRIORITASKAN INI

Jika nilai TP, Respons, atau Laporan tersedia per pertemuan secara berurutan, tambahkan eksperimen temporal.

Contoh:

```text
TP_First_Half_Mean
TP_Second_Half_Mean
TP_Trend
Respons_First_Half_Mean
Respons_Second_Half_Mean
Respons_Trend
Laporan_First_Half_Mean
Laporan_Second_Half_Mean
```

Contoh sederhana:

```python
TP_Trend = TP_Second_Half_Mean - TP_First_Half_Mean
```

Alasan:

Rata-rata akhir yang sama dapat berasal dari dua pola berbeda:

```text
Mahasiswa A: 90, 90, 90, 60, 60, 60
Mahasiswa B: 60, 60, 60, 90, 90, 90
```

Mean dapat sama, tetapi pola perkembangan kompetensinya berbeda.

**Fitur tren berpotensi lebih informatif daripada sekadar menambahkan banyak statistik acak.**

---

# 10. MODEL: JANGAN HANYA DT DAN RF

Tetap gunakan Decision Tree dan Random Forest sesuai desain awal, tetapi tambahkan baseline/model pembanding yang sederhana.

Minimal:

```text
DummyClassifier
Logistic Regression
Decision Tree
Random Forest
```

Jika library tersedia dan pembimbing menyetujui, boleh ditambahkan:

```text
HistGradientBoostingClassifier
XGBoost
LightGBM
CatBoost
```

Tetapi jangan memasukkan terlalu banyak model hanya demi mencari angka tertinggi.

Untuk artikel ilmiah, model harus dipilih berdasarkan prosedur yang ditentukan sebelum melihat test set.

---

# 11. WAJIB: DUMMY BASELINE

Tambahkan:

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(
    strategy="most_frequent",
    random_state=42
)
```

Bandingkan semua model dengan baseline mayoritas.

Jangan menggunakan 50% sebagai baseline otomatis.

Baseline aktual bergantung pada distribusi `Competency_Label`.

---

# 12. CLASS DISTRIBUTION AUDIT

Sebelum training, keluarkan:

```text
Total
Kompeten count
Belum Kompeten count
Kompeten %
Belum Kompeten %
```

Per kelas juga:

```text
Class × Competency_Label
```

Dan:

```text
Scoring_Scheme × Competency_Label
```

Tujuan: mengetahui apakah Accuracy tinggi/rendah karena imbalance.

---

# 13. METRIK YANG WAJIB

Gunakan minimal:

```text
Accuracy
Balanced Accuracy
Precision Macro
Recall Macro
F1 Macro
ROC-AUC
Confusion Matrix
```

Untuk target early-warning, tampilkan juga Recall kelas `Belum Kompeten` secara khusus.

Accuracy tetap menjadi metrik penting karena menjadi salah satu tujuan evaluasi penelitian, tetapi **tidak boleh menjadi satu-satunya metrik**.

---

# 14. REPEATED STRATIFIED CROSS-VALIDATION

Karena jumlah data relatif kecil, satu kali train/test split dapat menghasilkan variance tinggi.

Gunakan:

```python
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)
```

Ini menghasilkan 50 fold evaluation.

Laporkan:

```text
Mean Accuracy
Std Accuracy
Mean F1 Macro
Std F1 Macro
Mean Balanced Accuracy
Std Balanced Accuracy
```

Untuk hasil final, test set tetap hanya digunakan setelah model dan hyperparameter dikunci.

---

# 15. NESTED / PIPELINE-SAFE TUNING

Semua proses yang belajar dari training data harus dilakukan di dalam pipeline/CV.

Jangan melakukan tuning pada test set.

Contoh struktur:

```text
TRAIN
  ↓
Feature preprocessing
  ↓
Hyperparameter search
  ↓
CV
  ↓
Model locked
  ↓
TEST sekali
```

Jika memakai scaler/selection/imputation, letakkan di `Pipeline`.

---

# 16. HYPERPARAMETER TUNING RANDOM FOREST

Jangan hanya memakai default Random Forest.

Gunakan search space terbatas:

```python
param_grid = {
    "n_estimators": [200, 500, 800],
    "max_depth": [None, 3, 5, 8, 12],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced", "balanced_subsample"]
}
```

Jangan menganggap `class_weight="balanced"` pasti lebih baik.

Biarkan CV membuktikannya.

---

# 17. HYPERPARAMETER TUNING DECISION TREE

```python
param_grid = {
    "max_depth": [2, 3, 4, 5, 7, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
    "class_weight": [None, "balanced"]
}
```

---

# 18. THRESHOLD TUNING

Threshold default biasanya 0.50.

Untuk eksperimen tambahan, uji threshold:

```text
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
```

Tetapi threshold harus dipilih berdasarkan validation/CV data, bukan test set.

Karena penelitianmu berhubungan dengan early warning, kamu dapat mencari threshold yang meningkatkan Recall `Belum Kompeten` dengan tetap mempertahankan Accuracy yang memadai.

### ATURAN

Jangan memilih threshold berdasarkan test set.

---

# 19. FEATURE SELECTION

Jangan menambah fitur hanya untuk menaikkan angka.

Uji:

```text
S1
S2
S3
S4 temporal
S4 + selected features
```

Gunakan CV untuk menentukan apakah fitur tambahan benar-benar membantu.

Jika sebuah fitur:

```text
menaikkan CV sedikit
namun menurunkan test dan meningkatkan variance
```

jangan otomatis memilih fitur tersebut.

---

# 20. CEK MULTICOLLINEARITY / REDUNDANCY

Karena `TP_Mean` dan `Respons_Mean` dapat identik pada beberapa sumber, periksa:

```python
corr = X.corr(numeric_only=True)
```

Simpan:

```text
results/reports/feature_correlation.csv
```

Korelasi tinggi tidak otomatis berarti fitur harus dibuang pada Random Forest, tetapi perlu dijelaskan dan diperiksa pengaruhnya terhadap interpretasi SHAP.

---

# 21. ANALISIS CONFUSION MATRIX

Untuk model terbaik, simpan:

```text
TN
FP
FN
TP
```

Kemudian identifikasi:

```text
False Positive
False Negative
```

Khusus penelitianmu, **False Negative pada Belum Kompeten** perlu mendapat perhatian karena mahasiswa berisiko dapat lolos dari deteksi model.

---

# 22. ANALISIS ERROR CASE

Buat file:

```text
results/reports/error_analysis.csv
```

Dengan minimal:

```text
NIM_Anon
Class
True_Label
Predicted_Label
Predicted_Probability
S1/S2/S3
Error_Type
```

Kemudian analisis apakah mahasiswa yang salah prediksi memiliki pola seperti:

```text
nilai tugas tinggi tetapi Final rendah
nilai tugas rendah tetapi Final tinggi
kehadiran rendah tetapi Final tinggi
kehadiran tinggi tetapi Final rendah
```

Ini dapat menjadi bahan pembahasan penelitian.

---

# 23. SHAP

SHAP dijalankan setelah model final dipilih.

Minimal:

```text
Global feature importance
Beeswarm
Bar plot
1 true positive
1 true negative
1 false positive
1 false negative
```

Interpretasi:

> SHAP menunjukkan kontribusi fitur terhadap prediksi model, bukan bukti bahwa fitur tersebut menyebabkan kompetensi.

---

# 24. PEMILIHAN MODEL FINAL

Jangan memilih model hanya karena Accuracy test paling tinggi.

Gunakan prosedur:

1. Model dibandingkan berdasarkan repeated CV.
2. Accuracy menjadi metrik penting.
3. Balanced Accuracy dan F1 Macro digunakan sebagai pemeriksaan keseimbangan performa.
4. Recall `Belum Kompeten` diperhatikan untuk tujuan early warning.
5. Test set hanya digunakan sebagai evaluasi akhir.
6. Jika selisih model kecil, pilih model yang lebih stabil dan lebih sederhana.

Contoh:

```text
RF A:
CV Accuracy = 0.68 ± 0.05
Test Accuracy = 0.67

RF B:
CV Accuracy = 0.72 ± 0.14
Test Accuracy = 0.72
```

Jangan otomatis memilih RF B hanya karena test lebih tinggi. Variance-nya jauh lebih besar.

---

# 25. TARGET PERFORMA

Gunakan target performa sebagai **tujuan eksperimen**, bukan angka yang harus dipaksakan.

Target eksplorasi:

```text
Accuracy > baseline
Balanced Accuracy > baseline
F1 Macro > baseline
CV std semakin kecil
```

Target internal yang dapat digunakan:

```text
Accuracy sekitar >= 0.65 → mulai informatif
Accuracy sekitar >= 0.70 → cukup kuat untuk dibahas secara serius
Accuracy >= 0.75 → sangat menarik apabila stabil pada CV dan test
```

Namun angka tersebut tidak boleh dicapai dengan manipulasi data.

---

# 26. JIKA AKURASI TETAP 0.55–0.65

Jangan memodifikasi data.

Lakukan diagnosis berikut:

```text
1. Bandingkan Dummy
2. Cek class imbalance
3. Cek leakage
4. Cek kualitas label
5. Cek temporal features
6. Cek tuning
7. Cek repeated CV
8. Cek confusion matrix
9. Cek error cases
10. Cek apakah target memang dapat diprediksi dari data pre-final
```

Jika setelah seluruh prosedur performa tetap rendah, hasil tersebut harus diperlakukan sebagai temuan empiris.

---

# 27. JANGAN MENGGUNAKAN TEKNIK BERIKUT

Tidak boleh dilakukan hanya untuk meningkatkan Accuracy:

```text
Mengubah nilai 0 menjadi nilai positif
Mengganti nilai rendah dengan median
Menghapus mahasiswa berdasarkan hasil prediksi model
Menghapus data yang salah diprediksi setelah melihat test result
Memilih random_state yang menghasilkan Accuracy terbaik tanpa dilaporkan
Memilih test split terbaik
Tuning hyperparameter menggunakan test set
Memilih threshold berdasarkan test set
Duplicating minority rows sebelum split semestinya
SMOTE antes de train/test split
Menggunakan Final-derived columns sebagai X
```

---

# 28. SMOTE / CLASS IMBALANCE

SMOTE boleh diuji hanya jika class imbalance memang terbukti menjadi masalah.

Jika digunakan:

```text
Split dahulu
↓
SMOTE hanya pada training fold
↓
CV
↓
Test tetap untouched
```

Jangan melakukan SMOTE pada seluruh dataset sebelum split karena dapat menyebabkan leakage.

Untuk tree models, bandingkan terlebih dahulu:

```text
class_weight=None
class_weight=balanced
```

sebelum menggunakan synthetic oversampling.

---

# 29. MODEL YANG DIANJURKAN UNTUK EKSPERIMEN

Urutan eksperimen:

```text
E0 Dummy Majority
E1 Logistic Regression
E2 Decision Tree S1
E3 Random Forest S1
E4 Decision Tree S2
E5 Random Forest S2
E6 Decision Tree S3
E7 Random Forest S3
E8 Tuned Random Forest
E9 Tuned Random Forest + Temporal Features
E10 Gradient Boosting Model (opsional)
```

Semua eksperimen harus menggunakan split/CV dan seed yang konsisten.

---

# 30. OUTPUT FILE YANG DIHARAPKAN

Tambahkan output berikut:

```text
data/processed/population_P0.csv
data/processed/population_P1.csv
data/processed/population_P2.csv
results/reports/population_audit.csv
results/reports/class_distribution.csv
results/reports/baseline_comparison.csv
results/reports/feature_correlation.csv
results/reports/error_analysis.csv
results/reports/repeated_cv_results.csv
results/reports/model_comparison.csv
results/reports/threshold_analysis.csv
```

---

# 31. STRUKTUR TABEL HASIL FINAL

Tabel utama penelitian harus kurang lebih seperti:

| Population | Scenario | Model | CV Accuracy Mean | CV Accuracy SD | Test Accuracy | Balanced Accuracy | F1 Macro | Recall Belum Kompeten |
| ---------- | -------- | ----- | ---------------: | -------------: | ------------: | ----------------: | -------: | --------------------: |
| P0         | S1       | Dummy |              ... |            ... |           ... |               ... |      ... |                   ... |
| P0         | S1       | DT    |              ... |            ... |           ... |               ... |      ... |                   ... |
| P0         | S1       | RF    |              ... |            ... |           ... |               ... |      ... |                   ... |
| P1         | S1       | RF    |              ... |            ... |           ... |               ... |      ... |                   ... |
| P2         | S1       | RF    |              ... |            ... |           ... |               ... |      ... |                   ... |
| P2         | S2       | RF    |              ... |            ... |           ... |               ... |      ... |                   ... |
| P2         | S3       | RF    |              ... |            ... |           ... |               ... |      ... |                   ... |

---

# 32. STRATEGI PENELITIAN YANG PALING CERDAS

Jangan memulai dari pertanyaan:

> "Bagaimana membuat Accuracy menjadi 80%?"

Mulai dari:

> "Apa informasi pre-final yang paling mampu membedakan mahasiswa Kompeten dan Belum Kompeten?"

Kemudian lakukan:

```text
Data mentah valid
        ↓
Audit label + imbalance
        ↓
Leakage audit
        ↓
P0/P1/P2
        ↓
Baseline Dummy
        ↓
S1
        ↓
S2
        ↓
S3
        ↓
Temporal features
        ↓
Tuning
        ↓
Repeated CV
        ↓
Threshold analysis
        ↓
Final test
        ↓
SHAP + error analysis
```

Bagian yang paling berpotensi meningkatkan prediksi secara substantif bukan mengubah nilai 0, tetapi **mengubah representasi data menjadi lebih informatif**.

Prioritas eksplorasi:

```text
1. Leakage audit
2. Class distribution + dummy baseline
3. Temporal features
4. Tuned Random Forest
5. Logistic Regression sebagai baseline kuat
6. Balanced accuracy / F1 macro
7. Threshold tuning
8. Gradient boosting sebagai model tambahan
9. Error analysis
10. SHAP
```

---

# 33. HAL YANG HARUS DIPERIKSA DARI SCRIPT SAAT INI

Sebelum menilai kembali Accuracy 0.52–0.60, audit kode berikut:

```text
scripts/validate_data.py
scripts/build_features.py
scripts/run_experiments.py
scripts/generate_shap.py
```

Periksa khusus:

### A. Apakah 122 mahasiswa benar-benar P0/P1/P2?

Saat ini output hanya menunjukkan:

```text
S1 = 122
S2 = 122
S3 = 122
```

Belum cukup untuk membuktikan tiga populasi sensitivity analysis.

### B. Apakah F1 yang dicetak benar-benar `macro`?

Pastikan kode menggunakan:

```python
average="macro"
```

jika laporan menyebut F1 Macro.

### C. Apakah split dilakukan sebelum preprocessing yang belajar dari data?

Tidak boleh ada transformasi/data balancing yang memakai seluruh dataset sebelum split.

### D. Apakah `Attendance_PreFinal_Rate` benar-benar pre-final?

Pastikan tidak ada informasi Final yang ikut membentuk fitur tersebut.

### E. Apakah target diseimbangkan hanya pada training?

Jika menggunakan class weighting atau SMOTE, penerapannya harus berada di training/CV.

---

# 34. PRIORITAS IMPLEMENTASI

Urutan pengerjaan yang disarankan:

## PRIORITAS 1 — WAJIB

```text
[ ] Class distribution audit
[ ] DummyClassifier
[ ] Balanced Accuracy
[ ] F1 Macro verification
[ ] Population P0/P1/P2
[ ] Leakage audit
[ ] Repeated Stratified CV
[ ] RF/DT tuning
```

## PRIORITAS 2 — SANGAT DISARANKAN

```text
[ ] Temporal features
[ ] Error analysis
[ ] Threshold tuning
[ ] Feature correlation
[ ] Logistic Regression
```

## PRIORITAS 3 — EKSPERIMEN TAMBAHAN

```text
[ ] Gradient Boosting
[ ] SMOTE jika imbalance memang terbukti
[ ] Additional derived features
```

---

# 35. KRITERIA KEBERHASILAN REVISI

Revisi dianggap berhasil apabila:

```text
1. Tidak ada perubahan manual pada nilai akademik asli.
2. Semua nilai 0 yang valid tetap dipertahankan.
3. Populasi P0/P1/P2 dapat direproduksi.
4. Dummy baseline tersedia.
5. Accuracy dibandingkan dengan baseline.
6. Balanced Accuracy dan F1 Macro tersedia.
7. Repeated CV tersedia.
8. Hyperparameter tuning terdokumentasi.
9. Tidak ada post-final leakage.
10. Test set tidak dipakai untuk tuning.
11. Model final dipilih berdasarkan prosedur yang konsisten.
12. Error analysis dan SHAP tersedia.
```

---

# 36. PESAN UTAMA UNTUK PENELITIAN

Penelitian tetap boleh dan perlu mengejar Accuracy yang lebih tinggi.

Namun sumber peningkatan performa harus berasal dari:

```text
informasi yang lebih baik
+ preprocessing yang benar
+ feature engineering yang bermakna
+ model yang lebih sesuai
+ tuning yang benar
+ validasi yang lebih stabil
```

bukan dari:

```text
mengubah nilai mahasiswa
+ menghapus kasus buruk setelah melihat hasil
+ memilih test split terbaik
+ memasukkan informasi Final ke predictor
```

Dengan pendekatan ini, jika Accuracy naik menjadi 0.70 atau 0.75, kenaikan tersebut dapat dipertanggungjawabkan. Jika tetap sekitar 0.60, penelitian juga memiliki diagnosis yang jelas mengenai batas kemampuan fitur pre-final.

---

# 37. CATATAN KHUSUS UNTUK DATA A–E YANG SEKARANG SERAGAM

Karena user menyatakan kelas A, B, C, D, dan E sekarang sudah disatukan dan memiliki fitur serta format yang sama, pipeline baru harus **menghindari percabangan feature engineering berdasarkan kelas** selama memang sudah tidak ada perbedaan semantik.

Jadikan `Class` hanya sebagai metadata/context terlebih dahulu.

Eksperimen utama:

```text
ALL_CLASSES
 ├── S1
 ├── S2
 └── S3
```

Setelah model utama selesai, lakukan robustness analysis:

```text
ALL
AC subset
BDE subset
Class A
Class B
Class C
Class D
Class E
```

Subset class-level jangan digunakan sebagai lima model utama kecuali ukuran sampel tiap kelas memang memadai dan desain penelitian menetapkannya.

Tujuan analisis tersebut adalah mengetahui heterogenitas, bukan memaksa setiap kelas menghasilkan Accuracy tinggi.

---

# 38. HASIL YANG DIHARAPKAN DARI REVISI

Output final diharapkan dapat menjawab empat pertanyaan:

1. Apakah model benar-benar lebih baik daripada baseline mayoritas?
2. Apakah S2/S3 meningkatkan performa dibanding S1?
3. Apakah tuning dan temporal features meningkatkan Accuracy secara stabil pada CV?
4. Apakah model tetap memiliki performa yang dapat diterima pada test set yang tidak pernah digunakan untuk tuning?

Jika jawabannya ya, maka penelitian memiliki dasar yang jauh lebih kuat untuk membahas peningkatan performa.

Jika tidak, jangan memodifikasi data. Gunakan hasil tersebut untuk menjelaskan keterbatasan prediktor pre-final dan pola kesalahan model.
