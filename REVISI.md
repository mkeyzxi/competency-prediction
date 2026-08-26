# Prompt Revisi Feature Engineering PRD dan Pipeline ML

Saya memiliki PRD penelitian **“Sistem Prediksi Kompetensi Mahasiswa Praktikum Logika Pemrograman”** yang menggunakan Decision Tree, Random Forest, dan TreeSHAP.

Saya ingin Anda **merevisi PRD, preprocessing, feature engineering, konfigurasi, pseudocode, dan pipeline eksperimen** agar struktur fitur yang digunakan oleh seluruh kelas A, B, C, D, dan E menjadi konsisten.

## Tujuan Utama

Jangan mengubah bobot penilaian akademik asli setiap kelas.

Yang ingin diseragamkan adalah **representasi feature space untuk machine learning**.

### Skema A/C

A dan C memiliki:

- Kehadiran = 10%
- TP = 15%
- Respons = 15%
- Laporan = 35%
- Final = 25%

Nilai mentah:

```text
TP → TP_Mean
Respons → Respons_Mean
Laporan → Laporan_Mean
```

### Skema B/D/E

B, D, dan E memiliki:

- TP + Respons = 20%
- Laporan = 30%
- Keaktifan = 10%
- Final = 30%
- Kehadiran = 10%

Pada B/D/E, apabila sumber data hanya menyediakan **satu skor gabungan TP + Respons**, jangan membagi skor tersebut.

Contoh:

```text
Skor TP + Respons = 85
```

harus dipetakan menjadi:

```text
TP_Mean = 85
Respons_Mean = 85
```

Bukan:

```text
TP_Mean = 42.5
Respons_Mean = 42.5
```

Dengan demikian, seluruh kelas menggunakan feature space utama yang sama:

```text
Attendance_Rate
TP_Mean
Respons_Mean
Laporan_Mean
```

## Aturan Dual Feature Mapping

Implementasikan aturan berikut:

```python
if scoring_scheme == "AC":
    TP_Mean = nilai_tp
    Respons_Mean = nilai_respons

elif scoring_scheme == "BDE":
    combined = nilai_tp_respons
    TP_Mean = combined
    Respons_Mean = combined
```

Jangan melakukan pembagian:

```python
combined / 2
```

Jangan melakukan normalisasi yang menyebabkan skor 85 berubah menjadi 42.5.

Nilai mentah tetap dipertahankan pada skala aslinya.

## Penting: Jangan Mengubah Bobot Akademik

Feature duplication ini hanya untuk **menyamakan feature representation**, bukan untuk menyatakan bahwa BDE benar-benar mempunyai dua pengukuran independen.

Dokumentasikan secara eksplisit:

> Pada skema BDE, ketika komponen TP dan Respons hanya tersedia sebagai satu skor gabungan, skor gabungan tersebut dipetakan ke `TP_Mean` dan `Respons_Mean` dengan nilai yang sama untuk menjaga konsistensi feature space lintas kelompok. Nilai tersebut tidak dibagi menjadi dua karena pembagian akan menciptakan nilai yang tidak merepresentasikan observasi asli.

Catat pula bahwa:

```text
TP_Mean == Respons_Mean
```

untuk observasi BDE yang berasal dari skor gabungan.

Jangan mengklaim bahwa kedua fitur tersebut merupakan dua pengukuran independen.

## Feature Space Utama

Model utama harus menggunakan struktur fitur yang sama untuk seluruh kelas:

```text
S1 =
[
    Attendance_Rate,
    TP_Mean,
    Respons_Mean,
    Laporan_Mean
]
```

Jangan membuat model utama dengan:

```text
AC → TP + Respons
BDE → TP_Respons
```

Gunakan feature space yang sama.

## Feature Engineering S2

S2 harus mempertahankan feature space S1 kemudian menambahkan fitur behavioral yang benar-benar dapat dihitung dari data:

```text
S2 =
S1 +
[
    TP_Completion_Rate,
    Respons_Completion_Rate,
    Laporan_Completion_Rate
]
```

Namun jangan menciptakan completion rate dari data yang tidak dapat mendukung definisi tersebut.

Untuk BDE dengan TP+Respons gabungan:

```text
TP_Completion_Rate
Respons_Completion_Rate
```

boleh memiliki nilai yang sama hanya apabila status aktivitas yang mendasarinya memang sama/tersedia untuk komponen gabungan tersebut.

Jangan mengarang data.

## Feature Engineering S3

Pertahankan S2 dan tambahkan relational feature hanya jika definisinya valid.

Karena pada BDE:

```text
TP_Mean == Respons_Mean
```

maka:

```text
Respons_TP_Gap = TP_Mean - Respons_Mean
```

akan selalu 0 untuk observasi tersebut.

Jangan memaksakan interpretasi bahwa gap tersebut bermakna pada BDE.

Oleh karena itu, evaluasikan apakah `Respons_TP_Gap` memang layak dipertahankan pada S3. Bila tidak memberikan informasi atau menyebabkan feature redundancy, dokumentasikan alasannya dan bandingkan S3 dengan dan tanpa fitur tersebut sebagai eksperimen sensitivitas.

## Keaktifan

`Keaktifan` hanya tersedia pada BDE.

Jangan membuat:

```text
Keaktifan = 0
```

untuk A/C.

Keaktifan tidak boleh dimasukkan ke model utama lintas semua kelas apabila fitur tersebut tidak tersedia secara valid untuk seluruh kelas.

Boleh dibuat eksperimen tambahan khusus robustness/sensitivity pada BDE.

## Tujuan Evaluasi

Jangan mengasumsikan bahwa perubahan feature engineering pasti meningkatkan akurasi.

Bandingkan secara empiris:

```text
Baseline lama
vs
Feature Space baru
```

Gunakan prosedur evaluasi yang konsisten:

```text
80% Train
20% Test
stratify = Competency_Label
random_state = 42
```

Pada training:

```text
5-fold Stratified Cross Validation
```

Jangan menggunakan test set untuk:

- feature selection
- hyperparameter tuning
- threshold tuning
- pemilihan model
- pemilihan fitur berdasarkan hasil test

## Eksperimen yang Harus Dibandingkan

Buat tabel eksperimen:

```text
E1 = Old Feature Mapping + Decision Tree
E2 = Old Feature Mapping + Random Forest

E3 = New Consistent Feature Mapping + Decision Tree
E4 = New Consistent Feature Mapping + Random Forest

E5 = New Consistent Feature Mapping + S2 + Decision Tree
E6 = New Consistent Feature Mapping + S2 + Random Forest

E7 = New Consistent Feature Mapping + S3 + Decision Tree
E8 = New Consistent Feature Mapping + S3 + Random Forest
```

Gunakan split dan seed yang konsisten agar perbandingan adil.

## Evaluasi yang Wajib

Jangan hanya mengejar Accuracy.

Laporkan:

```text
Accuracy
Precision
Recall
F1-score
Macro F1
Confusion Matrix
CV Mean
CV Standard Deviation
```

Karena penelitian bertujuan mendukung early warning terhadap mahasiswa yang **Belum Kompeten**, perhatikan terutama:

```text
Recall kelas Belum Kompeten
F1 kelas Belum Kompeten
```

Namun jangan memilih model hanya berdasarkan hasil yang paling tinggi pada test set.

Gunakan hasil CV training untuk model selection dan gunakan test set hanya sebagai evaluasi final.

## Audit Data

Tambahkan pemeriksaan:

```python
assert feature_columns == [
    "Attendance_Rate",
    "TP_Mean",
    "Respons_Mean",
    "Laporan_Mean"
]
```

untuk model utama.

Tambahkan audit bahwa:

```text
Tidak ada Final
Tidak ada NILAI_AKHIR
Tidak ada Predikat
Tidak ada Flowchart
Tidak ada Kodingan
Tidak ada Final_Kelompok
Tidak ada NIM
Tidak ada Nama
```

dalam X.

Pastikan tidak terjadi leakage.

## Audit Khusus BDE

Buat kolom:

```text
TP_Response_Source
```

dengan nilai misalnya:

```text
SEPARATE
COMBINED_DUPLICATED
```

Untuk BDE dengan skor gabungan:

```text
TP_Response_Source = COMBINED_DUPLICATED
```

Tambahkan juga pengecekan:

```python
if TP_Response_Source == "COMBINED_DUPLICATED":
    assert TP_Mean == Respons_Mean
```

Tujuannya agar proses preprocessing dapat diaudit.

## Output yang Saya Inginkan

Revisi seluruh bagian PRD yang terdampak, terutama:

1. Struktur dua skema penilaian
2. Feature Engineering
3. S1
4. S2
5. S3
6. Desain eksperimen
7. Pseudocode preprocessing
8. `feature_config.yaml`
9. Acceptance Criteria
10. Risiko metodologis
11. Penjelasan metodologi untuk skripsi/paper

Jangan menghapus informasi penting dari PRD lama yang masih valid.

Jangan mengubah target:

```text
Final_Individu >= 75 → Kompeten
Final_Individu < 75 → Belum Kompeten
```

Jangan mengubah aturan attendance yang sudah ditetapkan.

Jangan mengklaim bahwa feature duplication otomatis meningkatkan akurasi.

Sebaliknya, tuliskan bahwa hipotesis penelitian adalah:

> Konsistensi representasi fitur lintas skema penilaian dapat menghasilkan feature space yang lebih seragam sehingga memungkinkan model belajar pola kompetensi lintas kelas secara lebih konsisten.

Kemudian buktikan hipotesis tersebut melalui perbandingan eksperimen.

## Prinsip Terakhir

Prioritaskan:

```text
KEBENARAN DATA
>
KONSISTENSI FEATURE SPACE
>
PENCEGAHAN LEAKAGE
>
VALIDITAS EKSPERIMEN
>
STABILITAS MODEL
>
BARU AKURASI
```

Jangan melakukan manipulasi data hanya untuk meningkatkan Accuracy.

Semua perubahan harus dapat dijelaskan kepada reviewer penelitian.
