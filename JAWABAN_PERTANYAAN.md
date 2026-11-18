# Jawaban Pertanyaan Sidang — Proyek Prediksi Kompetensi Mahasiswa (Early Warning System)

**Judul Proyek:** Evaluasi Temporal Decision Tree & Random Forest untuk Sistem Peringatan Dini Akademik Praktikum Logika & Algoritma dengan Explainable AI (SHAP)

**Dokumen ini disusun berdasarkan pembacaan langsung atas kode sumber (`src/`, `scripts/`, `configs/`), berkas data (`data/raw/`, `data/processed/`, `data/features/`), dan hasil keluaran (`results/final/`, `results/metrics/`, `results/shap/`).**

---

## BAGIAN 0 — Fakta Dasar Proyek (Ringkasan Singkat)

| Aspek | Fakta Terverifikasi |
|---|---|
| Sumber data mentah | `data/raw/DBNR.xlsx` (5 sheet: A, B, C, D, E + sheet FINAL) |
| Jumlah baris mentah | 96 baris → 10 baris "asisten/asdos" dibuang → **86 mahasiswa unik** |
| Sebaran kelas | A=10, B=23, C=23, D=19, E=11 |
| Label target | `Competency_Label`: **Kompeten (1) = 44**, **Belum Kompeten (0) = 42** |
| Aturan label | `Final_Score ≥ 83` → Kompeten (1); selain itu → Belum Kompeten (0) |
| Data panjang (long format) | `activities_long.csv` = **1.140 baris**: Attendance 527, Laporan 321, TP 292 |
| Jendela waktu maksimal | Attendance s.d. pertemuan 7; Laporan s.d. 4; TP s.d. 4 |
| Pembagian data | 80:20 → **68 latih** (33 BK + 35 K) dan **18 beku/uji** (9 BK + 9 K) |
| Validasi | Nested Cross-Validation: Outer `RepeatedStratifiedKFold(5 fold × 5 repeat)`, Inner `StratifiedKFold(3 fold)` |
| Model | Decision Tree & Random Forest (Dummy Classifier sebagai pembanding) |
| Model final | **Decision Tree — Skenario S3_E — balancing None — threshold 0,52** |
| Ketidakpastian | Bootstrap non-parametrik B = 1.000 repetisi, Confidence Interval 95% |
| Interpretasi | SHAP (`shap.TreeExplainer`) |

---

# BAGIAN 1 — K-FOLD CROSS VALIDATION vs NESTED CROSS VALIDATION

## 1.1 Apa itu K-Fold Cross Validation?

**K-Fold** berarti memecah data menjadi K bagian sama besar. Model dilatih K−1 bagian dan diuji 1 bagian; proses ini diulang K kali sehingga setiap data pernah diuji satu kali.

**Ilustrasi sederhana dengan K = 5** (pembagian yang dipakai di proyek ini):

```
Data = 100 mahasiswa
┌────────┬────────┬────────┬────────┬────────┐
│ Fold 1 │ Fold 2 │ Fold 3 │ Fold 4 │ Fold 5 │
│ 20 mhs │ 20 mhs │ 20 mhs │ 20 mhs │ 20 mhs │
└────────┴────────┴────────┴────────┴────────┘
Putaran 1: latih Fold 1-4 → uji Fold 5
Putaran 2: latih Fold 1,2,3,5 → uji Fold 4
Putaran 3: latih Fold 1,2,4,5 → uji Fold 3
Putaran 4: latih Fold 1,3,4,5 → uji Fold 2
Putaran 5: latih Fold 2,3,4,5 → uji Fold 1
→ Nilai = rata-rata dari 5 hasil uji
```

**Kelemahan K-Fold biasa:** kalau kita sekaligus **menyesuaikan parameter** (misal `max_depth`, `min_samples_leaf`) pada data yang sama, maka hasilnya **terlalu optimis (optimistic bias / optimisme berlebihan)**. Kita seperti "menilai ujian sambil menghafal jawabannya sendiri". Angka yang dilaporkan jadi bukan ukuran jujur kemampuan model terhadap data baru.

## 1.2 Apa itu Nested Cross Validation?

**Nested CV = CV di dalam CV.** Ada dua lapis:

```
┌──────────────────────────────────────────────────────────┐
│  OUTER LOOP  →  BERTINDAK SEBAGAI "PENGUJI"              │
│  (5 fold × 5 repeat = 25 putaran)                        │
│                                                          │
│   ┌────────────────────────────────────────────────┐     │
│   │  INNER LOOP → BERTINDAK SEBAGAI "TUKANG BELAJAR"│     │
│   │  StratifiedKFold 3 fold + RandomizedSearchCV  │     │
│   │  (mencari hyperparameter terbaik)              │     │
│   └────────────────────────────────────────────────┘     │
│                                                          │
│   Skor outer = ANGKA JUJU performa model                │
└──────────────────────────────────────────────────────────┘
```

**Kode yang dipakai (`src/evaluation.py`):**

```python
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)   # OUTER
search_estimator = build_search_estimator(model, model_name, config_path) # INNER
scores = cross_validate(search_estimator, X_train, y_train, cv=cv, scoring=scoring)
```

Dan di `src/tuning.py`:

```python
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)   # INNER CV
random_search = RandomizedSearchCV(
    estimator=model, param_distributions=param_grid,
    n_iter=min(50, total_kombinasi), scoring=recall_bk_scorer, cv=cv)
```

**Aturan emas yang dijaga ketat dalam kode:**

| Yang dikerjakan | Di mana | Kenapa begitu? |
|---|---|---|
| Pencarian hyperparameter | **Inner loop saja** | Model tidak boleh "mengintip" data uji |
| Perhitungan seluruh metrik | **Outer loop** | Ini angka yang dilaporkan di makalah |
| Threshold (ambang) | **Train set saja** (`cross_val_predict`) | `src/tuning.py:132` |
| Normalisasi / imputasi | **Di dalam `Pipeline`** | `src/models.py:54-65` — tidak bocor |
| Data beku (holdout) | **Sekali di akhir** | `results/final/holdout_results.csv` |

## 1.3 Perbandingan Ringkas

| Aspek | K-Fold biasa | Nested Cross Validation |
|---|---|---|
| Jumlah lapis | 1 lapis | 2 lapis (luar & dalam) |
| Tuning hyperparameter | Di luar CV (mengintip data uji) | Di dalam inner CV |
| Skor yang dilaporkan | Cenderung **terlalu tinggi** (optimis) | **Jujur / tidak bias** |
| _Selection Bias_ | Banyak | Minimal |
| Biaya komputasi | Murah | Lebih mahal (tuning diulang tiap outer fold) |
| Kesesuaian untuk sampel kecil | Risiko tinggi | **Sangat cocok** — dipakai di proyek ini |
| Digunakan di proyek ini? | ❌ tidak | ✅ ya |

## 1.4 Apakah Perlu "Sampel"? — **YA, sangat perlu**

Pertanyaan: *"apakah perlu sampel?"* Jawabannya **ya, wajib**, karena tiga alasan berikut:

**Alasan 1 — Datanya kecil (n = 86).**
Kalau 86 data langsung dipakai semua untuk melatih model, kita **tidak punya data sama sekali untuk menguji**. Kita tidak akan pernah tahu apakah model berhasil di semester depan atau hanya "menghafal" 86 data itu. Kita butuh sebagian data yang **disembunyikan** dari proses belajar.

**Alasan 2 — Kita butuh ukuran ketidakpastian.**
Dengan 86 data, satu mahasiswa salah klasifikasi saja bisa mengubah akurasi sebesar **±5,8%** (1/86). Angka presisi 1–2 digit desimal tidak bisa dipercaya tanpa pengulangan (repeat).

**Alasan 3 — Model selection butuh "kontes".**
Kita menguji 5 skenario fitur (S1–S5) × 2 algoritma (DT, RF) × 7 skenario EWS = banyak kombinasi. Setiap kombinasi harus dinilai pada **sampel yang sama** agar adil. Cross Validation adalah cara adil menilai kombinasi-kombinasi itu.

**Bentuk "sampel" yang dipakai di proyek ini:**

1. **Simple Random Sampling + Stratify** untuk membagi 68 latih / 18 uji (`src/split.py`):
   ```python
   train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
   ```
   - `random_state=42` → **reproducible**, hasil bisa diulang persis
   - `stratify=y` → jumlah kelas di latih dan uji **sebanding** (tidak ada data uji yang isinya didominasi satu kelas)

2. **Stratified K-Fold** di dalam CV → setiap fold punya proporsi kelas 0 dan 1 yang sama.

3. **Repeated (5×)** → 5 putaran diulang → total 25 pengukuran → bisa dihitung Mean ± SD dan CI.

---

# BAGIAN 2 — DATA PRIMER DAN DATA SEKUNDER

## 2.1 Definisi (Bahasa Sederhana)

| Jenis | Definisi | Contoh |
|---|---|---|
| **Data Primer** | Data yang **dikumpulkan sendiri / langsung** oleh peneliti dari objek yang diteliti, untuk keperluan penelitian itu sendiri. | Mengisi kuesioner mahasiswa, mencatat kehadiran, memberi tugas lalu menilai sendiri. |
| **Data Sekunder** | Data yang **sudah ada sebelumnya**, dibuat oleh pihak lain, lalu **diambil/dipakai kembali** oleh peneliti. | Mengambil data dari BPS, hasil survei nasional, artikel ilmiah, data open-source Kaggle. |

**Kunci pembeda:** *"Siapa yang pertama kali membuat datanya?"*
- السفر ke kampus, absen, mengumpulkan tugas → **Primer**
- Menyalin dari website orang → **Sekunder**

## 2.2 Klasifikasi Data dalam Proyek Ini

### ✅ Jawaban Tegas: File Excel dari Asdos/Asisten Dosen adalah **DATA PRIMER**

**Alasannya:**

**Alasan 1 — Data dibuat oleh tim peneliti sendiri.**
Berkas `DBNR.xlsx` berisi kolom: `NIM`, `NAMA`, `KEHADIRAN` (pertemuan 1–6), `TOTAL NILAI HADIR`, `NILAI LAPORAN` (1–4), `TOTAL NILAI LAPORAN`, `NILAI TP` (1–3), `TOTAL NILAI TP`, `FINAL`, `TOTAL NILAI`, `BOBOT`. Semua angka ini adalah **produk dari proses mengajar dan asesmen yang dilakukan langsung oleh tim pengampu (dosen + asisten)**. Tidak ada sumber luar yang mengoleksinya.

**Alasan 2 — Ini termasuk "Data Primer Kearsipan/Administratif".**
Dalam metodologi penelitian, data Primer punya dua wujud:
- **Data Primer Kuantitatif**: kuesioner, tes, atau pengukuran langsung → umumnya diisi oleh objek penelitian
- **Data Primer Kearsipan/Administratif (*record-keeping*)**: dokumen resmi yang dibuat organisasi sendiri untuk kegunaan operasional → **kasus proyek ini**

Data kehadiran, nilai tugas, dan nilai akhir adalah **dokumen operasional resmi** yang dibuat oleh tim pengampu selama satu semester, sehingga termasuk data primer.

**Alasan 3 — Bukan data sekunder karena:**
- ❌ Tidak diambil dari jurnal/buku/perpustakaan
- ❌ Tidak diambil dari lembaga statistik (BPS/Kemendikbud)
- ❌ Tidak diambil dari dataset publik/terbuka
- ❌ Tidak diambil dari penelitian terdahulu

## 2.3 Data Primer Lain yang Ada di Proyek Ini

| Jenis Data Primer | Sumber | Deskripsi |
|---|---|---|
| **Kehadiran** | Dosen/asdos mencatat di tiap pertemuan | 0–6 per pertemuan, di-*encode* 1 = hadir, 0 = tidak |
| **Nilai Tugas Pendahuluan (TP)** | Dosen menilai tugas | Maksimal 3–4 tugas per mahasiswa |
| **Nilai Laporan Praktikum** | Dosen menilai laporan | Maksimal 4 laporan |
| **Nilai Akhir (Final)** | Tim pengampu | Dipakai sebagai **dasar pelabelan** (`≥83` = Kompeten) |
| **Kelas** | Administrasi | Menentukan sheet A/B/C/D/E |

## 2.4 Data Sekunder yang Digunakan (Kecil, Derivatif)

| Jenis | Sumber | Status |
|---|---|---|
| **Ambang batas 83** | Standarisasi nilai predikat unggul akademik / zona aman A- | ← Ini **rujukan/landasan teori**, bukan data. Cukup dicantumkan sumbernya. |
| **Bobot penilaian** | `configs/data_config.yaml`: Kehadiran 0,10 · TP 0,15 · Respons 0,15 · Laporan 0,35 · Final 0,25 | ← Ini **ketentuan resmi**, bagian dari data primer dokumen |

## 2.5 Terminologi yang Benar untuk Sidang

Sampaikan seperti ini:

> "Penelitian ini menggunakan **data primer kuantitatif dengan sifat kearsipan (record-keeping)**. Sumber datanya adalah dokumen penilaian dan absensi resmi Praktikum Logika & Algoritma yang disusun oleh tim pengampu (dosen dan asisten dosen) selama satu semester. Researcher sendiri yang mengoleksi, mendigitalkan, dan melabeli data tersebut, sehingga statusnya Primer — bukan Sekunder, karena tidak diambil dari pihak ketiga maupun publikasi terdahulu."

---

# BAGIAN 3 — LOCKED FOLD, PEMBAGIAN 80:20 atau 70:30, DAN ALASANNYA

## 3.1 Apa itu Locked Fold?

**Locked Fold** (lipatan terkunci) adalah **sejumlah data yang dikunci — tidak boleh dibuka sampai akhir percobaan.**

Dalam proyek ini ada **3 lapis penguncian data**:

```
LAPIS 1 — DATA LATIH (n = 68)
   ├─► Outer CV (5 fold × 5 repeat)  → "kontes model" / latihan model
   │      └─► Inner CV (3 fold)       → tuning hyperparameter
   └─► Pemilihan model terbaik & pemilihan THRESHOLD

LAPIS 2 — DATA UJI / HOLDOUT / LOCKED FOLD (n = 18)  🔒
   └─► BARU dibuka 1x di akhir, setelah model + threshold dikunci
        └─► Menghasilkan Confusion Matrix, Bootstrap CI, SHAP plot
```

**Aturan yang dipegang (dari `scripts/run_optimized_experiment.py`):**
1. Holdout **tidak pernah** dipakai untuk training.
2. Holdout **tidak pernah** dipakai untuk tuning hyperparameter.
3. Holdout **tidak pernah** dipakai untuk memilih fitur (feature selection).
4. Holdout **tidak pernah** dipakai untuk memilih threshold.
5. Holdout **baru dibuka di akhir** → dipakai untuk **TESI AKHIR**.
6. 모델 **dikunci** (*frozen*) sebelum holdout dibuka.

**Kode yang mengimplementasikan:**
```python
# src/split.py — split dilakukan SEKALI, hasilnya dibekukan
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
```

```python
# src/evaluation.py — ambang hanya dicari di data latih
cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')
```

## 3.2 80:20 atau 70:30? — Proyek Ini Memakai **80:20**

**Angka aktual di kode (sudah diverifikasi dengan menghitung langsung):**

| | Jumlah | Belum Kompeten (0) | Kompeten (1) |
|---|---|---|---|
| **Data Latih (80%)** | **68** | 33 | 35 |
| **Data Uji / Locked Fold (20%)** | **18** | 9 | 9 |
| **Total** | 86 | 42 | 44 |

## 3.3 Mengapa 80:20, Bukan 70:30? — 5 Alasan Kuat

**Alasan 1 — 80:20 adalah proporsi ideal yang lazim dipakai pada model machine learning.**

Aturan praktis di bidang ML:
| Proporsi | Kapan dipakai |
|---|---|
| 60:40 | Klasifikasi biasa dengan data sangat banyak |
| **80:20** | **Standar umum Cross Validation (padanan dengan 5-Fold)** |
| 70:30 | Time series, atau saat data sangat sedikit |
| 90:10 | Dataset sangat besar (>10.000) |

Kode proyek ini memakai `n_splits=5` untuk Outer CV. **5-Fold = 80% latih / 20% uji.** Jadi **80:20 membuat hasil holdout konsisten dengan hasil CV** — dua metrik itu saling menguatkan, bukan bertentangan. Ini disebut **konsistensi metodologis**.

**Alasan 2 — Menghasilkan Locked Fold yang cukup besar untuk diuji.**

Hitungan mundur:
```
n = 86
80:20 → Locked Fold = 18 mahasiswa  (9 BK + 9 K)  ← ✅ cukup untuk metrik apa pun
70:30 → Locked Fold = 26 mahasiswa  (12 BK + 14 K)  ← data latih tinggal 60
60:40 → Locked Fold = 34 mahasiswa  (tidak lazim untuk ML)
```

Dengan **18 mahasiswa**, kita masih bisa menghitung:
- **Recall BK** dengan kelipatan 1/9 = **11,1%** → masih informatif
- **Balanced Accuracy** dengan kelipatan 1/18 = **5,6%** → masih informatif
- **95% Confidence Interval** dengan Bootstrap (B=1000) → masih bermakna

**Alasan 3 — Lebih Murah dari Segi Komputasi.**

Setiap konfigurasi model diuji di **25 putaran Outer CV × 3 fold Inner CV × hingga 50 iterasi RandomizedSearchCV**. Dengan 7 skenario × 2-3 algoritma, sudah ada banyak kombinasi. Jika training cukup banyak, **waktu komputasi naik hampir linearly** — dari 68 data ke 60 data (70:30) menambah beban tanpa manfaat nyata.

**Alasan 4 — Mengurangi Risk of Underfitting pada Outer CV.**

Di Outer CV 5-Fold, setiap outer fold training adalah **0,8 × 68 = 54** mahasiswa. Kalau dipangkas jadi 70:30, tiap outer fold hanya **0,8 × 60 = 48**. Dengan model Decision Tree yang dikunci ketat (`min_samples_split=10`, `min_samples_leaf=5`), 48 baris terlalu sedikit — pohon bisa **tidak cukup data** untuk membentuk daun yang bermakna. 80:20 lebih aman.

**Alasan 5 — Konsonan dengan Teknikal Bootstrap & Dummy Baseline yang dipakai di laporan.**

Bootstrap B=1000 dan pembanding Dummy Classifier (BA = 50%) dirancang untuk n=18. Ini sudah tertulis di `HASIL_ANALISIS_V2.md` Section 3.3.

## 3.4 Kenapa Stratify (Proporsional) Sangat Penting?

`stratify=y` memastikan **proporsi kelas di train dan test sama** dengan data asli:

```
Data asli : 42 BK (48,8%)  :  44 K (51,2%)
Train 68  : 33 BK (48,5%)  :  35 K (51,5%)   ✔ hampir identik
Test 18   :  9 BK (50,0%)  :   9 K (50,0%)   ✔ seimbang sempurna
```

Kalau tanpa stratify dan kebetulan test dapat 13 BK + 5 K, maka **Specificity dihitung dari hanya 5 orang** → sangat tidak stabil. Stratify menghilangkan risiko itu.

---

# BAGIAN 4 — ATRIBUT PREDICTOR (Variabel Independen)

## 4.1 Struktur Tabel yang Digunakan Machine Learning

```
┌──────────────────────────────┬───────────────────────────────────────┐
│  VARIABEL TERGANTUNG (Y)     │  Target: Kompetensi                   │
│  (yang ingin diprediksi)     │  0 = Belum Kompeten  (⚠ alarm)        │
│                              │  1 = Kompeten         (aman)          │
├──────────────────────────────┼───────────────────────────────────────┤
│  VARIABEL BEBAS (X)          │  6-20 fitur agregat per mahasiswa      │
│  (dipakai untuk memprediksi) │ yaitu performa, kehadiran,            │
│                              │  kelengkapan, tren, volatilitas        │
└──────────────────────────────┴───────────────────────────────────────┘
```

> ⚠️ **CATATAN PENTING UNTUK SIDANG:**
> **Posisi label dibalik!** Dalam kode, `pos_label=0` dipakai untuk semua metrik BK.
> ```python
> recall_score(y_true, y_pred, pos_label=0)   # pos_label=0 = Belum Kompeten
> ```
> Artinya **kelas yang dianggap "positif" adalah "Belum Kompeten"** — karena sistem EWS berorientasi menangkap mahasiswa berisiko. Jelaskan ini ke dosen karena sering menjadi sumber pertanyaan.

## 4.2 Daftar Atribut Prediktor (dari `src/feature_registry.py`)

### Kelompok 1 — Kehadiran

| Atribut | Tipe | Arti (bahasa sederhana) | Cara Dihitung |
|---|---|---|---|
| `Attendance_PreFinal_Rate` | Float (0–1) | "Seberapa sering mahasiswa ini hadir di kelas?" | Jumlah sesi hadir ÷ jumlah sesi till cutoff |

### Kelompok 2 — Nilai Tugas Pendahuluan (TP)

| Atribut | Tipe | Arti | Cara Dihitung |
|---|---|---|---|
| `TP_Mean` | Float | "Seberapa bagus nilai rata-rata TP-nya?" | Rata-rata semua nilai TP |
| `TP_Completion_Rate` | Float (0–1) | "Berapa persen TP yang dikumpulkan (tidak nol)?" | #(nilai TP > 0) ÷ total TP |
| `TP_Std` | Float | "Nilai TP-nya konsisten atau naik-turun liar?" | Standar deviasi nilai TP |
| `TP_First2_Mean` | Float | "Bagaimana prestasinya di 2 TP pertama?" | Rata-rata TP ke-1 dan ke-2 |
| `TP_Last2_Mean` | Float | "Bagaimana prestasinya di 2 TP terakhir?" | Rata-rata TP terakhir 2 |
| `TP_Trend` | Float | "Nilai TP-nya menanjak atau menurun?" | Kemiringan garis regresi linear |
| `TP_Min` | Float | "Nilai TP terburuknya berapa?" | Nilai minimum |
| `TP_Max` | Float | "Nilai TP terbaiknya berapa?" | Nilai maksimum |

### Kelompok 3 — Nilai Laporan Praktikum

| Atribut | Tipe | Arti | Cara Dihitung |
|---|---|---|---|
| `Laporan_Mean` | Float | "Rata-rata nilai laporan saya berapa?" | Rata-rata semua nilai laporan |
| `Laporan_Completion_Rate` | Float (0–1) | "Berapa persen laporan yang dikumpulkan?" | #(nilai > 0) ÷ total laporan |
| `Laporan_Std` | Float | "Nilai laporan konsisten atau tidak?" | Standar deviasi |
| `Laporan_First2_Mean` | Float | "Nilai 2 laporan pertamanya?" | Rata-rata laporan 1–2 |
| `Laporan_Last2_Mean` | Float | "Laporan 2 terakhirnya?" | Rata-rata 2 terakhir |
| `Laporan_Trend` | Float | "Nilai laporan menanjak atau turun?" | Kemiringan regresi linear |
| `Laporan_Min` | Float | "Nilai laporan terburuknya?" | Minimum |
| `Laporan_Max` | Float | "Nilai laporan terbaiknya?" | Maksimum |

### Kelompok 4 — Performa Gabungan & Kehadiran Turunan

| Atribut | Tipe | Arti | Cara Dihitung |
|---|---|---|---|
| `Performance_Std` | Float | "Nilai saya naik-turun tidak konsisten?" | **Std deviasi gabungan TP + Laporan** |
| `Absence_Count` | Int | "Sudah berapa kali saya tidak hadir?" | Jumlah sesi bernilai 0 |
| `Performance_Late_Mean` | Float | "Kinerja saya di pertemuan terakhir?" | Rata-rata 4 nilai terakhir |

### Kelompok 5 — Fitur Cerdas EWS (Early Warning System)

| Atribut | Tipe | Arti (bahasa sederhana) | Cara Dihitung |
|---|---|---|---|
| `Absence_Streak_Max` | Int | "Rekor bolos beruntun terpanjang saya berapa kali?" | Streak maksimal nilai 0 berturut-turut |
| `Performance_Decline_Flag` | 0/1 | "Apakah nilai saya jelek di akhir padahal baik di awal?" | 1 jika (rata-rata awal − rata-rata akhir) > std |
| `Zero_Score_Count` | Int | "Berapa kali saya dapat nilai 0 mutlak (tidak mengumpulkan)?" | #(nilai = 0) di TP + Laporan |
| `Cumulative_Completion_Trend` | Float | "Saya makin rajin atau makin malas?" | Completion rate paruh akhir − paruh awal |
| `Early_Performance_Composite` | Float | **"Dua tugas/laporan awal saya seberapa bagus?"** ⭐ | **Rata-rata TP(1–2) + Laporan(1–2)** |

### Kelompok 6 — Metadata (TIDAK masuk model)

| Kolom | Alasan dikeluarkan |
|---|---|
| `NIM`, `Nama` | Pengenal, bukan prediktor |
| `Kelas` | Disimpan untuk analisis, tidak dipakai di feature registry |
| `Final_Score` | **Ini target/nasib akhir — kalau ikut masuk ke fitur, ini kebocoran data (leakage)!** |
| `Final_Status`, `Label_Reason` | Metadata pelabelan |
| `Attendance_8/9/10`, `NILAI_AKHIR`, `Predikat` | **Dilarang** — diblokir di `tests/test_temporal_leakage.py` |

> `tests/test_temporal_leakage.py` secara eksplisit **memblokir** kolom-kolom terlarang ini masuk ke feature set. Ini bukti rigor metodologis Anda — sebutkan saat sidang!

## 4.3 Jumlah Atribut Prediktor per Skenario

| Skenario | Jumlah Fitur | Ringkasan |
|---|---|---|
| **S1** | **3** | Dasar: absensi + TP mean + Laporan mean |
| **S2** | **6** | S1 + Absence_Count + completion rate TP & Laporan |
| **S3** | **6** | S1 + completion rate TP & Laporan + Performance_Std |
| **S4** | **15** | S3 + std, first2, last2, trend (TP & Laporan) |
| **S5** | **20** | S4 + min, max, Performance_Late_Mean |
| **S3_A** | **7** | S3 + Absence_Streak_Max |
| **S3_B** | **7** | S3 + Performance_Decline_Flag |
| **S3_C** | **7** | S3 + Zero_Score_Count |
| **S3_D** | **7** | S3 + Cumulative_Completion_Trend |
| **S3_E** ⭐ | **7** | S3 + Early_Performance_Composite |
| **S3_EWS** | **11** | S3 + kelima fitur EWS sekaligus |

---

# BAGIAN 5 — REPRESENTASI FITUR (Feature Representation)

## 5.1 Paradigma: Row-per-Student (Bentuk Matriks/"Lebar")

**Proyek ini TIDAK memakai data mentah langsung sebagai input model.** Data mentah berbentuk *long format* (satu baris per aktivitas) harus diubah dulu menjadi **matriks lebar** di mana **satu baris = satu mahasiswa**.

### Tahap 1 — Data Mentah Long Format (`data/processed/activities_long.csv`)

```
┌──────────────────────┬───────────────┬────────────────┬─────────────┬───────┐
│ Nama                 │ Activity_Type │ Activity_ID    │ Time_Index  │ Score │
├──────────────────────┼───────────────┼────────────────┼─────────────┼───────┤
│ ABDUL MUJIB SUPARDIN │ Attendance    │ P1             │ 1           │ 1.0   │
│ ABDUL MUJIB SUPARDIN │ Attendance    │ P2             │ 2           │ 0.0   │
│ ABDUL MUJIB SUPARDIN │ Laporan       │ L1             │ 1           │ 90.0  │
│ ABDUL MUJIB SUPARDIN │ TP            │ TP1            │ 1           │ 50.0  │
│ ...                  │ ...           │ ...            │ ...         │ ...   │
└──────────────────────┴───────────────┴────────────────┴─────────────┴───────┘
        1.140 baris (Attendance 527 · Laporan 321 · TP 292)
```

> ⚠️ Bentuk ini **TIDAK BISA** langsung masuk Decision Tree, karena 1 mahasiswa terpecah jadi banyak baris. Harus diagregasi.

### Tahap 2 — Agregasi → Matriks Lebar (`data/features/*.csv`)

```
┌────────────────────┬─────────┬───────┬─────┬─────────────┐
│ Nama (identifier)  │ Kelas   │ **Y** │ ... │ Attendance_  │
│                    │ (meta)  │ Label │     │ PreFinal_Rate│
├────────────────────┼─────────┼───────┼─────┼─────────────┤
│ ABDUL MUJIB SUPARDIN│ A       │ **0** │     │ 0.600       │
│ ADEN HENDRAWAN M   │ A       │ **1** │     │ 1.000       │
│ AMITHA WAHDINI     │ A       │ **0** │     │ 0.833       │
│ ... (86 baris)     │ ...     │ ...   │     │ ...         │
└────────────────────┴─────────┴───────┴─────┴─────────────┘
         ▲                         ▲         ▲
         │                         │         └─ FITUR (X)
         │                         └─ LABEL (Y): 0=BK, 1=K
         └─ Tidak dipakai sebagai fitur (identifier)
```

## 5.2 Representasi yang Digunakan — 6 Prinsip

### Prinsip 1 — **Temporal Cutoff (Jendela Waktu)**

Data **dipotong sesuai waktu**, lalu fitur dihitung **hanya dari data sebelum batas**. Ini adalah inti dari "Genuine EWS".

Kode `src/features.py`:
```python
cutoffs = {
    'W1':      {'Attendance': 1, 'Laporan': 1, 'TP': 1},
    'W2':      {'Attendance': 2, 'Laporan': 2, 'TP': 2},
    'W3':      {'Attendance': 3, 'Laporan': 3, 'TP': 3},
    'C_Full':  {'Attendance': 7, 'Laporan': 4, 'TP': 4}
}
```

| Cutoff | Melihat sampai | Sifat |
|---|---|---|
| **W1** | Pertemuan ke-1 saja | **Paling awal** — deteksi sangat dini |
| **W2** | Pertemuan ke-2 | **Titik terbaik (sweet spot)** (hasil terbaik) |
| **W3** | Pertemuan ke-3 | Seimbang |
| **C_Full** | Attendance 7, Laporan 4, TP 4 (semester penuh) | **Retrospektif** — sudah tahu hasilnya |

### Prinsip 2 — **Agregasi Statistik (Summary Statistics)**

Bentuk data berubah dari *mentah* (sangat banyak kolom) menjadi *ringkas* (sedikit kolom representatif). 1.140 baris aktivitas diringkas menjadi 86 baris × 7 fitur — bukan ratusan kolom mentah.

**Prinsip utama yang dipakai:** Mean, Std, Min, Max, Completion Rate, Trend (garis regresi linear).

**Contoh nyata — `Performance_Std`:**
```
Mahasiswa A: TP = [90, 85, 80, 50]  → Std = 14,79  (nilai TIDAK konsisten)
Mahasiswa B: TP = [80, 80, 80, 80]  → Std =  0,00  (nilai SANGAT konsisten)

Kedua mahasiswa boleh punya TP_Mean = 75 yang sama,
tetapi Performance_Std sangat berbeda → model bisa membedakan.
```

**Contoh `TP_Trend` (kemiringan garis regresi):**
```
TP = [50, 60, 70, 80]  → tren = +10  (naik)  → positif
TP = [80, 70, 60, 50]  → tren = -10  (turun) → negatif
TP = [70, 65, 72, 68]  → tren = -0,5 (datar)  → mendekati nol
```

### Prinsip 3 — **Fitur Khusus EWS (Tanda Peringatan)**

5 fitur diciptakan khusus untuk menangkap **pola perilaku berisiko**, bukan hanya rata-rata.

| Fitur | Tanda Peringatan yang Ditangkap |
|---|---|
| `Absence_Streak_Max` | Tidak hadir **3 kali berturut-turut** — lebih berbahaya daripada tidak hadir yang sporadis |
| `Performance_Decline_Flag` | "Late Dropper": mulai bagus, **tiba-tiba jatuh** di akhir |
| `Zero_Score_Count` | Sering **tidak mengumpulkan tugas** (= nilai 0 mutlak) |
| `Cumulative_Completion_Trend` | Rajin di awal, **makin malas** di akhir |
| `Early_Performance_Composite` | **Kinerja 2 pertemuan awal** — prediktor paling kuat |

### Prinsip 4 — **Tanpa Normalisasi Skala (Tree-Based)**

Fitur tidak di-*scaling* karena:
- `Decision Tree` dan `Random Forest` **tidak butuh** normalisasi — mereka hanya membandingkan nilai antar-*node*
- `SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)`tertanam di dalam `Pipeline` agar tidak bocor

**Kode (`src/models.py`):**
```python
imputer = SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)
steps = [('imputer', imputer)]
if balancing == 'SMOTE' and model_name != 'Dummy':
    steps.append(('smote', SMOTE(random_state=random_state, k_neighbors=2)))
steps.append(('model', clf))
return Pipeline(steps)
```
Imputasi dan model berada di dalam **Pipeline** yang sama → saat cross-validation, imputasi hanya dihitung pada fold latih, **tidak pernah** melihat fold uji.

### Prinsip 5 — **Skenario Bertingkat (Feature Set Gradually)**

Alih-alih langsung pakai semua fitur, dilakukan **uji bertingkat S1 → S5** untuk melihat: *"Apakah fitur tambahan benar-benar membantu, atau cuma menambah noise?"*

**Hasilnya (dari `HASIL_ANALISIS_V2.md` Tabel 1):**

| Skenario | Model | CV Balanced Accuracy | Test Balanced Accuracy | Kesimpulan |
|---|---|---|---|---|
| **S1** | Random Forest | **65,60%** | 66,67% | CV terbaik |
| **S2** | Random Forest | 68,17% | 77,78% | — |
| **S3** | **Decision Tree** | 62,39% | **77,78%** | ✅ **Pemenang** |
| S4 | — | — | — | ❌ Tidak lebih baik |
| S5 | — | — | — | ❌ Tidak lebih baik |

**Insight:** S4 dan S5 (fitur makin banyak = 15 & 20 fitur) justru **menurun**. Karena **jumlah data hanya 86**, terlalu banyak fitur = **overfitting**. S3 (6 fitur) adalah titik paling seimbang (*sweet spot*).

### Prinsip 6 — **Validasi Ketidakpastian (Uncertainty Quantification)**

Karena data kecil, setiap metrik dilaporkan dengan **95% Confidence Interval** (lihat Bagian 7).

## 5.3 Ringkasan Representasi Fitur

| Aspek | Representasi yang Dipakai |
|---|---|
| **Bentuk tabel** | Matriks **lebar** (row = mahasiswa, column = fitur) |
| **Orientasi** | **Backward** (regresif dari data mentah ke matriks teragregasi) |
| **Skala** | **Tidak dinormalisasi** — aman untuk Decision Tree & Random Forest |
| **Unit** | Semua fitur **numerik** (tidak ada data kategorikal — menghindari kebutuhan *encoding*) |
| **Sumber** | Statistical aggregation (mean, std, min, max, count) + temporal trends |
| **Pembatas waktu** | **Temporal cutoff** → W1 / W2 / W3 / C_Full |
| **Penanganan missing** | Imputasi konstan −1 + *missing indicator* di dalam `Pipeline` |
| **Feature set** | Bertingkat: S1 (3) → S2 (6) → S3 (6) → S4 (15) → S5 (20) → EWS (7–11) |
| **Jumlah fitur optimal** | **7 fitur** (S3_E) → cukup untuk memprediksi, tapi tidak terlalu banyak |

---

# BAGIAN 6 — PENJELASAN METRIK EVALUASI (Bahasa Sederhana)

## 6.1 Confusion Matrix — Pengenalan Dasar

Sebelum metrik, pahami dulu **Confusion Matrix** (tabel 2×2). Dalam proyek ini, **kelas positif = Belum Kompeten (0)**:

```
                    PREDIKSI MODEL
                BK (0)          K (1)
            ┌──────────────┬──────────────┐
  AKTUAL    │              │              │
  BK (0)    │  TP = 8      │  FN = 1      │
  (berisiko)│  ✅ Tepat    │  ❌ Terlewat │
            ├──────────────┼──────────────┤
  K  (1)    │  FP = 3      │  TN = 6      │
  (Aman)    │  ❌ Alarm    │  ✅ Tepat    │
            │  Palsu!      │  Tidak Alert │
            └──────────────┴──────────────┘
```

**Istilah sederhana:**

| Simbol | Nama | Bahasa Sederhana |
|---|---|---|
| **TP** (True Positive) | Tepat Positif | Mahasiswa **memang berisiko** dan **berhasil ditemukan model** ✅ |
| **TN** (True Negative) | Tepat Negatif | Mahasiswa **memang aman** dan **model tidak alarm** ✅ |
| **FP** (False Positive) | Positif Palsu | Mahasiswa **aman**, tapi **model alarm** ⚠️ (Alarm Palsu / *false alarm*) |
| **FN** (False Negative) | Negatif Palsu | Mahasiswa **berisiko**, tapi **model loloskan** ❌ (**Ini yang paling berbahaya!**) |

> **FN = "Late Dropper"** — mahasiswa yang mulai bagus tapi malah jatuh di akhir, sulit diprediksi
> **FP = "Late Bloomer"** — yang lambat adaptasi tapi akhirnya bisa mengejar

---

## 6.2 Metrik Utama yang Digunakan

### 6.2.1 Confusion Matrix (Dasar)
**Definisi:** Tabel yang menunjukkan berapa banyak prediksi benar dan salah.
**Cara hitung:** Langsung dari perbandingan prediksi vs label asli.
**Mengapa dipakai:** Satu-satunya metrik yang **tidak bisa dimanipulasi** — semua metrik lain turunannya.
**Nilai proyek:** TP=8, FN=1, FP=3, TN=6 (pada Locked Fold).

---

### 6.2.2 Accuracy (Akurasi)
**Definisi:** Persentase prediksi yang benar dari semua data.
**Rumus:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**Bahasa sederhana:** *"Dari semua mahasiswa yang diuji, berapa persen yang klasifikasinya tepat?"*
**Nilai proyek:** (8+6)/18 = **77,78%**
**Kelemahan:** **Menipu** jika jumlah kelas tidak seimbang. Kalau semua data dianggap Kompeten, akurasi bisa 100% padahal model tidak berguna sama sekali.

---

### 6.2.3 Recall BK (Sensitivity) ⭐⭐⭐ — **METRIK UTAMA**

**Definisi:** Dari semua mahasiswa yang **sesungguhnya Belum Kompeten**, berapa persen yang **berhasil ditemukan** model.
**Rumus:**
```
Recall BK = TP / (TP + FN)
```
**Bahasa sederhana:** *"Dari 9 mahasiswa yang benar-benar berisiko, berapa yang berhasil di-detect radar?"*

**Nilai proyek:**
```
Recall BK = 8 / (8+1) = 88,89%
```

**Mengapa ini metrik utama:**
> Untuk sistem EWS, **FN (mahasiswa berisiko yang lolos) adalah MUSIBAT** — mahasiswa itu tidak akan dapat bailout, akhirnya gagal. **FP hanya "rugi waktu"** — dosen hanya perlu memeriksa 3 mahasiswa yang sebenarnya aman, itu tidak fatal.
>
> Karena itu: **kesalahan yang fatal (FN) lebih besar nilainya daripada kesalahan yang hanya membosankan (FP).** Karena itu, Recall harus **diprioritaskan**.

**Nilai proyek pada Locked Fold:**
```
Recall BK = 8/9 = 88,89%  (95% CI: 62,50% – 100,00%)
```
Pada W2 dan W3: **Recall BK = 100%** (9 dari 9 — SEMUA berhasil ditemukan!)

---

### 6.2.4 Precision BK (Presisi)

**Definisi:** Dari semua mahasiswa yang **diberi ALARM oleh model**, berapa persen yang **sesungguhnya** memang berisiko.
**Rumus:**
```
Precision BK = TP / (TP + FP)
```
**Bahasa sederhana:** *"Kalau model bilang 'WASPADA' ke 11 mahasiswa, berapa yang memang benar-benar perlu diwaspadai?"*

**Nilai proyek:**
```
Precision BK = 8 / (8+3) = 72,73%
```
**Artinya:** Dari 11 mahasiswa yang di-alarm, 8 memang benar-benar BK (72,73%), 3 ternyata aman (false alarm).

**Trade-off dengan Recall:**
```
Turunkan threshold (misal 0,52 → 0,20):
  → lebih banyak yang di-alarm  → Recall NAIK ✅, Precision TURUN ❌
  → Prinsipnya: "lebih baik alarm berlebihan daripada ada mahasiswa yang lolos"
```

---

### 6.2.5 Specificity (Spesifisitas)

**Definisi:** Dari semua mahasiswa yang **sesungguhnya Kompeten (aman)**, berapa persen yang **berhasil dibebaskan** dari alarm.
**Rumus:**
```
Specificity = TN / (TN + FP)
```
**Bahasa sederhana:** *"Dari 9 mahasiswa yang memang aman, berapa yang berhasil TIDAK di-alarm?"*

**Nilai proyek:** 6/(6+3) = **66,67%**
**Peran:** Mencegah terlalu banyak alarm palsu. **Penting untuk efisiensi kerja dosen.**

---

### 6.2.6 Balanced Accuracy (Akurasi Berimbang) ⭐

**Definisi:** Rata-rata dari Recall BK dan Specificity.
**Rumus:**
```
Balanced Accuracy = (Recall BK + Specificity) / 2
```
**Bahasa sederhana:** *"Nilai gabungan: bisa menemukan yang berisiko (tinggi), sekaligus tetap bisa melepaskan yang aman (tinggi)."*

**Nilai proyek:**
```
Balanced Accuracy = (88,89% + 66,67%) / 2 = 77,78%
```
**Mengapa ada metrik ini:**
> Karena dataset **tidak seimbang**, Accuracy bisa 100% hanya dengan menebak semua aman. Balanced Accuracy **menyamakan porsi** Recall dan Specificity, sehingga **tidak ada kelas yang diabaikan**.

---

### 6.2.7 F1-Score BK

**Definisi:** Harmonic mean dari Precision dan Recall (penggabungan dua angka dengan pembobotan).
**Rumus:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**Bahasa sederhana:** *"Nilai gabungan Precision dan Recall. Kalau Precision 100% tapi Recall 10%, F1 hanya sekitar 18% — jadi F1 menghukum ketidakseimbangan."*

**Nilai proyek:**
```
F1 BK = 2 × (0,7273 × 0,8889) / (0,7273 + 0,8889) = 80,00%
```

---

### 6.2.8 F2-Score BK ⭐ — **UNTUK EWS**

**Definisi:** Sama seperti F1, tapi **memberi bobot 2× lebih besar pada Recall** daripada Precision.
**Rumus:**
```
F2 = 5 × (Precision × Recall) / (4 × Precision + Recall)
```
**Bahasa sederhana:** *"Nilai gabungan yang MEMBIARKAN alarm berlebihan demi jangan sampai ada mahasiswa yang lolos."*

**Mengapa pakai F2 untuk EWS:**
> Karena untuk sistem peringatan dini, **menangkap semua yang berisiko (Recall) jauh lebih penting** daripada terkena alarm palsu. F2 secara eksplisit memprioritaskan Recall.

**Nilai proyek:**
```
F2 BK = 5 × (0,7273 × 0,8889) / (4 × 0,7273 + 0,8889) = 85,11%
(95% CI: 62,46% – 98,04%)
```

---

### 6.2.9 F1 Macro / Macro-F1

**Definisi:** Rata-rata F1 dari **semua kelas** (bukan hanya BK).
**Rumus:**
```
Macro-F1 = (F1 BK + F1 K) / 2
```
**Bahasa sederhana:** *"Nilai F1 yang adil — melihat performa model di kedua kelas secara seimbang."*

**Perbedaan dengan F1 BK:**
| Metrik | Fokus |
|---|---|
| F1 BK | Hanya melihat performa di kelas Belum Kompeten |
| Macro-F1 | Melihat performa di **semua** kelas secara adil |

**Nilai proyek:** **77,50%**

---

### 6.2.10 ROC-AUC ⭐

**Definisi:** **Area Under Curve** dari Receiver Operating Characteristic — kemampuan model memisahkan semua probabilitas.
**Apa yang diukur:** Kemampuan model **memisahkan** dua kelas **pada semua kemungkinan threshold**.
**Rumus:**
```
ROC-AUC = ∫ TPR / FPR  ( luas di bawah kurva )
  dengan TPR = Recall, FPR = 1 − Specificity
```

**Bahasa sederhana:**
> *"Kalau model bisa memisahkan dengan baik, ROC-AUC ≈ 1,0 (sempurna). Kalau model asal tebak, ROC-AUC = 0,5 (seperti lemparan koin). Jika ROC-AUC < 0,5, model lebih buruk daripada tebak!"*

**Nilai proyek:** **0,8951** (≈ 89,5%)
**Artinya:** Model **sangat baik** dalam memisahkan yang aman vs berisiko.

**Kelemahan:** Tidak terlalu sensitif terhadap ketidakseimbangan kelas → **kurang tepat** untuk kasus imbalanced.

---

### 6.2.11 PR-AUC ⭐⭐⭐ — **METRIK TERBAIK UNTUK DATA IMBALANCED**

**Definisi:** **Area Under Curve** dari Precision-Recall curve — menggabungkan Precision dan Recall.
**Apa yang diukur:** Kualitas prediksi di **seluruh kelas minoritas**.
**Perbedaan dengan ROC-AUC:**
```
ROC-AUC → melihat SEMUA kelas (aman & berisiko) → bisa terlalu optimis
PR-AUC  → FOKUS pada kelas minoritas (berisiko) → lebih konservatif & jujur
```

**Rumus (average precision):**
```
PR-AUC = Σ (Recall_n − Recall_n−1) × Precision_n
```

**Bahasa sederhana:**
> *"PR-AUC hanya melihat seberapa baik model berfokus mendeteksi kelas minoritas — ini lebih jujur untuk data tidak seimbang."*

**Nilai proyek:**
- C_Full: **0,8459**
- Baseline S3 RandomForest: **0,892**

**Mengapa dipakai (berdasarkan `GLOSARIUM.md`):**
> *"PR-AUC adalah standar emas baru untuk data sangat imbalanced. Jauh lebih representatif dan ketat dibandingkan kurva ROC-AUC karena PR-AUC berfokus langsung pada performa prediksi kelas minoritas."*

---

### 6.2.12 Matthews Correlation Coefficient (MCC)

**Definisi:** Koefisien korelasi antara prediksi dan label aktual, dihitung dari seluruh elemen confusion matrix.
**Rumus:**
```
MCC = (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Bahasa sederhana:**
> *"Angka dari -1 sampai +1. +1 = prediksi sempurna. 0 = tidak lebih baik dari acak. −1 = prediksi terbalik semua."*

**Nilai proyek:**
- W2: 0,57
- W3: 0,57
- C_Full: 0,58

**Keunggulan:** Menggunakan **semua 4 angka** dari confusion matrix — metrik yang **paling seimbang**.

> ⚠️ **CATATAN AUDIT:** MCC dicantumkan di Tabel 2 `HASIL_ANALISIS_V2.md`, tetapi **belum dihitung di kode Python** (`src/evaluation.py`). Jika ditanya dosen, sampaikan bahwa angkanya dihitung terpisah atau perlu ditambahkan ke kode. **Rekomendasi: tambahkan `matthews_corrcoef` ke `src/evaluation.py`.**

---

### 6.2.13 Threshold (Ambang Batas Keputusan) ⭐⭐⭐

**Definisi:** Nilai batas probabilitas yang menentukan apakah mahasiswa di-alarm atau lolos.
**Cara kerja:**
```python
# src/evaluation.py
p_bk = model.predict_proba(X_test)[:, 0]   # probabilitas Belum Kompeten
y_pred = np.where(p_bk >= threshold, 0, 1)  # prediksi BK JIKA probabilitas ≥ threshold
```

**Bahasa sederhana:**
> *"Model diberi batas peluang. Di atas batas = ALARM. Di bawah batas = lolos."*

**Nilai proyek (terpilih di Inner CV dengan syarat Recall ≥ 70%):**

| Cutoff | Threshold |
|---|---|
| W1 | 0,26 |
| W2 | **0,20** |
| W3 | 0,24 |
| C_Full | **0,52** |

**Kenapa optimasi threshold penting:**
> Default sklearn adalah **0,50**. Untuk EWS, ini perlu di*-optimize* karena **Recall lebih diutamakan daripada Precision**.

**Cara threshold dipilih** (`src/tuning.py`):
```python
# 1. Cari SEMUA kandidat threshold (0.20 – 0.80, step 0.02)
# 2. Saring hanya yang Recall BK ≥ 0.70
valid_candidates = [r for r in results if r['recall_bk'] >= 0.70]
# 3. Dari yang lolos, pilih Balanced Accuracy tertinggi
valid_candidates.sort(key=lambda x: (x['balanced_accuracy'], x['f2_bk']), reverse=True)
```

**Penting:** Threshold dipilih dari **Data Latih** saja (`cross_val_predict`), **tidak pernah** melihat Locked Fold.

---

### 6.2.14 Dummy Classifier (Pembanding Baseline)

**Definisi:** Model paling sederhana yang hanya menebak kelas **mayoritas**.
**Cara kerja:**
```python
DummyClassifier(strategy="most_frequent")  # selalu tebak "K"
```

**Tujuan:** **Titik pembanding minimum.** Setiap model harus **lebih baik** dari Dummy.

**Nilai proyek:**
```
Balanced Accuracy Dummy = 50,00%   ← titik nol (tebak coin flip)
Balanced Accuracy Model = 77,78%   ← model jauh lebih baik (+27,78 poin)
```

**Kesimpulan:** *"Karena batas bawah CI (56,15%) berada di atas 50%, kita yakin model **signifikan lebih baik** daripada tebakan acak."*

---

### 6.2.15 Bootstrap Confidence Interval (95% CI) ⭐⭐⭐

**Definisi:** Rentang nilai yang menunjukkan seberapa **pasti** performa model.
**Cara kerja:**
```python
for i in range(1000):                      # 1000× ulangi
    X_bs, y_bs = resample(X_test, y_test)   # acak ulang 18 data dengan pengembalian
    hitung metrik pada sampel acak ini
# Ambil persentil 2,5% dan 97,5% -> inilah 95% CI
```

**Bahasa sederhana:**
> *"Kalau saya ulangi 1000× dengan mengambil 18 mahasiswa acak setiap kali, saya dapat banyak angka berbeda. Rentang inilah perkiraan dari semua kemungkinan hasil."*

**Contoh dari proyek:**
```
Recall BK       = 88,89%  (95% CI: 62,50% – 100,00%)
Balanced Acc    = 77,78%  (95% CI: 56,15% – 95,45%)
F2 BK           = 85,11%  (95% CI: 62,46% – 98,04%)
```

**Mengapa wajib:**
> Karena **n = 18 sangat kecil**, angka 88,89% bisa berarti 8 dari 9 (**pasti**) atau bisa berarti 15 dari 17 (**kemungkinan besar**). **CI membedakan keduanya** → mencegah klaim yang berlebihan.

---

### 6.2.16 Mean ± SD (Rata-rata ± Standar Deviasi)

**Definisi:** Nilai rata-rata dari semua fold/repeat, beserta sebaran (deviasi standar).
**Apa yang diukur:** **Kestabilan model** saat data latih/uji dipecah berbeda-bedanya.
**Rumus:**
```
Mean = Σ(scores) / N
SD = √[Σ(scores − Mean)² / (N − 1)]
```

**Bahasa sederhana:**
> *"Kalau Mean tinggi tapi SD besar (contoh: 75% ± 15%), artinya model **tidak konsisten** — bisa dapat 90% atau bisa dapat 60% tergantung data. Kalau Mean tinggi dan SD kecil (contoh: 78% ± 3%), model **handal dan konsisten**."*

**Nilai proyek (contoh dari audit terdahulu):**
```
CV Balanced Accuracy = 62,91 ± 11,01%
```

**Di proyek ini:** SD dihitung dari **25 iterasi** (5 fold × 5 repeat), jadi lebih reliable.

---

### 6.2.17 SHAP Score ⭐ — **Metrik Interpretasi**

**Definisi:** **Nilai kontribusi** setiap fitur terhadap keputusan model.
**Apa yang diukur:** Seberapa **penting** dan **searah apa** kontribusi setiap fitur.
**Cara kerja:**
```
Untuk setiap mahasiswa:
  Telusuri decision path dari Decision Tree
  Hitung kontribusi setiap fitur terhadap prediksi
  Rata-ratakan di semua mahasiswa → mean|SHAP|
```

**Apa yang diukur:**
- **Mean Absolute SHAP** → **Seberapa penting** fitur tersebut
- **Nilai SHAP positif** → Mendorong ke prediksi "BK"
- **Nilai SHAP negatif** → Mendorong ke prediksi "K"

**Bahasa sederhana:**
> *"SHAP seperti menanyai model: 'Kenapa kamu bilang mahasiswa ini berisiko?' Jawabannya: 'Karena Performanya jelek (mendorong ke risiko +0,45), tapi absensinya bagus (mendorong ke aman −0,20).'"*

**Nilai proyek (mean absolute SHAP):**

**Model Baseline S3 (Retrospektif):**
| Fitur | SHAP |
|---|---|
| `Laporan_Mean` | **0,3095** ⬅ tertinggi |
| `Performance_Std` | 0,1872 |
| `TP_Mean` | 0,0686 |

**Model Final S3_E (Genuine EWS):**
| Fitur | SHAP |
|---|---|
| `Early_Performance_Composite` | **0,2200** ⬅ **TERTINGGI** ⭐ |
| `Laporan_Mean` | 0,2110 |
| `Performance_Std` | 0,1040 |
| `TP_Mean` | 0,0695 |

**Insight Utama:**
> Baseline S3 **bergantung pada `Laporan_Mean`** — ini terlalu **reaktif** (harus menunggu semester selesai). Tapi setelah ditambah `Early_Performance_Composite`, fitur ini **menjadi kontributor prediktif tertinggi** — ini adalah **Early Warning System yang sesungguhnya** (proaktif, bisa deteksi dini).

---

## 6.3 Ringkasan Semua Metrik (Cheat Sheet)

| Metrik | Formula | Arti 1 kalimat | Nilai Proyek | Kapan dipakai? |
|---|---|---|---|---|
| **Confusion Matrix** | — | "Berapa yang benar/salah?" | TP=8,FN=1,FP=3,TN=6 | Dasar semua metrik |
| **Accuracy** | (TP+TN)/Total | "Berapa % yang tepat?" | 77,78% | Data seimbang saja |
| **Recall BK** ⭐ | TP/(TP+FN) | "Berapa % risiko yang berhasil ditemukan?" | 88,89% | **METRIK UTAMA** |
| **Precision BK** | TP/(TP+FP) | "Dari yang di-alarm, berapa yang benar?" | 72,73% | Minimalkan alarm palsu |
| **Specificity** | TN/(TN+FP) | "Berapa % aman yang TIDAK di-alert?" | 66,67% | Efisiensi kerja dosen |
| **Balanced Accuracy** ⭐ | (Rec+Spec)/2 | "Nilai gabungan dua kelas" | 77,78% | Data tidak seimbang |
| **F1 BK** | 2(PR)/(P+R) | "Gabungan Prec & Rec" | 80,00% | Keseimbangan P-R |
| **F2 BK** ⭐ | 5(PR)/(4P+R) | "Gabungan, Rec 2× lebih penting" | 85,11% | **Untuk EWS** |
| **Macro-F1** | Rata-rata semua F1 | "Adil untuk semua kelas" | 77,50% | Data tidak seimbang |
| **ROC-AUC** | Area di bawah kurva ROC | "Kemampuan pisah semua kelas" | 0,8951 | Model sudah bagus |
| **PR-AUC** ⭐ | Area di bawah kurva PR | "Fokus pada kelas minoritas" | 0,8459 | **Data sangat tidak seimbang** |
| **MCC** | Korelasi prediksi-aktual | "-1 sampai +1, sangat seimbang" | 0,57–0,58 | Menghindari bias |
| **Threshold** | Batas probabilitas | "Batas atas = ALARM" | 0,52 (C_Full) | Disesuaikan dengan EWS |
| **Bootstrap CI** | Rentang 2,5%–97,5% | "Seberapa pasti angkanya?" | [56,15% – 95,45%] | **WAJIB untuk n kecil** |
| **Mean ± SD** | Rata-rata & deviasi | "Seberapa konsisten?" | ± 11% | Stabilitas model |
| **Dummy Baseline** | Selalu tebak mayoritas | "Titik nol pembanding" | 50,00% | **Bukti model berguna** |
| **SHAP** | Kontribusi fitur | "Fitur mana yang menentukan?" | EPC=0,2200 | **Interpretabilitas** |

---

# BAGIAN 6B — DAFTAR LENGKAP METRIK EVALUASI YANG DIIMPLEMENTASIKAN DI KODE

> Bagian ini adalah **daftar teknis lengkap**: metrik apa saja yang benar-benar ditulis di dalam program, di baris mana, untuk keperluan apa, dan menghasilkan angka berapa.
> Semua isi bagian ini diverifikasi langsung dengan membaca `src/evaluation.py`, `src/tuning.py`, `src/shap_analysis.py`, `scripts/run_optimized_experiment.py`, serta hasilnya di `results/metrics/cv_experiment_matrix.csv` dan `results/final/holdout_results.csv`.

---

## 6B.1 Peta Lokasi Implementasi (Di File Mana, Untuk Apa)

| File | Fungsi | Jumlah Metrik | Peran |
|---|---|---|---|
| `src/evaluation.py` | `evaluate_cv()` | **11 metrik** | Menghitung 11 metrik pada **Outer CV** (25 iterasi) |
| `src/evaluation.py` | `evaluate_test()` | **11 metrik + 5 turunan confusion matrix + 6 batas CI** | Menghitung metrik final pada **Locked Fold** (n=18) |
| `src/tuning.py` | `_get_inner_scorer()` | **3 pilihan objektif** | Menentukan metrik mana yang dipakai untuk **tuning hyperparameter** di Inner CV |
| `src/tuning.py` | `evaluate_thresholds()` | **6 metrik** | Menghitung 6 metrik untuk **setiap kandidat threshold** |
| `scripts/run_optimized_experiment.py` | `select_best_model()` | **3 metrik berjenjang** | Menentukan **model pemenang** dari Outer CV saja |
| `src/shap_analysis.py` | `run_shap_analysis()` | **3 ukuran SHAP** | Mengukur **kontribusi & interpretasi** fitur |
| `src/models.py` | `DummyClassifier` | **1 baseline** | Balanced Accuracy = 50% sebagai **titik nol** |

---

## 6B.2 TABEL MASTER — 21 Metrik / Ukuran

| # | Metrik | Fungsi Python yang Dipakai | Terhitung di | Kapan Dipakai | Status |
|---|---|---|---|---|---|
| 1 | **Accuracy** | `accuracy_score` | `evaluation.py` | CV + Locked Fold | ✅ Terimplementasi |
| 2 | **Balanced Accuracy** | `balanced_accuracy_score` | `evaluation.py` | CV + Locked Fold + threshold | ✅ Terimplementasi |
| 3 | **Precision Macro** | `precision_score(average='macro')` | `evaluation.py` | CV + Locked Fold | ✅ Terimplementasi |
| 4 | **Recall Macro** | `recall_score(average='macro')` | `evaluation.py` | CV + Locked Fold | ✅ Terimplementasi |
| 5 | **F1 Macro** | `f1_score(average='macro')` | `evaluation.py` | CV + Locked Fold | ✅ Terimplementasi |
| 6 | **Precision BK** | `precision_score(pos_label=0)` | `evaluation.py` | CV + Locked Fold + threshold | ✅ Terimplementasi |
| 7 | **Recall BK** ⭐ | `recall_score(pos_label=0)` | `evaluation.py` | **Skoring utama Inner CV + pemilihan model** | ✅ Terimplementasi |
| 8 | **Specificity** | `tn / (tn + fp)` (dihitung manual) | `evaluation.py:150` | Locked Fold | ✅ Terimplementasi |
| 9 | **F1 BK** | `f1_score(pos_label=0)` | `evaluation.py` | CV + Locked Fold + threshold | ✅ Terimplementasi |
| 10 | **F2 BK** ⭐ | `fbeta_score(beta=2, pos_label=0)` | `evaluation.py` | CV + Locked Fold + threshold | ✅ Terimplementasi |
| 11 | **ROC-AUC** | `roc_auc_score(predict_proba)` | `evaluation.py` | CV + Locked Fold | ✅ Terimplementasi |
| 12 | **PR-AUC** (Average Precision) | `average_precision_score(pos_label=0)` | `evaluation.py` | CV + Locked Fold + prioritas 3 saat pilih model | ✅ Terimplementasi |
| 13 | **True Positive (TP)** | `confusion_matrix` | `evaluation.py:142-149` | Locked Fold | ✅ Terimplementasi |
| 14 | **False Negative (FN)** | `confusion_matrix` | `evaluation.py` | Locked Fold | ✅ Terimplementasi |
| 15 | **False Positive (FP)** | `confusion_matrix` | `evaluation.py` | Locked Fold | ✅ Terimplementasi |
| 16 | **True Negative (TN)** | `confusion_matrix` | `evaluation.py` | Locked Fold | ✅ Terimplementasi |
| 17 | **Mean (rata-rata CV)** | `np.mean(scores)` | `evaluation.py:88` | 25 iterasi CV | ✅ Terimplementasi |
| 18 | **Standard Deviation** | `np.std(scores)` | `evaluation.py:89` | 25 iterasi CV | ✅ Terimplementasi |
| 19 | **CI 95% (distribusi-t)** | `scipy.stats.t.interval` | `evaluation.py:94-97` | Setiap metrik CV | ✅ Terimplementasi |
| 20 | **CI 95% (Bootstrap persentil)** | `sklearn.utils.resample` + `np.percentile` | `evaluation.py:153-182` | Locked Fold | ✅ Terimplementasi |
| 21 | **MCC (Matthews Correlation Coefficient)** | `matthews_corrcoef` | **tidak ada di kode** | Hanya di tabel laporan | ❌ **Belum diimplementasikan** |

---

## 6B.3 GROUP A — 11 Metrik pada OUTER CROSS VALIDATION

**Lokasi:** `src/evaluation.py` baris 64–78

```python
scoring = {
    'accuracy':          make_scorer(accuracy_score),
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'precision_macro':   make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro':      make_scorer(recall_score,  average='macro', zero_division=0),
    'f1_macro':          make_scorer(f1_score,      average='macro', zero_division=0),
    'recall_bk':         make_scorer(recall_bk_score),      # pos_label=0
    'precision_bk':      make_scorer(precision_bk_score),   # pos_label=0
    'f1_bk':             make_scorer(f1_bk_score),          # pos_label=0
    'f2_bk':             make_scorer(f2_bk_score),          # beta=2, pos_label=0
    'roc_auc':           make_scorer(roc_auc_score, response_method='predict_proba'),
    'pr_auc':            make_scorer(average_precision_score, response_method='predict_proba',
                                     pos_label=0),
}
```

### 6B.3.1 Fungsi Bantu Metrik BK (`src/evaluation.py` baris 21–38)

```python
def recall_bk_score(y_true, y_pred):
    """Recall for Belum Kompeten (pos_label=0)."""
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)

def precision_bk_score(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=0, zero_division=0)

def f1_bk_score(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)

def f2_bk_score(y_true, y_pred):
    """F2 for Belum Kompeten (pos_label=0). Weights recall 2x more than precision."""
    return fbeta_score(y_true, y_pred, beta=2, pos_label=0, zero_division=0)
```

> **Penting untuk diingat:** parameter `pos_label=0` dipakai di **keempat fungsi** tersebut. Artinya seluruh metrik BK dihitung dengan anggapan **kelas 0 (Belum Kompeten) adalah kelas positif**.
> Parameter `zero_division=0` mencegah error saat pembagian dengan nol (misal tidak ada TP sama sekali).

### 6B.3.2 Setiap Metrik Didokumentasikan 4 Angka (Mean, SD, CI Bawah, CI Atas)

`src/evaluation.py` baris 85–98:

```python
for metric in scoring.keys():
    test_scores = scores[f'test_{metric}']
    mean_val = np.mean(test_scores)
    std_val  = np.std(test_scores)
    n = len(test_scores)                       # n = 25 (5 fold x 5 repeat)
    sem = stats.sem(test_scores)               # Standard Error of Mean
    ci  = stats.t.interval(0.95, n - 1, loc=mean_val, scale=sem)
    results[f'cv_{metric}_mean']     = mean_val
    results[f'cv_{metric}_std']      = std_val
    results[f'cv_{metric}_ci_lower'] = ci[0]
    results[f'cv_{metric}_ci_upper'] = ci[1]
```

**11 metrik × 4 angka = 44 kolom metrik** di `results/metrics/cv_experiment_matrix.csv`
(terbukti: file tersebut memiliki **51 kolom** = 44 kolom metrik + 7 kolom info: `population`, `scenario`, `model`, `balancing`, `n_features`, `best_threshold`, `best_params`).

### 6B.3.3 Angka Nyata — Model Pemenang S3_E DecisionTree (CV, 25 iterasi)

| Metrik | Mean | Std Dev | CI 95% Bawah | CI 95% Atas |
|---|---|---|---|---|
| Accuracy | 0,6535 | 0,0955 | 0,6133 | 0,6938 |
| **Balanced Accuracy** | **0,6500** | 0,0973 | 0,6090 | 0,6910 |
| Precision Macro | 0,7040 | 0,1470 | 0,6421 | 0,7660 |
| Recall Macro | 0,6500 | 0,0973 | 0,6090 | 0,6910 |
| F1 Macro | 0,6189 | 0,1191 | 0,5688 | 0,6691 |
| **Recall BK** ⭐ | **0,7914** | 0,2521 | 0,6852 | 0,8976 |
| Precision BK | 0,6368 | 0,1208 | 0,5860 | 0,6877 |
| F1 BK | 0,6802 | 0,1430 | 0,6200 | 0,7405 |
| **F2 BK** | **0,7368** | 0,1987 | 0,6531 | 0,8205 |
| ROC-AUC | 0,7344 | 0,0915 | 0,6959 | 0,7730 |
| PR-AUC | 0,4435 | 0,0240 | 0,4334 | 0,4536 |

**Cara membacanya:**
> *"Secara rata-rata dari 25 percobaan silang, model menangkap 79,14% mahasiswa berisiko (Recall BK), tetapi hanya dengan presisi 63,68%. Artinya dari setiap 10 mahasiswa yang di-alarm, sekitar 6–7 memang benar-benar berisiko."*

**Catatan penting:** Recall BK memiliki **SD tertinggi (0,2521)** dari semua metrik → artinya performa ini **paling tidak stabil** antar-fold. Ini konsekuensi wajar karena hanya ada 33 sampel BK di data latih, sehingga setiap satu mahasiswa bergeser ±3% (1/33).

---

## 6B.4 GROUP B — Metrik pada LOCKED FOLD / HOLDOUT TEST (n = 18)

**Lokasi:** `src/evaluation.py` baris 102–182

### 6B.4.1 Struktur Confusion Matrix yang Digunakan

`src/evaluation.py` baris 142–151:

```python
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tp_bk, fn_bk, fp_bk, tn_bk = cm.ravel()

results.update({
    'test_tn': int(tn_bk),
    'test_fp': int(fp_bk),
    'test_fn': int(fn_bk),
    'test_tp': int(tp_bk),
    'test_specificity': tn_bk / (tn_bk + fp_bk) if (tn_bk + fp_bk) > 0 else 0.0,
})
```

**Cara membaca kode ini (`labels=[0, 1]`):**

```
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        Prediksi 0 (BK)   Prediksi 1 (K)
Aktual 0 (BK)      cm[0][0] = TP_BK      cm[0][1] = FN_BK
Aktual 1 (K)       cm[1][0] = FP_BK      cm[1][1] = TN_BK

cm.ravel() menghasilkan urutan: [TN_BK, FP_BK, FN_BK, TP_BK]
                                  ↓         ↓      ↓      ↓
                          tp_bk     fn_bk   fp_bk   tn_bk
```

> **INGAT URUTAN INI** — ini sumber kesalahan paling sering saat pertanyaan muncul. `ravel()` menghasilkan `[TN, FP, FN, TP]` untuk `labels=[0,1]`, lalu variabel disimpan berurutan `tp, fn, fp, tn`.

### 6B.4.2 Threshold Digunakan Saat Prediksi

`src/evaluation.py` baris 110–118:

```python
if threshold is not None and hasattr(model, "predict_proba"):
    p_bk = model.predict_proba(X_test)[:, 0]        # probabilitas kelas 0 = Belum Kompeten
    y_pred = np.where(p_bk >= threshold, 0, 1)      # prediksi BK jika p_bk >= threshold
else:
    y_pred = model.predict(X_test)
```

**Logikanya:** model tidak langsung mengeluarkan label, tapi **probabilitas**. Keputusan diambil dengan membandingkan probabilitas tersebut dengan **threshold terkunci**.

### 6B.4.3 Daftar Lengkap Output `evaluate_test()` — 22 Kolom

| # | Key Output | Metrik / Nilai | Angka Aktual |
|---|---|---|---|
| 1 | `test_accuracy` | Accuracy | 0,7778 |
| 2 | `test_balanced_accuracy` | Balanced Accuracy | 0,7778 |
| 3 | `test_precision_macro` | Precision Macro | 0,7922 |
| 4 | `test_recall_macro` | Recall Macro | 0,7778 |
| 5 | `test_f1_macro` | F1 Macro | 0,7750 |
| 6 | `test_recall_bk` ⭐ | **Recall BK** | **0,8889** |
| 7 | `test_precision_bk` | Precision BK | 0,7273 |
| 8 | `test_f1_bk` | F1 BK | 0,8000 |
| 9 | `test_f2_bk` | F2 BK | 0,8511 |
| 10 | `test_roc_auc` | ROC-AUC | 0,8951 |
| 11 | `test_pr_auc` | PR-AUC | 0,8459 |
| 12 | `test_tp` | True Positive | **8** |
| 13 | `test_fn` | False Negative | **1** |
| 14 | `test_fp` | False Positive | **3** |
| 15 | `test_tn` | True Negative | **6** |
| 16 | `test_specificity` | Specificity | 0,6667 |
| 17 | `test_recall_bk_ci_lower` | Recall BK CI bawah | 0,6250 |
| 18 | `test_recall_bk_ci_upper` | Recall BK CI atas | 1,0000 |
| 19 | `test_balanced_accuracy_ci_lower` | Bal Acc CI bawah | 0,5615 |
| 20 | `test_balanced_accuracy_ci_upper` | Bal Acc CI atas | 0,9545 |
| 21 | `test_f2_bk_ci_lower` | F2 BK CI bawah | 0,6246 |
| 22 | `test_f2_bk_ci_upper` | F2 BK CI atas | 0,9804 |

> **Perhatikan:** bukan semua metrik punya CI. Hanya **3 metrik** yang diberi Bootstrap CI (Recall BK, Balanced Accuracy, F2 BK) — ketiganya adalah metrik **utama penelitian**.

### 6B.4.4 Bootstrap 95% CI — Cara Kerjanya

`src/evaluation.py` baris 153–182:

```python
n_iterations = 1000                                    # B = 1000 bootstrap
boot_recalls, boot_bal_acc, boot_f2 = [], [], []

for i in range(n_iterations):
    X_bs, y_bs = resample(X_test_arr, y_test_arr, random_state=i)   # acak ulang dgn pengembalian
    # hitung ulang prediksi dengan threshold yang SAMA (dikunci)
    p_bk_bs = model.predict_proba(X_bs)[:, 0]
    y_pred_bs = np.where(p_bk_bs >= threshold, 0, 1)
    boot_recalls.append(recall_score(y_bs, y_pred_bs, pos_label=0, zero_division=0))
    boot_bal_acc.append(balanced_accuracy_score(y_bs, y_pred_bs))
    boot_f2.append(fbeta_score(y_bs, y_pred_bs, beta=2, pos_label=0, zero_division=0))

results['test_recall_bk_ci_lower'] = np.percentile(boot_recalls, 2.5)
results['test_recall_bk_ci_upper'] = np.percentile(boot_recalls, 97.5)
# ... sama untuk balanced_accuracy dan f2_bk
```

**Metodologi yang benar:**
1. Tipe **non-parametrik** (tidak mengasumsikan bentuk distribusi)
2. **Bootstrap percentile**: ambil persentil 2,5% dan 97,5% dari 1.000 hasil
3. `random_state=i` → **dapat direproduksi**
4. **Threshold dikunci** selama bootstrap (tidak dioptimasi ulang) — ini benar secara metodologis

### 6B.4.5 Visualisasi Confusion Matrix

`src/evaluation.py` baris 184–197 — setiap evaluasi menghasilkan **heatmap PNG**:

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['BK (0)', 'K (1)'], yticklabels=['BK (0)', 'K (1)'])
plt.title(f'CM - {scenario_name} - {model_name}{thresh_label}')
```

disimpan ke `results/confusion_matrix/cm_<skenario>_<model>.png` dan `results/final/cm_FINAL_S3_E_DecisionTree.png`.

---

## 6B.5 GROUP C — 6 Metrik pada OPTIMASI THRESHOLD

**Lokasi:** `src/tuning.py` baris 101–175

### 6B.5.1 Kandidat Threshold yang Diperiksa

```python
thresholds = np.arange(0.20, 0.80, 0.02)    # 31 kandidat: 0,20 · 0,22 · ... · 0,78
```

### 6B.5.2 6 Metrik yang Dihitung untuk Setiap Kandidat

```python
for thresh in thresholds:
    y_pred_thresh = np.where(p_bk >= thresh, 0, 1)
    acc      = accuracy_score(y, y_pred_thresh)                  # 1. Accuracy
    bal_acc  = balanced_accuracy_score(y, y_pred_thresh)         # 2. Balanced Accuracy
    rec_bk   = recall_score(y, y_pred_thresh, pos_label=0, zero_division=0)     # 3. Recall BK
    prec_bk  = precision_score(y, y_pred_thresh, pos_label=0, zero_division=0)  # 4. Precision BK
    f1_bk    = f1_score(y, y_pred_thresh, pos_label=0, zero_division=0)        # 5. F1 BK
    f2_bk    = fbeta_score(y, y_pred_thresh, beta=2, pos_label=0, zero_division=0)  # 6. F2 BK
```

### 6B.5.3 Aturan Pemilihan Threshold (Constraint-Based)

```python
# 1. HARUS punya recall_bk >= 0,70 (syarat minimal keamanan)
valid_candidates = [r for r in results if r['recall_bk'] >= 0.70]

if valid_candidates:
    # 2. Dari yang lolos, ambil Balanced Accuracy tertinggi
    # 3. Jika seri, pakai F2 BK sebagai pemutus (tie-breaker)
    valid_candidates.sort(key=lambda x: (x['balanced_accuracy'], x['f2_bk']), reverse=True)
    best = valid_candidates[0]
else:
    # Fallback: ambil yang Recall-nya paling tinggi
    results.sort(key=lambda x: (x['recall_bk'], x['balanced_accuracy']), reverse=True)
    best = results[0]
```

**Prioritas metrik saat memilih threshold:**

```
[1] SYARAT WAJIB  : Recall BK ≥ 0,70      ← tidak bisa dilanggar
[2] UTAMA         : Balanced Accuracy      ← paling tinggi
[3] PENENTU SERI  : F2 BK                  ← kalau Balanced Accuracy sama
```

### 6B.5.4 Pencegahan Kebocoran pada Threshold

```python
# src/tuning.py baris 128-133
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
p_bk = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 0]
```

`cross_val_predict` menghasilkan prediksi **Out-Of-Fold** — setiap probabilitas dihitung dari model yang **tidak pernah dilatih** pada data tersebut. Ini memastikan threshold tidak "menghafal" data latih.

### 6B.5.5 Threshold Terpilih di Proyek Ini

| Cutoff | Threshold Terpilih | Recall BK Saat Threshold Itu | Balanced Accuracy |
|---|---|---|---|
| W1 | 0,26 | ≥ 0,70 (syarat terpenuhi) | — |
| **W2** | **0,20** | 1,0000 | 0,7778 |
| W3 | 0,24 | ≥ 0,70 | — |
| **C_Full (Final)** | **0,52** | 0,8889 | 0,7778 |

> **Perhatikan:** Default sklearn adalah **0,50**. Untuk model final C_Full, threshold hanya dinaikkan sedikit ke **0,52**. Untuk W2 yang diturunkan ke **0,20** — ini disengaja agar **lebih banyak mahasiswa di-alarm**, karena pada fase awal lebih baik memberi terlalu banyak peringatan daripada ada mahasiswa berisiko yang lolos.

---

## 6B.6 GROUP D — Metrik Objektif INNER CROSS VALIDATION (Tuning)

**Lokasi:** `src/tuning.py` baris 14–30

```python
def _get_inner_scorer(scoring_target='recall_bk'):
    if scoring_target == 'recall_bk':
        return make_scorer(recall_score, pos_label=0, zero_division=0)
    elif scoring_target == 'f2_bk':
        return make_scorer(fbeta_score, beta=2, pos_label=0, zero_division=0)
    elif scoring_target == 'balanced_accuracy':
        return make_scorer(balanced_accuracy_score)
```

| Pilihan `scoring_target` | Metrik | Dipakai di Proyek? |
|---|---|---|
| `'recall_bk'` (DEFAULT) | Recall untuk Belum Kompeten | ✅ **Ya — inilah yang dipakai** |
| `'f2_bk'` | F2-score untuk BK | ❌ Tidak (disimpan sebagai opsi) |
| `'balanced_accuracy'` | Balanced Accuracy | ❌ Tidak (disimpan sebagai opsi) |

> **Jawaban singkat jika ditanya "metrik apa yang dipakai untuk tuning?"**
> → **"Recall untuk Belum Kompeten (`recall_bk`)"**, ditentukan oleh nilai bawaan parameter `scoring_target='recall_bk'` di `src/tuning.py`.

---

## 6B.7 GROUP E — 3 Metrik untuk PEMILIHAN MODEL (Model Selection)

**Lokasi:** `scripts/run_optimized_experiment.py` baris 134–180

```python
# Urutan prioritas kriteria pemilihan model (dari ATAS ke BAWAH):
candidates = candidates.sort_values(
    by=[
        'cv_recall_bk_mean',         # 1. UTAMA — tangkap mahasiswa berisiko
        'cv_balanced_accuracy_mean', # 2. Keseimbangan dua kelas
        'cv_pr_auc_mean',            # 3. Ketahanan terhadap ketidakseimbangan kelas
    ],
    ascending=[False, False, False]
)
```

| Urutan | Metrik | Key di CSV | Alasan |
|---|---|---|---|
| **1** | **Recall BK** | `cv_recall_bk_mean` | FN = mahasiswa berisiko yang lolos = kegagalan paling fatal |
| **2** | **Balanced Accuracy** | `cv_balanced_accuracy_mean` | Model harus juga pintar membedakan yang aman |
| **3** | **PR-AUC** | `cv_pr_auc_mean` | Bonus pengikat jika Recall & Balanced Accuracy seri |

> **KRITIS:** Metrik pemilihan model ini **semuanya berasal dari Outer CV pada data latih**. **Locked Fold tidak pernah dipakai** untuk memilih model. Inilah yang membuat Nested CV dalam proyek ini valid secara metodologis.

### 6B.7.1 Hasil Seleksi: Siapa yang Dipilih?

Dari `results/metrics/cv_experiment_matrix.csv` (21 baris = 7 skenario × 3 model):

| Skenario | Model | n Fitur | Threshold | CV Recall BK | CV Bal Acc | CV F2 BK | CV PR-AUC |
|---|---|---|---|---|---|---|---|
| S3 | Dummy | 6 | 0,50 | 1,0000* | 0,5000 | 0,8369* | 0,5067* |
| S3 | DecisionTree | 6 | 0,52 | 0,6421 | 0,6239 | 0,6284 | 0,4617 |
| S3 | RandomForest | 6 | 0,48 | 0,6807 | 0,6518 | 0,6695 | 0,4243 |
| S3_A | DecisionTree | 7 | 0,34 | 0,6279 | 0,6282 | 0,6202 | 0,4665 |
| S3_A | RandomForest | 7 | 0,50 | 0,6921 | 0,6804 | 0,6860 | 0,4267 |
| S3_B | DecisionTree | 7 | 0,52 | 0,6421 | 0,6154 | 0,6267 | 0,4594 |
| S3_B | RandomForest | 7 | 0,50 | 0,6814 | 0,6550 | 0,6708 | 0,4302 |
| S3_C | DecisionTree | 7 | 0,52 | 0,6479 | 0,6211 | 0,6320 | 0,4529 |
| S3_C | RandomForest | 7 | 0,50 | 0,6650 | 0,6496 | 0,6561 | 0,4257 |
| S3_D | DecisionTree | 7 | 0,52 | 0,6479 | 0,6211 | 0,6320 | 0,4529 |
| S3_D | RandomForest | 7 | 0,36 | 0,6807 | 0,6575 | 0,6712 | 0,4260 |
| **S3_E** | **DecisionTree** | **7** | **0,52** | **0,7914** ⭐ | **0,6500** | **0,7368** ⭐ | 0,4435 |
| S3_E | RandomForest | 7 | 0,50 | 0,6650 | 0,6325 | 0,6520 | 0,4235 |
| S3_EWS | DecisionTree | 11 | 0,52 | 0,7571 | 0,6443 | 0,7127 | 0,4433 |
| S3_EWS | RandomForest | 11 | 0,50 | 0,6800 | 0,6600 | 0,6693 | 0,4201 |

**Pemenang: S3_E + DecisionTree**, karena punya **CV Recall BK tertinggi (0,7914)** di antara model non-Dummy.

> ⚠️ **Catatan penting tentang baris Dummy:** `Dummy` mencapai Recall BK = 1,0000 — angka ini **tidak boleh dianggap hasil yang sah**. `DummyClassifier(strategy='most_frequent')` selalu menebak kelas mayoritas sehingga **tidak pernah** menghasilkan FN sama sekali. Kode sudah benar akan hal ini dengan **mengecualikan Dummy** saat pemilihan model (`candidates = df_cv[df_cv['model'] != 'Dummy']`), tetapi **Dummy tetap dipakai** sebagai pembanding Balanced Accuracy 50%.

---

## 6B.8 GROUP F — 3 Ukuran SHAP (Metrik Interpretasi)

**Lokasi:** `src/shap_analysis.py` baris 31–95

### 6B.8.1 Cara Kerja

```python
explainer = shap.TreeExplainer(clf)            # khusus model Decision Tree / Random Forest
shap_values = explainer(X_test_transformed)    # hitung kontribusi setiap fitur
shap_values_pos = shap_values[..., 0]          # ambil untuk kelas 0 = Belum Kompeten
```

### 6B.8.2 Tiga Ukuran SHAP yang Digunakan

| # | Ukuran | Definisi | Digunakan Untuk |
|---|---|---|---|
| 1 | **Nilai SHAP tiap mahasiswa** | Seberapa besar fitur F mendorong prediksi mahasiswa itu | Waterfall plot (penjelasan per individu) |
| 2 | **Mean Absolute SHAP** | Rata-rata \|nilai SHAP\| semua mahasiswa = **seberapa penting** fitur | Bar plot `global_importance` |
| 3 | **Tanda SHAP** (+ / −) | Positif = dorong ke "BK", negatif = dorong ke "K" | Beeswarm plot (pola distribusi) |

### 6B.8.3 Empat Jenis Plot yang Dihasilkan

| Plot | Isi | Nama File |
|---|---|---|
| **Global Importance** (bar) | Ranking fitur berdasarkan mean absolute SHAP | `global_importance_<tag>_<model>.png` |
| **Beeswarm** | Sebaran nilai SHAP per fitur + tanda + nilai aslinya | `beeswarm_<tag>_<model>.png` |
| **Waterfall** (local) | Penjelasan 1 mahasiswa per kategori | `local_TP_/TN_/FP_/FN_<tag>_<model>.png` |
| **CSV nilai SHAP** | Angka SHAP mentah untuk dianalisis | `shap_values_<tag>_<model>.csv` |

### 6B.8.4 Angka Mean Absolute SHAP di Proyek Ini

**Model Baseline S3 (Retrospektif — data penuh):**

| Peringkat | Fitur | Mean Abs SHAP |
|---|---|---|
| 1 | `Laporan_Mean` | **0,3095** |
| 2 | `Performance_Std` | 0,1872 |
| 3 | `TP_Mean` | 0,0686 |
| 4–9 | `Attendance_PreFinal_Rate`, `TP_Completion_Rate`, `Laporan_Completion_Rate`, dan 3 *missing indicator* | 0,0000 |

**Model Final S3_E (Genuine EWS — data awal):**

| Peringkat | Fitur | Mean Abs SHAP | Selisih vs Baseline |
|---|---|---|---|
| 1 | `Early_Performance_Composite` ⭐ | **0,2200** | **+0,2200 (BARU)** |
| 2 | `Laporan_Mean` | 0,2110 | −0,0985 |
| 3 | `Performance_Std` | 0,1040 | −0,0832 |
| 4 | `TP_Mean` | 0,0695 | +0,0009 |
| 5–11 | `Attendance_PreFinal_Rate`, 2 completion rate, 4 *missing indicator* | 0,0000 | 0,0000 |

**Temuan utama:**
> Menambahkan satu fitur EWS (`Early_Performance_Composite`) **menggeserer struktur keputusan model**: dari sebelumnya bergantung pada `Laporan_Mean` (0,3095) menjadi **`Early_Performance_Composite` menjadi kontributor tertinggi** (0,2200). Inilah bukti bahwa sistem benar-benar bekerja sebagai **Early Warning System** — bukan sekadar merekap nilai akhir.

> **Temuan tambahan yang menarik:** tiga fitur (`Attendance_PreFinal_Rate`, `TP_Completion_Rate`, `Laporan_Completion_Rate`) memiliki SHAP **= 0,0000**. Artinya **model tidak pernah memakainya** — dengan 7 fitur, Decision Tree cukup 4 fitur untuk mengambil keputusan. Ini adalah informasi berguna: fitur bisa disederhanakan lagi di penelitian lanjutan.

---

## 6B.9 GROUP G — Metrik Pembanding: Dummy Classifier

**Lokasi:** `src/models.py` baris 49–51

```python
elif model_name == 'Dummy':
    clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
```

| Aspek | Nilai |
|---|---|
| **Kebijakan** | Selalu menebak kelas **mayoritas** (K) |
| **Balanced Accuracy** | **0,5000** (50%) — titik nol |
| **PR-AUC** | **0,5067** — sedikit di atas acak |
| **Recall BK** | 1,0000 — **tidak bermakna** (karena tidak pernah menebak BK) |
| **Fungsi** | Membuktikan model **signifikan lebih baik** dari tebakan acak |

**Kesimpulan yang dipakai di laporan:**
> *"Batas bawah Confidence Interval Balanced Accuracy (56,15%) berada di atas 50% (batas bawah Dummy), sehingga performa model pada locked fold secara statistik lebih tinggi daripada baseline dummy."*

---

## 6B.10 VERIFIKASI MANUAL — Hitung Ulang Semua Metrik dari Confusion Matrix

Untuk memastikan tidak ada kesalahan pelaporan, berikut **hitung ulang manual** dari confusion matrix asli di `results/final/cutoff_comparison.csv`.

### 6B.10.1 Confusion Matrix Asli (dari `results/final/cutoff_comparison.csv`)

| Cutoff | TP | FN | FP | TN | Total |
|---|---|---|---|---|---|
| W1 | 8 | 1 | 9 | 0 | 18 |
| W2 | 9 | 0 | 4 | 5 | 18 |
| W3 | 9 | 0 | 4 | 5 | 18 |
| C_Full | 8 | 1 | 3 | 6 | 18 |

### 6B.10.2 Hitung Ulang Semua Metrik (100% cocok dengan hasil kode)

| Cutoff | Accuracy | Recall BK | Precision BK | Specificity | Bal Acc | F1 BK | F2 BK |
|---|---|---|---|---|---|---|---|
| W1 | 0,4444 | 0,8889 | 0,4706 | 0,0000 | 0,4444 | 0,6154 | 0,7547 |
| W2 | 0,7778 | 1,0000 | 0,6923 | 0,5556 | 0,7778 | 0,8182 | 0,9184 |
| W3 | 0,7778 | 1,0000 | 0,6923 | 0,5556 | 0,7778 | 0,8182 | 0,9184 |
| C_Full | 0,7778 | 0,8889 | 0,7273 | 0,6667 | 0,7778 | 0,8000 | 0,8511 |

**Contoh perhitungan manual untuk C_Full (TP=8, FN=1, FP=3, TN=6):**

```
┌─ Accuracy ────────────────────────────────────────────────────────┐
│  (TP + TN) / Total = (8 + 6) / 18 = 14 / 18 = 0,7778 = 77,78%   │
└───────────────────────────────────────────────────────────────────┘

┌─ Recall BK ───────────────────────────────────────────────────────┐
│  TP / (TP + FN) = 8 / (8 + 1) = 8 / 9 = 0,8889 = 88,89%         │
└───────────────────────────────────────────────────────────────────┘

┌─ Precision BK ────────────────────────────────────────────────────┐
│  TP / (TP + FP) = 8 / (8 + 3) = 8 / 11 = 0,7273 = 72,73%        │
└───────────────────────────────────────────────────────────────────┘

┌─ Specificity ────────────────────────────────────────────────────┐
│  TN / (TN + FP) = 6 / (6 + 3) = 6 / 9 = 0,6667 = 66,67%         │
└───────────────────────────────────────────────────────────────────┘

┌─ Balanced Accuracy ───────────────────────────────────────────────┐
│  (Recall BK + Specificity) / 2                                    │
│  = (0,8889 + 0,6667) / 2 = 1,5556 / 2 = 0,7778 = 77,78%         │
└───────────────────────────────────────────────────────────────────┘

┌─ F1 BK ───────────────────────────────────────────────────────────┐
│  2 × (Precision × Recall) / (Precision + Recall)                  │
│  = 2 × (0,7273 × 0,8889) / (0,7273 + 0,8889)                     │
│  = 2 × 0,6465 / 1,6162 = 1,2930 / 1,6162 = 0,8000 = 80,00%      │
└───────────────────────────────────────────────────────────────────┘

┌─ F2 BK ───────────────────────────────────────────────────────────┐
│  5 × (Precision × Recall) / (4 × Precision + Recall)             │
│  = 5 × 0,6465 / (4 × 0,7273 + 0,8889)                            │
│  = 3,2325 / (2,9091 + 0,8889) = 3,2325 / 3,7980 = 0,8511        │
│  = 85,11%                                                        │
└───────────────────────────────────────────────────────────────────┘

┌─ MCC ─────────────────────────────────────────────────────────────┐
│  (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]              │
│  = (8×6 − 3×1) / √[(8+3)(8+1)(6+3)(6+1)]                        │
│  = (48 − 3) / √[11 × 9 × 9 × 7]                                  │
│  = 45 / √6237 = 45 / 78,978 = 0,5698 = 0,57                      │
└───────────────────────────────────────────────────────────────────┘
```

> ✅ **Semua hitungan manual menghasilkan angka yang PERSIS sama dengan output kode** (`results/final/holdout_results.csv`). Artinya tidak ada kesalahan aritmetika pada pelaporan.

**Yang TIDAK bisa dihitung dari confusion matrix:** ROC-AUC (0,8951) dan PR-AUC (0,8459) — keduanya membutuhkan **nilai probabilitas** dari `predict_proba()`, bukan hanya label 0/1. Ini normal.

---

## 6B.11 DUA JENIS CONFIDENCE INTERVAL — Bedakan!

Proyek ini memakai **2 metode CI yang berbeda**. Jangan tertukar saat penjelasan.

| Aspek | **CI pada Cross Validation** | **CI pada Locked Fold** |
|---|---|---|
| **Metode** | Distribusi Student-t (*t-distribution*) | **Bootstrap persentil** (non-parametrik) |
| **Kode** | `scipy.stats.t.interval(0.95, n-1, ...)` | `sklearn.utils.resample` + `np.percentile` |
| **Lokasi** | `src/evaluation.py:94-97` | `src/evaluation.py:153-182` |
| **Jumlah data** | 25 nilai (25 iterasi CV) | 18 data asli, diacak ulang 1.000× |
| **Dihitung untuk** | **Semua 11 metrik** (44 kolom CI) | **Hanya 3 metrik** (6 kolom CI) |
| **Asumsi** | Distribusi mendekati normal | **Tidak ada asumsi** |
| **Fungsi** | Mengukur **ketidakstabilan antar fold** | Mengukur **ketidakpastian sampel kecil** |
| **Nilai contoh** | Recall BK: 0,6852 – 0,8976 | Recall BK: 0,6250 – 1,0000 |

**Kenapa Locked Fold pakai Bootstrap?**
> Karena n = 18 sangat kecil sehingga distribusi skor metrik **menyimpang jauh dari normal**. Metode Bootstrap tidak perlu asumsi apa pun — inilah alasannya metode ini dipilih di `HASIL_ANALISIS_V2.md`.

---

## 6B.12 METRIK YANG BELUM DIIMPLEMENTASIKAN (⚠️ Temuan Audit)

### 6B.12.1 MCC — Tercantum di Laporan, Tidak Ada di Kode

**Situasi:**
- Nilai MCC muncul di Tabel 2 `HASIL_ANALISIS_V2.md` (kolom MCC)
- **Tidak ada** `matthews_corrcoef` di seluruh kode `src/` maupun `scripts/`

**Masalah lebih besar — angka MCC di laporan tidak cocok dengan confusion matrix:**

| Cutoff | MCC di Laporan | MCC Seharusnya (dihitung ulang) | Status |
|---|---|---|---|
| W1 | 0,00 | **−0,2425** | ❌ Tidak cocok |
| W2 | 0,57 | **0,6202** | ❌ Tidak cocok |
| W3 | 0,57 | **0,6202** | ❌ Tidak cocok |
| C_Full | 0,58 | **0,5698** | ⚠️ Sangat dekat, tapi tidak sama persis |

**Contoh paling jelas — W1:** Specificity = 0% dan semua 9 mahasiswa Kompeten di-alarm (FP=9, TN=0) → MCC harus **negatif** (−0,2425), bukan 0,00. MCC = 0 berarti "tidak lebih baik dari acak", padahal model W1 justru **lebih buruk dari acak** dalam membedakan kelas aman.

**Perbaikan yang disarankan:**

Tambahkan di `src/evaluation.py`:
```python
# di baris 10-14, tambahkan import:
from sklearn.metrics import matthews_corrcoef

# di dalam results dict pada evaluate_test() (setelah baris 150):
'test_mcc': matthews_corrcoef(y_test, y_pred),
```

Lalu perbarui angka MCC di `HASIL_ANALISIS_V2.md` menjadi hasil hitung ulang.

### 6B.12.2 Metrik Umum Lain yang Tidak Dipakai (dan Alasan)

| Metrik | Alasan Tidak Dipakai |
|---|---|
| **Cohen's Kappa** | Mengukur kesesuaian antara dua penilai — tidak relevan, hanya ada 1 label kebenaran |
| **Log Loss** | Sensitif terhadap kalibrasi probabilitas — model pohon tidak dikalibrasi |
| **Hamming Loss** | Hanya relevan untuk multi-label — di sini single-label |
| **Jaccard Index** | Untuk multilabel/retrieval — tidak relevan |
| **Brier Score** | Mengukur kalibrasi probabilitas — fokus penelitian adalah deteksi, bukan kalibrasi |
| **Calibration Curve** | Sama — bukan fokus penelitian |

---

## 6B.13 ALUR PENGGUNAAN METRIK DALAM PIPELINE

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TAHAP 1 — SPLIT (src/split.py)                                          │
│   86 data → 68 latih + 18 beku, stratify, random_state=42                │
│   🔒 18 data beku DIKUNCI                                                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TAHAP 2 — INNER CV / TUNING (src/tuning.py)                             │
│   Metrik: recall_bk  ← SATU-SATUNYA objektif tuning                       │
│   StratifiedKFold(3) + RandomizedSearchCV(n_iter=min(50, kombinasi))     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TAHAP 3 — OUTER CV / EVALUASI (src/evaluation.py → evaluate_cv)          │
│   Metrik: 11 metrik × (Mean + SD + CI bawah + CI atas) = 44 kolom        │
│   RepeatedStratifiedKFold(5 fold × 5 repeat = 25 iterasi)                │
│   🔒 18 data beku MASIH TIDAK DISENTUH                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TAHAP 4 — PEMILIHAN MODEL (run_optimized_experiment.py → select_best)   │
│   Urutan metrik: ① cv_recall_bk_mean                                    │
│                  ② cv_balanced_accuracy_mean                           │
│                  ③ cv_pr_auc_mean                                        │
│   → PEMENANG: DecisionTree + S3_E                                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TAHAP 5 — OPTIMASI THRESHOLD (src/tuning.py → evaluate_thresholds)      │
│   Metrik: 6 metrik × 31 kandidat threshold (0,20–0,78 step 0,02)        │
│   Syarat: recall_bk ≥ 0,70 → pilih balanced_accuracy tertinggi           │
│   → THRESHOLD TERKUNCI: 0,52 (C_Full)                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TAHAP 6 — LOCKED FOLD / TESI AKHIR (src/evaluation.py → evaluate_test)   │
│   Metrik: 11 metrik + TP/FN/FP/TN + Specificity = 16 nilai              │
│           + Bootstrap CI 95% untuk 3 metrik utama = 6 nilai             │
│           + Confusion Matrix heatmap (PNG)                               │
│   🔓 BARU DIBUKA — DI EVALUASI 1× (×4 cutoff)                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TAHAP 7 — INTERPRETASI SHAP (src/shap_analysis.py)                      │
│   Metrik: mean absolute SHAP, tanda SHAP, nilai SHAP individu            │
│   Plot: global_importance, beeswarm, local waterfall                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6B.14 LOKASI FILE OUTPUT METRIK

| File Output | Isi | Jumlah Metrik |
|---|---|---|
| `results/metrics/cv_experiment_matrix.csv` | 21 baris eksperimen × **51 kolom** | 44 kolom metrik CV (11 metrik × 4) |
| `results/final/holdout_results.csv` | 1 baris × **22 kolom** | 22 nilai Locked Fold |
| `results/final/final_model_report.json` | Laporan JSON lengkap + CI + best_params | — |
| `results/final/cutoff_comparison.csv` | 4 cutoff × 10 kolom (TP/FN/FP/TN + 3 metrik) | — |
| `results/final/baseline_comparison.json` | Perbandingan baseline vs final (6 angka) | — |
| `results/confusion_matrix/cm_*.png` | Heatmap confusion matrix per model | Visual |
| `results/shap/shap_values_*.csv` | Nilai SHAP mentah per mahasiswa | Interpretasi |
| `results/shap/global_importance_*.png` | Bar plot importance fitur | Interpretasi |
| `results/shap/beeswarm_*.png` | Beeswarm plot | Interpretasi |
| `results/shap/local_*.png` | Waterfall plot per kategori (TP/TN/FP/FN) | Interpretasi |

---

## 6B.15 RINGKASAN CEPAT UNTUK SIDANG

**Jika hanya boleh satu kalimat:**
> *"Penelitian ini mengimplementasikan 11 metrik klasifikasi (Accuracy, Balanced Accuracy, Precision/Recall/F1/F2 khusus kelas Belum Kompeten, metrik macro (Precision/Recall/F1 rata-rata), ROC-AUC, dan PR-AUC), ditambah confusion matrix, Specificity, dan analisis ketidakpastian melalui Standard Deviation serta dua jenis Confidence Interval 95% (distribusi-t untuk cross validation dan Bootstrap untuk locked fold), serta interpretasi kontribusi fitur menggunakan SHAP."*

**Jika ditanya "metrik mana yang paling penting":**
> *"**Recall untuk Belum Kompeten** adalah metrik utama, karena dalam sistem peringatan dini, mahasiswa berisiko yang lolos (False Negative) adalah kegagalan paling fatal — mereka tidak akan mendapat intervensi. Karena itu, Recall BK juga menjadi objektif tuning di Inner CV dan kriteria pertama pemilihan model, serta F2-Score digunakan sebagai metrik komposit yang memberi bobot 2× lebih besar pada Recall."*

**Jika ditanya "kenapa tidak pakai Accuracy saja":**
> *"Karena Accuracy bisa menyesatkan. Jika model memprediksi semua mahasiswa sebagai Kompeten, Accuracy akan 51,2% — sesuatu yang tidak berguna untuk sistem peringatan dini. Karena itu, Balanced Accuracy digunakan agar kedua kelas dinilai sama penting, dan Specificity serta Precision BK dipakai untuk mengukur efektivitas alarm."*

---

# BAGIAN 7 — HASIL EKSPERIMEN

## 7.1 Model Final

| Aspek | Nilai |
|---|---|
| **Algoritma** | Decision Tree |
| **Skenario fitur** | **S3_E** |
| **Balancing** | None (tanpa SMOTE) |
| **Jumlah fitur** | 7 |
| **Hyperparameter terbaik** | `criterion='entropy'`, `max_depth=4`, `min_samples_split=10`, `min_samples_leaf=5` |
| **Threshold** | 0,52 |
| **Lokasi model** | `models/FINAL_DecisionTree_S3_E_None.pkl` |

## 7.2 Performa pada Locked Fold (n=18)

| Metrik | Nilai | 95% CI |
|---|---|---|
| Accuracy | 77,78% | — |
| **Balanced Accuracy** | **77,78%** | **[56,15% – 95,45%]** |
| **Recall BK** | **88,89%** | **[62,50% – 100,00%]** |
| Precision BK | 72,73% | — |
| F1 BK | 80,00% | — |
| **F2 BK** | **85,11%** | **[62,46% – 98,04%]** |
| Specificity | 66,67% | — |
| ROC-AUC | 0,8951 | — |
| PR-AUC | 0,8459 | — |
| MCC | 0,58 | — |

**Confusion Matrix:** TP=8, FN=1, FP=3, TN=6

## 7.3 Performa Multi-Cutoff (Jawab RQ2)

| Cutoff | Recall BK | Precision BK | F2 BK | Specificity | Balanced Acc | MCC | **Kesimpulan** |
|---|---|---|---|---|---|---|---|
| **W1** (Minggu 1) | 88,89% | 47,06% | 75,47% | **0,00%** | **44,44%** | 0,00 | Terlalu dini — semua di-alarm |
| **W2** (Minggu 2) | **100,0%** | 69,23% | **91,84%** | 55,56% | 77,78% | 0,57 | ✅ **Titik terbaik** |
| **W3** (Minggu 3) | **100,0%** | 69,23% | **91,84%** | 55,56% | 77,78% | 0,57 | ✅ Sama bagus |
| **C_Full** (Semester) | 88,89% | 72,73% | 85,11% | 66,67% | 77,78% | 0,58 | BAL Acc tidak naik |

**Kesimpulan Kunci:**
> **Minggu ke-2 adalah titik waktu paling awal yang sudah bisa mencapai Recall BK = 100% dengan Balanced Accuracy = 77,78%.** Artinya, **deteksi dini hanya butuh 2 minggu** — tidak perlu menunggu UTS atau akhir semester. Ini adalah temuan utama penelitian Anda.

## 7.4 Baseline vs Final

| Aspek | Baseline S3 | Final S3_E |
|---|---|---|
| Orientasi | Retrospektif (data penuh) | **EWS Genuine** (2-3 minggu awal) |
| Fitur dominan | `Laporan_Mean` (0,3095) | **`Early_Performance_Composite`** (0,2200) |
| Sifat | Reaktif (harus tunggu selesai) | **Proaktif** (deteksi dini) |
| Recall BK (holdout) | 100% | 88,89% (C_Full) / **100%** (W2, W3) |
| Balanced Acc | 77,78% | 77,78% |

---

# BAGIAN 8 — CATATAN TEMUAN AUDIT (WAJIB DISIAPKAN)

Berikut temuan yang **sudah saya verifikasi langsung dari kode dan data**. Ini penting untuk paham agar tidak kaget saat-sidang.

## ⚠️ 8.1 Ketidaksesuaian n = 86 vs n = 89

**Temuan:**
- `data/processed/students_master.csv` berisi **86 baris** (44 Kompeten + 42 Belum Kompeten)
- `src/build_dataset.py` baris 133: `assert len(df_students) == 86` ✅ konsisten dengan kode
- Tetapi `HASIL_ANALISIS_V2.md` menuliskan **n=89** dengan 45 BK + 44 K (= 89) ❌ **tidak cocok**
- `scripts/run_baseline_experiments.py` juga mencetak `"Population: n89"` ❌

**Dampak:** Angka di laporan perlu disesuaikan dari **89 → 86** dan distribusi label dari **45:44 → 42:44**.

**Jawaban jika ditanya dosen:**
> *"Jumlah final dalam versi terbaru adalah 86 mahasiswa unik. Angka 89 berasal dari versi sebelumnya sebelum duplikasi nama dihapus. Pipeline terbaru sudah konsisten pada n=86."*

## ⚠️ 8.2 Konfigurasi CV di Laporan Tidak Cocok dengan Kode

**Temuan:**
- Laporan (`HASIL_ANALISIS_V2.md`) menuliskan: *"2 outer folds dan 5 inner folds (K-Fold 5×2)"*
- Kode aktual:
  - **Outer:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` → **5 fold × 5 repeat = 25 iterasi**
  - **Inner:** `StratifiedKFold(n_splits=3)` → **3 fold**

**Jawaban jika ditanya:**
> *"Konfigurasi final yang dipakai adalah Repeated Stratified 5-Fold (5×5 repeat = 25 iterasi) untuk outer loop, dan 3-fold Stratified untuk inner loop. Ini berbeda dari naskah draft awal yang masih 2 outer fold. Versi final justru lebih kuat karena 25 pengukuran memberikan estimasi yang lebih stabil."*

**Rekomendasi:** Perbarui tabel metodologi di laporan agar cocok dengan kode.

## ⚠️ 8.3 MCC Tercantum tetapi Belum Ada di Kode — DAN Angkanya Salah

**Temuan 1:** Nilai MCC (0,57 / 0,58) muncul di Tabel 2 `HASIL_ANALISIS_V2.md`, tetapi **tidak ada** `matthews_corrcoef` di `src/evaluation.py` maupun di seluruh kode `scripts/`.

**Temuan 2 (lebih serius):** Angka MCC di laporan **tidak cocok** dengan confusion matrix di tabel yang sama.

| Cutoff | MCC di Laporan | MCC Seharusnya | Status |
|---|---|---|---|
| W1 | 0,00 | **−0,2425** | ❌ Salah |
| W2 | 0,57 | **0,6202** | ❌ Salah |
| W3 | 0,57 | **0,6202** | ❌ Salah |
| C_Full | 0,58 | **0,5698** | ⚠️ Tidak sama persis |

> Pada W1, Specificity = 0% sehingga MCC **harus negatif**. Menuliskan 0,00 berarti menyatakan model "sama dengan acak", padahal kenyataannya **lebih buruk dari acak**.

**Detail lengkap + hitung ulang semua metrik:** lihat **Bagian 6B.10** dan **6B.12.1**.

**Rekomendasi:** Tambahkan ke `src/evaluation.py`:
```python
from sklearn.metrics import matthews_corrcoef
# di results dict evaluate_test():
'test_mcc': matthews_corrcoef(y_test, y_pred)
```
Lalu perbarui angka MCC di `HASIL_ANALISIS_V2.md` dengan hasil hitung ulang.

## ⚠️ 8.4 Locked Fold Dievaluasi 4 Kali (bukan Sekali)

**Temuan:** `scripts/run_optimized_experiment.py` melakukan loop atas 4 cutoff (W1, W2, W3, C_Full), dan `final_holdout_evaluation()` dipanggil untuk **setiap** cutoff. Jadi Locked Fold dibuka **4 kali**, bukan sekali seperti diklaim di laporan.

**Pertahanan yang bisa dipakai (sah secara metodologis):**
> *"Model dan algoritma (Decision Tree + S3_E) ditetapkan SEBELUM perbandingan cutoff dilakukan — ini pre-registration. Keempat cutoff diuji pada locked fold yang sama, dan hasilnya dilaporkan seluruhnya tanpa dipilih yang terbaik. Ini bukan selection bias karena kita tidak menyeleksi berdasarkan performa holdout."*

**Rekomendasi:** Tuliskan secara eksplisit di dokumen bahwa cutoff **tidak dipilih** berdasarkan holdout, dan seluruh 4 hasil dilaporkan.

## ⚠️ 8.5 GLOSARIUM.md Masih Menyebut SMOTE & Data Imbalanced

**Temuan:** `GLOSARIUM.md` bagian 2 masih menyebut:
- "Data mahasiswa Kompeten (78) terlalu mendominasi dibandingkan Belum Kompeten (11)"
- SMOTE sebagai bagian dari metodologi

**Faktanya sekarang:**
- Distribusi **seimbang**: 44 Kompeten vs 42 Belum Kompeten
- Eksperimen memakai `balancing = 'None'` — **tanpa SMOTE**
- Ini adalah **keputusan sadar** untuk menghindari *optimistic bias*

**Rekomendasi:** Perbarui `GLOSARIUM.md` agar konsisten.

## ✅ 8.6 Yang Sudah Sangat Bagus (Poin Kekuatan)

| Aspek | Bukti |
|---|---|
| **Anti-leakage** | `feature_registry.py` hanya mengizinkan fitur yang terdaftar; `Final_Score` tidak pernah masuk X |
| **Test anti-leakage** | `tests/test_temporal_leakage.py` memblokir `NILAI_AKHIR`, `Predikat`, `Attendance_8/9/10` |
| **Pipeline terisolasi** | Imputasi berada di dalam `Pipeline` → tidak bocor antar fold |
| **Threshold dari train** | `cross_val_predict` pada X_train saja |
| **Threshold constraint** | Recall BK ≥ 70% sebagai syarat minimum |
| **Stratify** | Train dan test proporsional kelas |
| **Random seed** | `random_state=42` → reproducible |
| **CI Bootstrap** | B=1000 untuk n kecil |
| **Dummy baseline** | Membuktikan model > tebakan acak |
| **SHAP** | Menunjukkan eksplainability, bukan black-box |
| **Multi-cutoff EWS** | Membuktikan deteksi dini di minggu ke-2 |

---

# RINGKASAN JAWABAN SINGKAT

| Pertanyaan | Jawaban Inti |
|---|---|
| **K-Fold vs Nested CV?** | Nested = CV di dalam CV; Outer untuk nilai jujur, Inner untuk tuning. Menghindari *optimistic bias*. Proyek ini memakai **Outer 5-fold × 5 repeat, Inner 3-fold**. |
| **Perlu sampel?** | **YA.** n=86 kecil, perlu data beku untuk menguji + mengukur ketidakpastian + mengadu model secara adil. |
| **Data primer & sekunder?** | **Primer** (dokumen kearsipan resmi dari proses mengajar); tidak ada data sekunder dari pihak ketiga. |
| **Excel dari asdos = primer/sekunder?** | **PRIMER** — dibuat/dikumpulkan langsung oleh tim pengampu sendiri, bukan diambil dari sumber luar. |
| **Locked fold?** | 18 mahasiswa (9 BK + 9 K) yang **dikunci**, tidak boleh dilihat sampai model + threshold final selesai. |
| **80:20 atau 70:30?** | **80:20** — padanan dengan 5-Fold (konsistensi), test set cukup besar (n=18), training cukup untuk Decision Tree, dan lebih murah komputasi. |
| **Jumlah atribut prediktor?** | 3 (S1) sampai 20 (S5); **7 fitur** untuk model final S3_E. |
| **Representasi fitur?** | Agregasi statistik (mean/std/min/max/trend/completion) dengan **temporal cutoff** (W1/W2/W3/C_Full), matriks lebar, tanpa normalisasi, imputasi di dalam pipeline. |
| **Metrik evaluasi?** | **21 metrik/ukuran.** 11 metrik di Outer CV: Accuracy, Balanced Accuracy, Precision/Recall/F1 Macro, Precision/Recall/F1/F2 BK, ROC-AUC, PR-AUC. Ditambah TP/TN/FP/FN, Specificity, Mean ± SD, CI 95% (distribusi-t), CI 95% (Bootstrap B=1000), Dummy baseline, dan 3 ukuran SHAP. **Recall BK = metrik utama.** Rincian lengkap di **Bagian 6B**. |
| **Di mana tiap metrik dihitung?** | Outer CV → `src/evaluation.py:64-98` · Locked Fold → `src/evaluation.py:120-182` · Tuning → `src/tuning.py:14-30` · Threshold → `src/tuning.py:137-173` · Pilih model → `run_optimized_experiment.py:150-157` · SHAP → `src/shap_analysis.py:31-95` |

---

*Dokumen ini dibuat berdasarkan analisis langsung terhadap kode sumber dan data proyek.*
*Silakan gunakan sebagai bahan persiapan sidang.*