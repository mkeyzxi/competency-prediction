# 📋 Flow Penelitian: Prediksi Kompetensi Mahasiswa Praktikum Logika & Algoritma

Dokumen ini menjelaskan **seluruh alur penelitian** dari awal hingga akhir — mulai dari data mentah hingga menghasilkan model *Early Warning System* yang mampu mendeteksi potensi kegagalan mahasiswa secara dini.

---

## 🎯 Tujuan Penelitian

Membangun model *Machine Learning* untuk mendeteksi dini potensi kegagalan (Belum Kompeten) mahasiswa pada Praktikum Logika & Algoritma berdasarkan data aktivitas dan nilai tugas selama beberapa minggu pertama.

---

## 📊 Ringkasan Metodologi

| Aspek | Detail |
|-------|--------|
| **Ambang Batas Kompetensi** | Nilai ≥ 83 (Zona Aman A-) |
| **Data** | DBNR.xlsx (Data Nilai Raw mahasiswa) |
| **Jenis Model** | Decision Tree, Random Forest |
| **Validasi** | Nested Cross Validation |
| **Evaluasi** | Holdout 20% (Data Beku), Bootstrap CI 95% |
| **Interpretasi** | SHAP (SHapley Additive exPlanations) |
| **Pendekatan** | Murni tanpa SMOTE (menghindari optimistic bias) |

---

## 🗺️ Tahapan Penelitian

### Tahap 1: Persiapan & Konfigurasi Lingkungan

```
├── Instalasi dependencies (environment.yml / requirements.txt)
├── Strukturisasi folder proyek (src/, scripts/, data/, models/, results/)
└── Verifikasi data mentah tersedia (data/raw/DBNR.xlsx)
```

**Tindakan:**
- Pastikan Python environment aktif dengan semua library terinstall
- Siapkan data mentah `DBNR.xlsx` di folder `data/raw/`
- Buat folder output: `data/processed/`, `data/features/`, `models/`, `results/shap/`

---

### Tahap 2: Pembersihan Data Mentah (Data Preparation)

**Script:** `python src/build_dataset.py`

```
DBNR.xlsx (Raw)
     │
     ▼
┌─────────────────────────────────┐
│  src/build_dataset.py           │
│  • Membaca Excel dosen          │
│  • Melabeli kelulusan (≥ 83)    │
│  • Konversi ke format terstruktur│
└─────────────────────────────────┘
     │
     ▼
data/processed/
├── activities_long.csv   (Riwayat aktivitas per mahasiswa)
└── students_master.csv    (Data master mahasiswa)
```

**Output:**
- `activities_long.csv` — Baris riwayat data terstruktur (aktivitas, nilai, absensi per mahasiswa)
- `students_master.csv` — Data master dengan label kelulusan (Kompeten=1 / Belum Kompeten=0)

---

### Tahap 3: Ekstraksi & Rekayasa Fitur (Feature Engineering)

**Script:** `python src/features.py`

```
activities_long.csv
     │
     ▼
┌─────────────────────────────────┐
│  src/features.py                │
│  • Perhitungan statistik        │
│  • Temporal cutoffs (anti-bocor)│
│  • Rumus EWS                    │
└─────────────────────────────────┘
     │
     ▼
data/features/
├── C_Full_S1.csv    (Skenario 1 - Dasar)
├── C_Full_S2.csv    (Skenario 2)
├── C_Full_S3.csv    (Skenario 3 - Historis Terbaik)
├── C_Full_S4.csv    (Skenario 4)
├── C_Full_S5.csv    (Skenario 5)
├── C_Full_S3_A.csv  (Skenario EWS A)
├── C_Full_S3_B.csv  (Skenario EWS B)
├── C_Full_S3_C.csv  (Skenario EWS C)
├── C_Full_S3_D.csv  (Skenario EWS D)
└── C_Full_S3_E.csv  (Skenario EWS E - Genuine EWS)
```

**Fitur Utama yang Diekstrak:**
- Rata-rata nilai tugas per aktivitas
- Tingkat absensi
- Pola *Early Warning System* (EWS):
  - Tren Penurunan
  - Absen Berturut-turut
  - Performa Awal (2-3 minggu pertama)
  - `Early_Performance_Composite`

---

### Tahap 4: Penetapan Batas Kompetensi (Labeling)

```
Nilai Akhir Mahasiswa
     │
     ▼
┌────────────────────────────┐
│  IF Nilai ≥ 83:            │
│     → Kompeten (1)         │
│  ELSE:                     │
│     → Belum Kompeten (0)   │
└────────────────────────────┘
```

**Justifikasi:**
- Berdasarkan standarisasi nilai predikat unggul akademik (Zona Aman A-)
- Ambang 83 dipilih sebagai batas minimum kompetensi

---

### Tahap 5: Eksperimen Baseline — Komparasi Skenario (S1–S5)

**Script:** `python scripts/run_baseline_experiments.py`

```
┌──────────────────────────────────────────────┐
│  NESTED CROSS VALIDATION                      │
│                                              │
│  Outer CV (5-fold) ──► Evaluasi Performa    │
│       │                                      │
│       ▼                                      │
│  Inner CV (RandomizedSearchCV) ──► Tuning   │
│       │                                      │
│       ▼                                      │
│  Training ──► Model ──► Evaluation           │
└──────────────────────────────────────────────┘
```

**Skenario yang Dibandingkan:**

| Skenario | Jenis Fitur | Deskripsi |
|----------|------------|-----------|
| S1 | Dasar | Fitur minimum (absensi + nilai dasar) |
| S2 | +Metadata | S1 + metadata tambahan |
| S3 | Historis Lengkap | Fitur riwayat penuh satu semester |
| S4 | +Temporal | S3 + fitur temporal |
| S5 | Eksperimen | Kombinasi fitur eksperimental |

**Output:**
- **Leaderboard** performa di terminal
- Model S3 terbaik disimpan di `models/`
- Terbukti: **S3 (Historis)** adalah konfigurasi terbaik

---

### Tahap 6: Analisis SHAP untuk Model S3 Dasar (Retrospektif)

**Script:** `python scripts/generate_shap_s3_dasar.py`

```
Model S3 Dasar (Retrospektif)
     │
     ▼
┌─────────────────────────────────┐
│  src/evaluate_shap.py           │
│  • SHAP Values Calculation      │
│  • Feature Importance Plot      │
│  • Beeswarm Plot                │
└─────────────────────────────────┘
     │
     ▼
results/shap/
└── Baseline_S3_*.png
```

**Temuan:**
- Model S3 Dasar sangat bergantung pada `Laporan_Mean` (nilai rata-rata laporan)
- Belum mencerminkan deteksi *dini* — masih menggunakan data retrospektif penuh

---

### Tahap 7: Eksperimen Lanjutan — Uji Coba EWS (S3_A hingga S3_EWS)

**Script:** `python scripts/run_optimized_experiment.py`

```
┌──────────────────────────────────────────────────┐
│  EKSperimen Teroptimasi (Standar SINTA-2)        │
│                                                  │
│  1. Nested CV: Bandingkan S3_A s/d S3_EWS       │
│  2. Pilih 1 Pemenang berdasarkan CV             │
│  3. Kunci Model Pemenang                         │
│  4. Evaluasi pada Data Beku (Holdout)            │
│  5. Hitung Bootstrap CI 95%                     │
│  6. Simpan Model Final                          │
└──────────────────────────────────────────────────┘
```

**Kelompok Fitur EWS:**

| Kode | Fitur EWS | Deskripsi |
|------|----------|-----------|
| S3_A | Early Warning Dasar | Fitur peringatan dini dasar |
| S3_B | +Tren Penurunan | Menangkap pola penurunan performa |
| S3_C | +Absen Berturut | Deteksi ketidaktambatan beruntun |
| S3_D | Performa Awal | Fokus 2-3 minggu pertama |
| S3_E | **Genuine EWS** | Kombinasi fitur deteksi dini |

**Metodologi Kunci:**
- Model hanya menggunakan data **2-3 minggu pertama**
- **Zero Temporal Leakage** — masa depan tidak diketahui
- Evaluasi ketat pada data yang belum pernah dilihat

**Output:**
- Model Final (S3_E) disimpan di `models/`
- Laporan performa lengkap dengan Confidence Interval

---

### Tahap 8: Analisis SHAP untuk Model Final (S3_E — Genuine EWS)

**Script:** `python scripts/generate_shap.py`

```
Model Final S3_E (Genuine EWS)
     │
     ▼
┌─────────────────────────────────┐
│  src/evaluate_shap.py           │
│  • SHAP Values Calculation      │
│  • Feature Importance Plot      │
│  • Beeswarm Plot                │
└─────────────────────────────────┘
     │
     ▼
results/shap/
└── FINAL_S3_E_*.png
```

**Temuan:**
- `Early_Performance_Composite` menjadi **kontributor prediktif tertinggi**
- Model berhasil mendeteksi kegagalan berdasarkan performa awal
- Membuktikan konsep *Early Warning System* yang sesungguhnya

---

### Tahap 9: Perbandingan & Kesimpulan

```
┌─────────────────────────────────────────────────────┐
│              PERBANDINGAN MODEL                     │
├─────────────────────┬───────────────────────────────┤
│   S3 Dasar          │   S3_E Final                  │
│   (Retrospektif)    │   (Genuine EWS)               │
├─────────────────────┼───────────────────────────────┤
│ • Laporan_Mean     │ • Early_Performance_Composite │
│ • Data penuh 1      │ • Data 2-3 minggu pertama     │
│   semester          │ • Deteksi dini yang sesungguhnya│
│ • Reaktif           │ • Proaktif                    │
└─────────────────────┴───────────────────────────────┘
```

**Kesimpulan:**
1. Model **S3_E (Genuine EWS)** berhasil menjadi *Early Warning System* yang andal
2. Deteksi dilakukan berdasarkan performa awal (2-3 minggu pertama)
3. Tidak ada kebocoran temporal — masa depan tidak digunakan
4. Metodologi sesuai standar SINTA-2 dengan validasi ketat

---

## 📁 Struktur Output Akhir

```
Logika-Algoritma/
├── data/
│   ├── raw/
│   │   └── DBNR.xlsx              # Data mentah
│   ├── processed/
│   │   ├── activities_long.csv    # Data terstruktur
│   │   └── students_master.csv     # Master mahasiswa
│   └── features/
│       ├── C_Full_S1.csv          # Fitur Skenario 1
│       ├── C_Full_S2.csv          # Fitur Skenario 2
│       ├── C_Full_S3.csv          # Fitur Skenario 3
│       ├── C_Full_S4.csv          # Fitur Skenario 4
│       ├── C_Full_S5.csv          # Fitur Skenario 5
│       └── C_Full_S3_*.csv        # Fitur EWS A-E
├── models/
│   ├── best_s3_model.pkl          # Model S3 Dasar
│   └── final_s3e_model.pkl        # Model Final (S3_E)
├── results/
│   ├── shap/
│   │   ├── Baseline_S3_*.png      # SHAP S3 Dasar
│   │   └── FINAL_S3_E_*.png       # SHAP Model Final
│   ├── leaderboard.csv            # Hasil komparasi
│   └── evaluation_report.json     # Laporan evaluasi
├── src/                           # Source code (mesin internal)
├── scripts/                       # Script eksekusi
└── reports/                       # Laporan akhir
```

---

## 🚀 Cara Menjalankan Seluruh Pipeline

### Opsi 1: Otomatis (Satu Klik)

```bash
run_pipeline.bat
```

### Opsi 2: Manual (Berurutan)

```bash
# 1. Bersihkan data
python src/build_dataset.py

# 2. Ekstrak fitur
python src/features.py

# 3. Jalankan komparasi baseline
python scripts/run_baseline_experiments.py

# 4. Analisis SHAP S3 Dasar
python scripts/generate_shap_s3_dasar.py

# 5. Jalankan eksperimen EWS
python scripts/run_optimized_experiment.py

# 6. Analisis SHAP Model Final
python scripts/generate_shap.py
```

---

## 📚 Referensi Dokumen Terkait

| Dokumen | Kegunaan |
|---------|----------|
| `README.md` | Panduan eksekusi lengkap |
| `FUNCTION.md` | Referensi arsitektur & fungsi |
| `GLOSARIUM.md` | Istilah teknis |
| `HASIL_ANALISIS_V2.md` | Hasil analisis data |
| `AUDIT.md` | Hasil audit data |

---

## ⚠️ Catatan Penting

1. **Tanpa SMOTE** — Dataset dijaga murni untuk menghindari *optimistic bias*
2. **Zero Temporal Leakage** — Model EWS tidak pernah melihat data masa depan
3. **Holdout Ketat** — 20% data beku tidak pernah digunakan untuk training/tuning
4. **Bootstrap CI 95%** — Karena sampel kecil (n=18), confidence interval wajib dilaporkan
5. **Standar SINTA-2** — Metodologi mengikuti standar publikasi ilmiah

---

*Dokumen ini diperbarui terakhir: 2026*
