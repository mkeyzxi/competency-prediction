# PRODUCT REQUIREMENTS DOCUMENT
## Sistem Prediksi Kompetensi Mahasiswa Praktikum Logika Pemrograman
**PRD v1.1 - Revisi berdasarkan struktur data aktual, dua skema penilaian, dan aturan preprocessing kehadiran**

- **Jenis:** Research / Machine Learning Classification / Explainable AI
- **Platform:** Python + Anaconda/Conda
- **Status:** Draft implementasi final / siap diterjemahkan ke pipeline

## 0. Kontrol Dokumen
| Item | Isi |
| :--- | :--- |
| **Versi** | 1.1 |
| **Basis** | PRD v1.0 dan laporan eksperimen awal 64 mahasiswa yang diberikan; diperbarui menggunakan struktur data terbaru dan aturan preprocessing yang disampaikan pengguna. |
| **Unit analisis** | 1 baris = 1 mahasiswa setelah resolusi duplikasi NIM dan integrasi sumber data. |
| **Target** | Kompeten jika Final Individu >= 75; Belum Kompeten jika Final Individu < 75. |
| **Model utama** | Decision Tree dan Random Forest. |
| **XAI** | TreeSHAP pada model/skema terpilih. |
| **Skema penilaian** | SCHEME_AC untuk kelas A,C; SCHEME_BDE untuk kelas B,D,E. |
| **Kelompok evaluator** | Group 1: A,C; Group 2: B,D,E. Kelompok evaluator/scoring tidak otomatis menjadi fitur model utama. |
| **Status hasil 64 mahasiswa** | Preliminary, bukan hasil final penelitian. |

## 1. Ringkasan Penelitian
Penelitian membangun model klasifikasi untuk memprediksi ketercapaian kompetensi operasional mahasiswa pada Praktikum Logika Pemrograman berdasarkan aktivitas dan performa sebelum evaluasi Final Individu. Kompetensi operasional didefinisikan dari Final Individu: nilai >= 75 dikategorikan Kompeten dan nilai < 75 sebagai Belum Kompeten.

Dataset aktual memiliki dua skema penilaian. Kelas A dan C menggunakan skema AC, sedangkan kelas B, D, dan E menggunakan skema BDE. Perbedaan skema tidak dianggap sebagai kesalahan data; perbedaan tersebut merupakan karakteristik konteks penelitian yang harus dipertahankan dan didokumentasikan.

## 2. Tujuan dan Pertanyaan Penelitian
Tujuan penelitian adalah membandingkan Decision Tree sebagai baseline dan Random Forest sebagai model pembanding/utama, menguji tiga skenario feature engineering, mengevaluasi generalisasi, dan menjelaskan kontribusi fitur dengan TreeSHAP.

### 2.1 Pertanyaan Penelitian
- **RQ1** - Bagaimana membangun model prediksi ketercapaian kompetensi mahasiswa berdasarkan aktivitas dan performa praktikum sebelum Final Individu?
- **RQ2** - Bagaimana perbandingan performa Decision Tree dan Random Forest pada data dengan dua skema penilaian yang berbeda?
- **RQ3** - Fitur apa yang memberikan kontribusi terbesar terhadap prediksi menurut TreeSHAP pada model final yang dipilih?
- **RQ4 (analisis konteks)** - Apakah performa model menunjukkan pola yang berbeda antara kelompok kelas A,C dan kelompok B,D,E? Analisis ini bersifat robustness/context analysis, bukan eksperimen model utama.

## 3. Ruang Lingkup
### 3.1 In Scope
- Ingestion CSV/XLSX dan sumber Final terpisah.
- Validasi struktur, NIM, duplikasi, missing/empty, rentang nilai, dan status kehadiran.
- Rekonstruksi 10 pertemuan kehadiran sesuai aturan sumber kelas.
- Kriteria kelayakan berdasarkan jumlah ketidakhadiran.
- Pembentukan label dari Final Individu.
- Feature engineering S1-S3 dengan semantic mapping per skema.
- Stratified 80:20 split dan 5-fold Stratified CV pada training.
- Decision Tree dan Random Forest.
- Hyperparameter tuning terbatas dan terdokumentasi.
- Accuracy, Precision, Recall, F1, confusion matrix, CV mean +/- SD.
- TreeSHAP global dan local.
- Analisis konteks AC vs BDE dan kelas A-E tanpa memaksa model terpisah.
- Ekspor data processed, konfigurasi, model, metrics, plots, dan laporan reproducibility.

### 3.2 Out of Scope
- Model deep learning.
- Menyatakan hubungan kausal antara fitur dan kompetensi.
- Menggunakan NILAI AKHIR sebagai target utama.
- Memasukkan NIM, Nama, Predikat, Final, atau informasi final-derived ke X.
- Menjadikan kelompok asisten sebagai prediktor utama tanpa eksperimen metodologis terpisah.
- Membuat lima model terpisah A-E sebagai eksperimen utama.

## 4. Struktur Data dan Dua Skema Penilaian
### 4.1 Skema AC - Kelas A dan C
| Komponen | Bobot |
| :--- | :--- |
| Kehadiran | 10% |
| TP | 15% |
| Respons | 15% |
| Laporan | 35% |
| Final | 25% |

### 4.2 Skema BDE - Kelas B, D, dan E
| Komponen | Bobot |
| :--- | :--- |
| Tugas Pendahuluan + Respons | 20% |
| Laporan | 30% |
| Keaktifan | 10% |
| Final | 30% |
| Kehadiran | 10% |

Bobot yang berbeda bukan alasan untuk mengubah nilai mentah agar seragam. Pipeline harus menjaga nilai komponen asli dan hanya menyeragamkan makna fitur (misalnya TP_Mean, Respons_Mean, Laporan_Mean) jika komponen tersebut memang tersedia pada kedua skema.

### 4.3 Context Variables
| Variabel | Nilai | Peran |
| :--- | :--- | :--- |
| Class | A/B/C/D/E | Metadata/context |
| Scoring Scheme | SCHEME AC / SCHEME BDE | Metadata/context |
| Assistant Group | GROUP AC / GROUP BDE | Metadata/context |
| NIM | identifier | Key untuk join; bukan fitur |
| Nama | identifier | Metadata; bukan fitur |

Karena kelompok A, C menggunakan kelompok asisten yang sama sekaligus skema penilaian yang sama, dan B,D,E juga demikian, efek skema penilaian dan kelompok asisten tidak dapat diidentifikasi secara terpisah hanya dari dataset observasional ini. Oleh sebab itu laporan tidak boleh menyimpulkan bahwa perbedaan performa disebabkan oleh bobot atau asisten secara kausal.

## 5. Arsitektur Sumber Data
Pipeline menggunakan dua sumber logis: (1) tabel nilai/aktivitas praktikum dan (2) tabel Final/PENILAIAN_UAS yang memuat final dan nilai flowchart/kodingan. Kedua sumber digabung berdasarkan NIM, tetapi hanya setelah validasi keunikan NIM dan aturan resolusi duplikasi.

| Sumber | Informasi utama | Digunakan untuk |
| :--- | :--- | :--- |
| Tabel praktikum AC | Kehadiran, laporan/asistensi, TP, respons, Final Individu, Final Kelompok, Total, NILAI AKHIR, Predikat | Aktivitas/prediktor + metadata; Final hanya untuk target |
| Tabel praktikum BDE | Kehadiran, keaktifan, tugas pendahuluan, laporan, nilai/assessments per aktivitas | Aktivitas/prediktor + metadata |
| PENILAIAN_UAS/tabel Final | NIM, final, nilai flowchart, nilai kodingan | Ground truth Final dan rekonstruksi kehadiran sesi UAS |

Aturan penting: apabila satu NIM muncul lebih dari satu kali pada tabel Final, pipeline tidak boleh otomatis mengambil baris pertama/terakhir. Duplikasi harus masuk ke `data_quality/duplicate_final.csv` dan diselesaikan berdasarkan sumber akademik sebelum modelling.

## 6. Target dan Leakage Prevention
### 6.1 Target
- `Competency_Label = 1 if Final_Individu >= 75 else 0`
- `Competency_Name = "Kompeten" if Competency_Label == 1 else "Belum Kompeten"`

NILAI AKHIR tidak digunakan untuk membentuk target. Final Individu adalah target operasional karena penelitian memprediksi ketercapaian evaluasi final, bukan agregasi nilai akhir berbobot.

### 6.2 Kolom yang dilarang masuk X
- Final_Individu
- Final_UAS / final pada sumber Final
- Nilai_Flowchart
- Nilai_Kodingan
- Final_Kelompok
- Total dan total agregasi yang mengandung Final
- NILAI_AKHIR
- Predikat
- NIM
- Nama
- Nomor urut
- Kolom lain yang dihitung dari atau tersedia setelah evaluasi final.

Nilai Flowchart/Kodingan berada pada sumber Final dan tidak boleh digunakan sebagai fitur prediksi jika tujuan prediksi adalah sebelum Final. Hal yang sama berlaku untuk nilai atau kolom apa pun yang baru tersedia setelah evaluasi final.

## 7. Aturan Preprocessing Kehadiran
Dataset dinyatakan sudah bersih dari sisi kualitas akademik: nilai 0 atau sel kosong mempunyai makna operasional bahwa mahasiswa tidak melaksanakan aktivitas tersebut; nilai rendah dipertahankan sebagai nilai valid dan dapat merepresentasikan keterlambatan, kualitas rendah, atau pada laporan hanya melakukan asistensi pada satu asisten praktikan. Pipeline tidak boleh mengganti nilai rendah menjadi missing dan tidak boleh menghapus nilai 0 hanya karena kecil.

### 7.1 Status kehadiran
Status kehadiran harus direkonstruksi menjadi 10 pertemuan bernilai 1, 0.5, atau 0 sesuai sumber. Untuk aturan kelayakan, ketidakhadiran dihitung sebagai sesi dengan status 0. Status 0.5 adalah hadir parsial dan bukan otomatis absen penuh.

### 7.2 Kelas A,C
| Pertemuan | Sumber/aturan | Status hadir |
| :--- | :--- | :--- |
| 1 | Kontrak kuliah; semua mahasiswa dianggap hadir | 1 untuk semua |
| 2-7 | Kolom kehadiran 1-6 pada sheet utama AC | Gunakan nilai 1/0.5/0 aktual |
| 8 | Merujuk ke nilai individu | Jika nilai individu tersedia -> hadir; jika kosong -> tidak hadir |
| 9 | Merujuk ke nilai final | Jika nilai final tersedia -> hadir; jika kosong -> tidak hadir |
| 10 | Semua mahasiswa dianggap hadir | 1 untuk semua |

### 7.3 Kelas B,D,E
| Pertemuan | Sumber/aturan | Status hadir |
| :--- | :--- | :--- |
| 1-7 | Kolom kehadiran pada sheet BDE | Gunakan nilai kehadiran aktual; blank = tidak hadir bila blank memang merepresentasikan tidak hadir |
| 8-9 | Sheet PENILAIAN_UAS; gunakan aturan kolom/session yang telah ditetapkan sumber. Nilai flowchart yang tersedia menjadi bukti kehadiran pada sesi terkait. | Nilai flowchart tersedia -> hadir; kosong -> tidak hadir |
| 10 | Sheet PENILAIAN_UAS / nilai sesi ke-10 | Ada nilai -> hadir; tidak ada nilai -> tidak hadir |

Karena pemetaan pertemuan 8-9 BDE bergantung pada sheet PENILAIAN_UAS, implementasi wajib menyimpan mapping tersebut dalam config dan tidak boleh mengandalkan posisi kolom. Jika terdapat dua kolom/session berbeda untuk pertemuan 8 dan 9, masing-masing harus dipetakan eksplisit di konfigurasi.

### 7.4 Kriteria keluar awal dan kelayakan praktikum
Aturan pengguna menetapkan dua ambang. Keduanya dicatat sebagai status yang berbeda agar tidak saling menghapus informasi: (a) mahasiswa dengan lebih dari 1 ketidakhadiran diberi flag bahwa ia diperlakukan sebagai telah keluar dari praktikum sejak awal; (b) mahasiswa dengan lebih dari 3 ketidakhadiran dinyatakan tidak memenuhi syarat untuk lolos praktikum. Karena >3 merupakan subset dari >1, flag dapat aktif bersamaan.

| Flag | Definisi | Konsekuensi modelling |
| :--- | :--- | :--- |
| Early_Exit_Flag | absence count > 1 | Kandidat excluded dari eksperimen utama sesuai kebijakan penelitian; harus dilaporkan jumlahnya. |
| Attendance_Ineligible_Flag | absence_count > 3 | Tidak memenuhi syarat lulus berdasarkan kehadiran; excluded dari modelling final jika definisi target penelitian membutuhkan peserta yang eligible. |
| Attendance_Rate | sum(session_presence_score)/10 | Fitur numerik; 0.5 dipertahankan sebagai hadir parsial. |

Rekomendasi implementasi: simpan kedua flag pada dataset audit walaupun baris akhirnya dikeluarkan dari modelling. Jangan menghapus secara silent. Buat laporan jumlah mahasiswa: raw -> early exit -> attendance ineligible -> eligible.

## 8. Aturan Missing, 0, dan Nilai Rendah
| Kondisi | Interpretasi | Tindakan |
| :--- | :--- | :--- |
| Sel kosong pada aktivitas | Tidak melaksanakan aktivitas menurut aturan data yang diberikan | Representasikan sebagai 0 pada fitur aktivitas bila definisi komponen memang menyatakan tidak mengerjakan = 0 |
| Nilai 0 | Tidak melaksanakan aktivitas | Pertahankan sebagai 0; jangan jadi NaN. |
| Nilai rendah | Nilai valid; dapat merefleksikan keterlambatan / kualitas / asistensi parsial | Pertahankan nilai asli. |
| Laporan rendah karena hanya asistensi pada 1 asisten | Informasi akademik yang valid | Pertahankan; jangan outlier removal otomatis. |
| Missing pada Final | Target tidak dapat dibentuk | Masukkan quality issue; perbaiki dari sumber resmi atau keluarkan dari modelling. |
| Missing nilai flowchart/kodingan | Bukan otomatis nilai nol; konteksnya dipakai untuk rekonstruksi kehadiran sesuai aturan sumber | Jangan imputasi tanpa aturan. |

## 9. Feature Engineering
Feature engineering dibangun berdasarkan makna komponen, bukan posisi kolom. Karena skema AC dan BDE berbeda, pipeline menggunakan *semantic mapping* per skema.

**Hipotesis Penelitian**:
> Konsistensi representasi fitur lintas skema penilaian dapat menghasilkan feature space yang lebih seragam sehingga memungkinkan model belajar pola kompetensi lintas kelas secara lebih konsisten.

### 9.1 Common features & Aturan Dual Feature Mapping
| Feature | Definisi |
| :--- | :--- |
| Attendance_Rate | Jumlah skor kehadiran seluruh 10 pertemuan / 10; mempertahankan bobot 0.5 sebagai hadir parsial. |
| TP_Mean | Rata-rata nilai TP yang tersedia dan berlaku. |
| Respons_Mean | Rata-rata nilai respons. |
| Laporan_Mean | Rata-rata nilai laporan/asistensi yang tersedia sesuai komponen laporan. |

**Aturan untuk Kelas BDE**:
Pada skema BDE, ketika komponen TP dan Respons hanya tersedia sebagai satu skor gabungan, skor gabungan tersebut dipetakan ke `TP_Mean` dan `Respons_Mean` dengan nilai yang sama (duplikasi nilai) untuk menjaga konsistensi feature space lintas kelompok. 
Nilai tersebut **tidak dibagi menjadi dua** karena pembagian akan menciptakan nilai yang tidak merepresentasikan observasi asli (Misal: 85 tetap menjadi 85 di TP dan 85 di Respons). Catatan bahwa `TP_Mean == Respons_Mean` untuk observasi BDE yang berasal dari skor gabungan; kedua fitur ini BUKAN merupakan pengukuran independen. Feature duplication ini hanya untuk **menyamakan feature representation**.

### 9.2 S1 - Basic
`S1 = [Attendance_Rate, TP_Mean, Respons_Mean, Laporan_Mean]`
Model utama harus menggunakan struktur fitur yang sama untuk seluruh kelas (tidak ada lagi `TP_Respons_Mean` di output fitur akhir).

### 9.3 S2 - Behavioral
`S2 = S1 + [TP_Completion_Rate, Respons_Completion_Rate, Laporan_Completion_Rate]`
Completion rate hanya boleh dibuat jika status pengerjaan benar-benar dapat diverifikasi. Untuk BDE dengan TP+Respons gabungan, `TP_Completion_Rate` dan `Respons_Completion_Rate` boleh memiliki nilai yang sama hanya apabila status aktivitas yang mendasarinya memang sama untuk komponen gabungan tersebut.

### 9.4 S3 - Relational
`S3 = S2 + [Respons_TP_Gap]`
Respons_TP_Gap didefinisikan sebagai `Respons_Mean - TP_Mean`. Karena pada BDE skor TP dan Respons adalah sama hasil duplikasi, maka fitur gap akan **selalu 0** untuk BDE. Fitur ini tetap disertakan untuk menjaga arsitektur *feature space* yang konsisten dengan A/C dan berperan sebagai eksperimen sensitivitas.

### 9.5 Keaktifan BDE
Keaktifan adalah komponen khusus BDE. Jangan membuat `Keaktifan = 0` untuk A/C. Keaktifan tidak boleh dimasukkan ke model utama lintas semua kelas apabila fitur tersebut tidak tersedia secara valid untuk seluruh kelas. Dapat dilakukan eksperimen tambahan khusus robustness pada subset BDE.

## 10. Desain Eksperimen
Jangan mengasumsikan bahwa perubahan feature engineering pasti meningkatkan akurasi. Bandingkan secara empiris: Baseline lama vs Feature Space baru.

| Eksperimen | Fitur/Mapping | Model |
| :--- | :--- | :--- |
| E1 | Old Feature Mapping | Decision Tree |
| E2 | Old Feature Mapping | Random Forest |
| E3 | New Consistent Feature Mapping (S1) | Decision Tree |
| E4 | New Consistent Feature Mapping (S1) | Random Forest |
| E5 | New Consistent Feature Mapping (S2) | Decision Tree |
| E6 | New Consistent Feature Mapping (S2) | Random Forest |
| E7 | New Consistent Feature Mapping (S3) | Decision Tree |
| E8 | New Consistent Feature Mapping (S3) | Random Forest |

Eksperimen utama dilakukan pada dataset eligible yang sudah melalui preprocessing dan resolusi target. Gunakan split dan seed (42) yang konsisten agar perbandingan adil. Analisis AC vs BDE dilakukan sebagai analisis konteks/robustness.

### 10.1 Model selection
- Gunakan CV 5-fold pada training sebagai dasar pemilihan konfigurasi.
- Tetapkan metrik utama sebelum melihat hasil test.
- Recall kelas Belum Kompeten diprioritaskan bila tujuan operasional adalah early warning.
- Gunakan test set satu kali sebagai evaluasi final setelah model dikunci.
- Jika dua konfigurasi sangat dekat, pertimbangkan mean/SD CV dan kestabilan, bukan hanya selisih pada satu test split.

## 11. Split, Cross-Validation, dan Tuning
- **Train/Test** = 80/20, stratify=Competency_Label, random_state=42
- **CV** = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Jika proporsi label dan kelompok scoring sangat timpang, distribusi Class x Competency_Label dan Scoring_Scheme x Competency_Label wajib diaudit sebelum split. Stratifikasi utama tetap pada label agar prosedur konsisten dan tidak memaksakan strata yang terlalu kecil.

Semua langkah yang belajar dari data - termasuk imputasi jika diperlukan, selection, tuning, threshold fitting, dan transformasi - harus dilakukan di dalam pipeline/CV training. Test set tidak boleh dipakai untuk tuning, pemilihan fitur, atau penetapan threshold.

### 11.1 Hyperparameter
- **Decision Tree:** max_depth, min_samples_split, min_samples_leaf, class_weight bila diperlukan.
- **Random Forest:** n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight.
- Gunakan grid/randomized search yang kecil dan didokumentasikan karena ukuran sampel relatif terbatas.

## 12. Evaluasi
| Metrik | Penggunaan |
| :--- | :--- |
| Accuracy | Proporsi prediksi benar secara keseluruhan. |
| Precision | Ketepatan prediksi pada kelas target. |
| Recall | Kemampuan menangkap kelas target; sangat penting untuk Belum Kompeten pada early warning. |
| F1 | Keseimbangan precision-recall. |
| Confusion Matrix | Analisis kesalahan kelas. |
| CV mean +/- SD | Stabilitas hasil lintas fold. |

Laporan final minimal memuat metrik per kelas untuk Precision/Recall/F1, macro atau weighted aggregate yang jelas, confusion matrix, CV mean +/- SD, dan hasil test. Jangan hanya melaporkan accuracy.

## 13. Explainable AI - TreeSHAP
- Jalankan setelah model/skema final dipilih.
- **Global:** mean absolute SHAP, bar plot, beeswarm.
- **Local:** minimal satu kasus Kompeten dan satu Belum Kompeten; tambahkan kasus near-boundary atau false prediction bila tersedia.
- Interpretasi menggunakan bahasa "kontribusi terhadap prediksi model", bukan "menyebabkan kompetensi".
- Jika model terbaik ternyata Decision Tree, XAI dapat menggunakan penjelasan pohon dan/atau TreeSHAP untuk model tree-based tersebut, dengan alasan metodologis yang konsisten.

## 14. Prosedur Preprocessing Lengkap
Bagian ini merupakan spesifikasi implementasi yang wajib diikuti. Tujuannya adalah menghasilkan satu dataset analitik yang konsisten dari dua sumber data dan dua skema penilaian.

1. Baca semua sheet sumber menggunakan mapping konfigurasi, bukan posisi kolom.
2. Normalisasi NIM sebagai string: trim whitespace, hilangkan format numerik .0 bila muncul, dan pertahankan leading zero bila ada.
3. Tambahkan metadata Class, Scoring_Scheme, dan Assistant_Group berdasarkan sumber/kelas.
4. Validasi satu baris = satu mahasiswa pada setiap sumber aktivitas.
5. Validasi NIM kosong, duplikat, dan mismatch antar sumber.
6. Untuk sumber Final, cari duplikat NIM. Jangan pilih nilai secara otomatis; simpan unresolved duplicates sampai sumber resmi menentukan baris yang benar.
7. Rekonstruksi 10 sesi kehadiran A/C sesuai tabel mapping M1=hadir semua, M2-M7 dari attendance 1-6, M8 dari nilai individu, M9 dari nilai final, M10 hadir semua.
8. Rekonstruksi B/D/E: M1-M7 dari sheet BDE; M8-M9 mengikuti mapping PENILAIAN_UAS dan bukti nilai flowchart sesuai aturan sumber; M10 berdasarkan ada/tidaknya nilai sesi ke-10.
9. Hitung `absence_count` = jumlah sesi dengan `attendance score == 0`. Simpan partial_count untuk sesi 0.5.
10. Buat `Early_Exit_Flag` = `absence_count > 1`.
11. Buat `Attendance_Ineligible_Flag` = `absence_count > 3`.
12. Buat eligibility status dan simpan alasan eksklusi setiap mahasiswa.
13. Pertahankan nilai 0/blank pada aktivitas sebagai tidak melaksanakan sesuai aturan data. Jangan outlier filtering otomatis terhadap nilai rendah.
14. Buat target dari `Final_Individu >= 75` dan `< 75` setelah Final resmi terverifikasi.
15. Buang Final, flowchart, kodingan, Final Kelompok, Total, NILAI AKHIR, Predikat, dan identifier dari X.
16. Bangun common features dan fitur khusus skema sesuai data yang tersedia.
17. Jalankan quality checks: no duplicate student rows, no target missing, no forbidden columns, no post-final predictor.
18. Simpan clean master, eligible dataset, excluded dataset, feature datasets S1/S2/S3, dan audit log.

## 15. Pseudocode Preprocessing
```python
for each source_sheet:
    df = read_sheet(source_sheet)
    df = normalize_nim(df)
    validate_schema(df)

merge activity_source with final_source on NIM after resolving duplicates
assign Class, Scoring_Scheme, Assistant_Group

if class in [A, C]:
    attendance = [1, raw1, raw2, raw3, raw4, raw5, raw6,
                  presence(individual_score), presence(final_score), 1]
else: # B, D, E
    attendance = [raw1, raw2, raw3, raw4, raw5, raw6, raw7,
                  presence(flowchart_for_session_8),
                  presence(flowchart_for_session_9),
                  presence(session_10_value)]

absence_count = count(attendance == 0)
early_exit = absence_count > 1
attendance_ineligible = absence_count > 3

label = 1 if final_individu >= 75 else 0
X = remove([NIM, Nama, Final, Flowchart, Kodingan,
            Final_Kelompok, Total, NILAI_AKHIR, Predikat,
            all_post_final_columns])

features = build_semantic_features(X, scoring_scheme)
run_S1_S2_S3(features)
```
*Catatan: fungsi `presence()` berarti non-empty/terdapat nilai sesuai aturan sumber. Ia tidak boleh memakai besar kecil nilai untuk menyimpulkan hadir, kecuali aturan sumber memang menyatakannya.*

## 16. Data Quality dan Audit Trail
| Check | Expected |
| :--- | :--- |
| NIM unique after resolution | 1 baris per mahasiswa |
| Final unique per NIM | 1 nilai Final resmi per mahasiswa |
| Target non-missing | 100% eligible rows |
| Attendance sessions | 10 sesi per mahasiswa |
| Forbidden columns in X | 0 (tidak ada Final, NIM, Nama, NILAI_AKHIR, dll) |
| Duplicate model rows | 0 |
| Unresolved Final duplicates | 0 before modelling |
| TP_Response_Source | Harus berisi `SEPARATE` (AC) atau `COMBINED_DUPLICATED` (BDE) |
| TP_Mean == Respons_Mean | Harus True apabila TP_Response_Source == `COMBINED_DUPLICATED` |
| Feature Columns | Harus mutlak identik untuk semua kelas (Attendance_Rate, TP_Mean, dll) |
| Early Exit count | reported |
| Attendance Ineligible count | reported |
| Scoring group counts | reported |
| Class x label distribution | reported |

Semua eksklusi harus memiliki `reason_code`, misalnya `EARLY_EXIT_GT1`, `ATTENDANCE_INELIGIBLE_GT3`, `MISSING_FINAL`, `DUPLICATE_FINAL_UNRESOLVED`, atau `INVALID_NIM`.

## 17. Analisis Konteks dan Heterogenitas
Karena skema AC dan BDE sekaligus terkait dengan kelompok asisten, analisis konteks difokuskan pada dua grup scoring, bukan membuat model A-E terpisah sebagai eksperimen utama.

| Analisis | Tujuan |
| :--- | :--- |
| Jumlah mahasiswa per Class | Audit representasi |
| Proporsi label per Class | Audit imbalance |
| Rata-rata fitur per Class | Descriptive analysis |
| Performa Overall vs AC vs BDE | Robustness |
| Distribusi Early Exit/Attendance_Ineligible per group | Audit selection |
| Perbedaan skema penilaian | Context only; bukan causal attribution |

Jika perbedaan performa besar antara AC dan BDE ditemukan, tuliskan sebagai heterogenitas konteks. Jangan menyatakan bahwa bobot penilaian atau kelompok asisten menyebabkan perbedaan tanpa desain kausal.

## 18. Struktur Direktori Implementasi
```text
project-root/
├── README.md
├── PRD_v1_1.md
├── environment.yml
├── configs/
│   ├── data_config.yaml
│   ├── attendance_mapping.yaml
│   ├── feature_config.yaml
│   └── model_config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       ├── master_clean.csv
│       ├── eligible.csv
│       ├── excluded.csv
│       ├── featured_S1.csv
│       ├── featured_S2.csv
│       └── featured_S3.csv
├── src/
│   ├── data_loader.py
│   ├── validation.py
│   ├── attendance.py
│   ├── target.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── tuning.py
│   ├── evaluation.py
│   ├── shap_analysis.py
│   └── reporting.py
├── results/
│   ├── data_quality/
│   ├── metrics/
│   ├── cv/
│   ├── confusion_matrix/
│   ├── predictions/
│   ├── shap/
│   ├── reports/
│   ├── tables/
│   └── figures/
├── logs/
└── tests/
    ├── test_nim_resolution.py
    ├── test_attendance_mapping.py
    ├── test_target.py
    ├── test_leakage.py
    └── test_features.py
```

## 19. Acceptance Criteria / Definition of Done
- [ ] Dataset A/C dan B/D/E dapat dibaca melalui konfigurasi tanpa mengubah kode inti.
- [ ] Source Final/PENILAIAN_UAS dapat di-join berdasarkan NIM tanpa menghasilkan multiple rows yang tidak terjelaskan.
- [ ] Seluruh duplicate NIM pada Final sudah resolved atau excluded dengan alasan yang tercatat.
- [ ] 10 sesi kehadiran dapat direkonstruksi dan audit per sesi tersedia.
- [ ] Aturan >1 absen dan >3 absen diterapkan sebagai flag yang terpisah dan jumlahnya dilaporkan.
- [ ] Final Individu >= 75 menghasilkan Kompeten dan Final Individu < 75 menghasilkan Belum Kompeten.
- [ ] Tidak ada Final/Flowchart/Kodingan/NILAI AKHIR/Predikat/identifier pada X.
- [ ] S1-S3 dapat dibuat ulang tanpa manual editing dataset.
- [ ] 6 eksperimen DT/RF x S1/S2/S3 berjalan dengan split/CV yang sama.
- [ ] CV mean +/- SD dan test metrics tersedia.
- [ ] TreeSHAP tersedia untuk model final.
- [ ] Analisis AC vs BDE tersedia sebagai konteks/robustness.
- [ ] Exclusion log dan data-quality report tersedia.
- [ ] Random seed, versi library, parameter model, mapping kolom, dan mapping attendance tersimpan.

## 20. Risiko Metodologis dan Mitigasi
| Risiko | Dampak | Mitigasi |
| :--- | :--- | :--- |
| Duplicate NIM pada Final | Satu mahasiswa menjadi lebih dari satu observasi | Resolve dari sumber resmi; stop modelling jika unresolved |
| Perbedaan scoring scheme | Distribusi/arti nilai berbeda | Semantic mapping + context analysis AC/BDE |
| Assistant group confounding | Salah tafsir efek skema/asisten | Jangan klaim causal; group analysis only |
| Early exit selection bias | Model hanya mewakili mahasiswa yang bertahan | Laporkan jumlah eksklusi dan batas populasi |
| Attendance >3 | Target lulus dipengaruhi aturan eligibility | Simpan flag dan define analytic population eksplisit |
| Post-final leakage | Performa model palsu tinggi | Forbidden-column audit dan temporal audit |
| Completion salah definisi | Feature noise/invalidity | Gunakan hanya status yang dapat diverifikasi |
| Dataset kecil | Variance tinggi | Model sederhana, CV, beberapa seed/repeated CV bila memungkinkan |
| Class imbalance | Recall kelas prioritas buruk | Laporkan per-class metrics dan pertimbangkan class_weight |

## 21. Ringkasan Metodologi Final
```text
DATA A/C + B/D/E + PENILAIAN UAS
               │
               ▼
    Schema + NIM validation
               │
               ▼
     Resolve duplicate Final
   Reconstruct 10 attendances
               │
               ▼
  absences / early-exit / eligibility
               │
               ▼
  Final >= 75 -> competency label
               │
               ▼
   Drop all post-final fields
               │
               ▼
  Semantic feature engineering
          S1 -> S2 -> S3
               │
               ▼
        Stratified 80:20
        5-fold CV on train
      DT vs Random Forest
         final test set
               │
               ▼
           TreeSHAP
               │
               ▼
      Overall vs AC vs BDE
           Reporting
```

## Lampiran A. Mapping Konfigurasi yang Direkomendasikan
```yaml
scoring_scheme:
  AC:
    classes: [A, C]
    weights:
      attendance: 0.10
      tp: 0.15
      response: 0.15
      report: 0.35
      final: 0.25
  BDE:
    classes: [B, D, E]
    weights:
      tp_response: 0.20
      report: 0.30
      activity: 0.10
      final: 0.30
      attendance: 0.10

attendance:
  AC:
    M1: always_present
    M2_M7: attendance_columns_1_to_6
    M8: presence(individual_score)
    M9: presence(final_score)
    M10: always_present
  BDE:
    M1_M7: attendance_columns
    M8: PENILAIAN_UAS flowchart-session mapping
    M9: PENILAIAN_UAS flowchart-session mapping
    M10: presence(session_10_value)
```
*Mapping M8/M9 BDE harus diisi dengan nama kolom/session sebenarnya dari sheet PENILAIAN_UAS saat implementasi. PRD tidak mengunci nama kolom karena sumber aktual dapat memiliki header berbeda.*

## Lampiran B. Prinsip Interpretasi Hasil
- Random Forest tidak boleh dipastikan menjadi model terbaik sebelum hasil final diperoleh.
- Jika S2/S3 tidak meningkatkan performa, itu adalah temuan yang sah.
- Feature importance/SHAP menunjukkan kontribusi model, bukan sebab-akibat.
- Perbedaan AC vs BDE dilaporkan sebagai heterogenitas konteks kecuali ada desain yang memungkinkan identifikasi kausal.
- Accuracy 84.6% dan F1 0.889 dari 64 mahasiswa pada dokumen eksperimen awal hanya dianggap preliminary; angka final harus berasal dari dataset final setelah preprocessing baru selesai.
