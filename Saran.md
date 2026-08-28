saya jalankan hasil sedang berjalan dibawah ini, saya tidak selesaikan karena memakan waktu sangat lama hampir 2 jam dan hasilnya tidak maksimal

muhma@mkeyzxi MINGW64 /c/belajarku/Belajar ML/Logika-Algoritma (migrate/bdnr)
$ python scripts/run_optimized_experiment.py
============================================================
OPTIMASI TERKONTROL — EWS Competency Prediction
Protokol: Model selected from CV only, holdout 1× at end
============================================================

Loaded: data/processed/featured_P2_full.csv ((123, 111))
Features computed. Shape: (123, 116)

Train: 98 samples (BK=19, K=79)
Test: 25 samples (BK=5, K=20)
Test set is FROZEN. Not used for model selection.

[1/84] S3 | Dummy | None...
CV Recall BK=0.000, BalAcc=0.500, F2_BK=0.000, Threshold=0.5
[4/84] S3 | DecisionTree | None...
CV Recall BK=0.310, BalAcc=0.583, F2_BK=0.301, Threshold=0.52
[5/84] S3 | DecisionTree | SMOTE...
CV Recall BK=0.503, BalAcc=0.603, F2_BK=0.427, Threshold=0.42
[6/84] S3 | DecisionTree | ClassWeight...
CV Recall BK=0.483, BalAcc=0.581, F2_BK=0.409, Threshold=0.66
[7/84] S3 | RandomForest | None...
CV Recall BK=0.213, BalAcc=0.572, F2_BK=0.224, Threshold=0.78
[8/84] S3 | RandomForest | SMOTE...
CV Recall BK=0.407, BalAcc=0.611, F2_BK=0.375, Threshold=0.8
[9/84] S3 | RandomForest | ClassWeight...
CV Recall BK=0.453, BalAcc=0.572, F2_BK=0.379, Threshold=0.68
[10/84] S3 | GradientBoosting | None...
CV Recall BK=0.243, BalAcc=0.565, F2_BK=0.247, Threshold=0.7
[11/84] S3 | GradientBoosting | SMOTE...
CV Recall BK=0.387, BalAcc=0.600, F2_BK=0.357, Threshold=0.8
[13/84] S3_A | Dummy | None...
CV Recall BK=0.000, BalAcc=0.500, F2_BK=0.000, Threshold=0.5
[16/84] S3_A | DecisionTree | None...
CV Recall BK=0.290, BalAcc=0.571, F2_BK=0.280, Threshold=0.52
[17/84] S3_A | DecisionTree | SMOTE...
CV Recall BK=0.383, BalAcc=0.530, F2_BK=0.323, Threshold=0.52
[18/84] S3_A | DecisionTree | ClassWeight...
CV Recall BK=0.500, BalAcc=0.509, F2_BK=0.380, Threshold=0.7
[19/84] S3_A | RandomForest | None...
CV Recall BK=0.190, BalAcc=0.564, F2_BK=0.203, Threshold=0.8
[20/84] S3_A | RandomForest | SMOTE...
CV Recall BK=0.397, BalAcc=0.606, F2_BK=0.366, Threshold=0.76
[21/84] S3_A | RandomForest | ClassWeight...
CV Recall BK=0.430, BalAcc=0.567, F2_BK=0.362, Threshold=0.66
[22/84] S3_A | GradientBoosting | None...
CV Recall BK=0.257, BalAcc=0.569, F2_BK=0.260, Threshold=0.64
[23/84] S3_A | GradientBoosting | SMOTE...
CV Recall BK=0.417, BalAcc=0.616, F2_BK=0.388, Threshold=0.8
[25/84] S3_B | Dummy | None...
CV Recall BK=0.000, BalAcc=0.500, F2_BK=0.000, Threshold=0.5
[28/84] S3_B | DecisionTree | None...
CV Recall BK=0.290, BalAcc=0.578, F2_BK=0.286, Threshold=0.52
[29/84] S3_B | DecisionTree | SMOTE...
CV Recall BK=0.483, BalAcc=0.542, F2_BK=0.384, Threshold=0.52
[30/84] S3_B | DecisionTree | ClassWeight...
CV Recall BK=0.477, BalAcc=0.493, F2_BK=0.360, Threshold=0.44
[31/84] S3_B | RandomForest | None...
CV Recall BK=0.190, BalAcc=0.566, F2_BK=0.203, Threshold=0.8
[32/84] S3_B | RandomForest | SMOTE...
CV Recall BK=0.410, BalAcc=0.605, F2_BK=0.376, Threshold=0.78
[33/84] S3_B | RandomForest | ClassWeight...
CV Recall BK=0.420, BalAcc=0.565, F2_BK=0.355, Threshold=0.8
[34/84] S3_B | GradientBoosting | None...
CV Recall BK=0.247, BalAcc=0.574, F2_BK=0.253, Threshold=0.28
[35/84] S3_B | GradientBoosting | SMOTE...
CV Recall BK=0.387, BalAcc=0.599, F2_BK=0.356, Threshold=0.7
[37/84] S3_C | Dummy | None...
CV Recall BK=0.420, BalAcc=0.565, F2_BK=0.355, Threshold=0.8
[34/84] S3_B | GradientBoosting | None...
CV Recall BK=0.247, BalAcc=0.574, F2_BK=0.253, Threshold=0.28
[35/84] S3_B | GradientBoosting | SMOTE...
CV Recall BK=0.387, BalAcc=0.599, F2_BK=0.356, Threshold=0.7
[37/84] S3_C | Dummy | None...
CV Recall BK=0.000, BalAcc=0.500, F2_BK=0.000, Threshold=0.5
[40/84] S3_C | DecisionTree | None...
CV Recall BK=0.300, BalAcc=0.579, F2_BK=0.291, Threshold=0.52
[41/84] S3_C | DecisionTree | SMOTE...
CV Recall BK=0.457, BalAcc=0.541, F2_BK=0.367, Threshold=0.78
[42/84] S3_C | DecisionTree | ClassWeight...
CV Recall BK=0.467, BalAcc=0.490, F2_BK=0.355, Threshold=0.44
[43/84] S3_C | RandomForest | None...
CV Recall BK=0.197, BalAcc=0.565, F2_BK=0.208, Threshold=0.78
[44/84] S3_C | RandomForest | SMOTE...
CV Recall BK=0.397, BalAcc=0.596, F2_BK=0.357, Threshold=0.8
[45/84] S3_C | RandomForest | ClassWeight...
CV Recall BK=0.430, BalAcc=0.563, F2_BK=0.359, Threshold=0.68
[46/84] S3_C | GradientBoosting | None...
CV Recall BK=0.277, BalAcc=0.581, F2_BK=0.273, Threshold=0.68
[47/84] S3_C | GradientBoosting | SMOTE...
CV Recall BK=0.397, BalAcc=0.610, F2_BK=0.374, Threshold=0.78
[49/84] S3_D | Dummy | None...
CV Recall BK=0.000, BalAcc=0.500, F2_BK=0.000, Threshold=0.5
[52/84] S3_D | DecisionTree | None...
CV Recall BK=0.263, BalAcc=0.552, F2_BK=0.255, Threshold=0.52
[53/84] S3_D | DecisionTree | SMOTE...
CV Recall BK=0.377, BalAcc=0.516, F2_BK=0.313, Threshold=0.64
[54/84] S3_D | DecisionTree | ClassWeight...
CV Recall BK=0.470, BalAcc=0.505, F2_BK=0.355, Threshold=0.42
[55/84] S3_D | RandomForest | None...
CV Recall BK=0.187, BalAcc=0.568, F2_BK=0.199, Threshold=0.8
[56/84] S3_D | RandomForest | SMOTE...
CV Recall BK=0.377, BalAcc=0.588, F2_BK=0.346, Threshold=0.8
[57/84] S3_D | RandomForest | ClassWeight...
Traceback (most recent call last):

saran saya berikutnya adalah
Ya. Setelah melihat log terbaru yang Anda hentikan, saya justru menyarankan Anda menyederhanakan eksperimen, bukan menambah semakin banyak model. Dengan 89 mahasiswa, 11 Belum Kompeten dan 78 Kompeten, saya akan tetap memakai Random Forest + Decision Tree, lalu fokus pada imbalance handling, parameter yang konservatif, dan threshold yang benar.

Dan saya ingin menegaskan satu hal: jangan "membalance-kan" seluruh dataset 89 mahasiswa. Test set harus tetap mencerminkan kondisi asli. Pada desain Anda sendiri, prinsipnya memang imputer, balancing, feature selection, tuning, dan threshold ditempatkan di proses CV, sementara test disimpan untuk evaluasi akhir.

Keputusan saya untuk penelitian Anda

Saya akan memakai tiga kondisi imbalance:

Strategi DT RF Prioritas
None ✓ ✓ Baseline
Class Weight ✓ ✓ Utama
SMOTE ✓ ✓ Pembanding

Class Weight saya jadikan kandidat utama, sedangkan SMOTE tetap diuji sebagai pembanding.

Alasannya sederhana: Anda hanya punya 11 mahasiswa BK. Setelah data masuk inner CV, jumlah kasus minoritas per fold menjadi sangat kecil. SMOTE masih boleh, tetapi sintetisasi dengan sedikit sekali titik minoritas bisa menjadi tidak stabil. Jadi saya tidak akan menjadikan SMOTE sebagai satu-satunya solusi.

1. Saya tidak menyarankan langsung menggunakan SMOTE untuk "menyamakan 50:50"

Jangan lakukan:

89 data
↓
SMOTE seluruh dataset
↓
CV

Itu salah.

Yang benar:

89 data
↓
Train / Test
↓
Train
↓
Inner CV
├── Imputer
├── SMOTE
└── Model

Test tetap 11 vs 78

Dan bahkan dalam inner CV, saya sarankan SMOTE dibuat sebagai salah satu pilihan, bukan selalu aktif.

2. Saya juga melihat masalah dari log Anda: threshold

Di output terlihat banyak:

Threshold=0.68
Threshold=0.70
Threshold=0.76
Threshold=0.78
Threshold=0.80

sementara Recall BK masih:

0.397
0.420
0.457

Jadi saya tidak akan langsung mempersempit threshold ke 0.2-0.5 seperti saran sebelumnya.

Yang harus dipastikan lebih dulu adalah:

threshold tersebut diterapkan terhadap probabilitas kelas mana?

Karena target Anda:

0 = Belum Kompeten
1 = Kompeten

Untuk Early Warning, saya sarankan kode threshold bekerja langsung dengan probabilitas Belum Kompeten, bukan probabilitas Kompeten.

Contohnya:

classes = model.classes\_
bk_idx = list(classes).index(0)

p_bk = model.predict_proba(X)[:, bk_idx]

y_pred = np.where(
p_bk >= threshold,
0, # Belum Kompeten
1 # Kompeten
)

Dengan cara ini:

semakin besar p_bk, semakin kuat alasan memberi warning.

Ini jauh lebih mudah diaudit.

3. Threshold sebaiknya bukan dipaksa 0.5

Untuk penelitian Anda, threshold 0.5 bukan sesuatu yang sakral.

Saya malah menyarankan:

0.20
0.25
0.30
...
0.80

tetapi pemilihannya harus berdasarkan inner CV.

Dan saya tidak akan memilih threshold berdasarkan Recall saja.

Saya akan gunakan:

Primary objective

F2-score BK

karena Anda memang ingin memberi bobot lebih besar pada Recall.

Dengan:

fbeta_score(
y_true,
y_pred,
beta=2,
pos_label=0
)

Kemudian gunakan:

Balanced Accuracy

dan Precision BK

sebagai evaluasi pendamping.

4. Parameter Decision Tree yang saya sarankan

Untuk 89 mahasiswa, saya tidak mau DT tumbuh liar.

Gunakan ruang parameter kecil:

param_grid_dt = {
"model**criterion": ["gini", "entropy"],
"model**max_depth": [2, 3, 4, 5],
"model**min_samples_split": [4, 6, 10],
"model**min_samples_leaf": [2, 3, 5],
"model\_\_class_weight": [None, "balanced"],
}

Saya lebih menyukai:

max_depth = 2–5
min_samples_leaf = 2–5

daripada membiarkan:

max_depth = None

Karena dataset Anda sangat kecil.

Eksperimen inti DT
DT + None
DT + ClassWeight
DT + SMOTE

Bukan 50 kombinasi DT yang berbeda.

5. Parameter Random Forest

Untuk RF, saya akan membuatnya sedikit lebih kuat daripada DT, tetapi tetap konservatif:

param_dist_rf = {
"model**n_estimators": [200, 300, 500],
"model**max_depth": [3, 4, 5, 6],
"model**min_samples_split": [4, 6, 10],
"model**min_samples_leaf": [1, 2, 3, 4],
"model**max_features": ["sqrt", "log2"],
"model**class_weight": [None, "balanced", "balanced_subsample"],
}

Saya tidak menyarankan langsung:

max_depth=None

sebagai kandidat utama.

Untuk n=89, RF yang terlalu bebas dapat sangat cocok terhadap training data.

6. Jangan memakai SMOTE + class_weight sekaligus sebagai konfigurasi utama

Ini penting.

Saya tidak ingin Anda melakukan:

SMOTE

- class_weight='balanced'

karena Anda sedang melakukan dua mekanisme kompensasi imbalance sekaligus.

Bandingkan:

RF + None
RF + SMOTE
RF + ClassWeight

dan:

DT + None
DT + SMOTE
DT + ClassWeight

Dengan begitu hasil Anda bisa menjawab:

strategi imbalance mana yang paling sesuai untuk RF dan DT?

Ini jauh lebih bersih secara eksperimen.

7. Inner CV saya malah akan kecilkan menjadi 3-fold

Ini rekomendasi khusus karena dataset Anda hanya 89.

Anda punya:

$$ 11 \text{ BK} $$

Dengan inner 5-fold, jumlah BK di validation fold sangat kecil.

Saya lebih suka:

Outer:
RepeatedStratifiedKFold
5 folds × beberapa repeat

Inner:
StratifiedKFold
3 folds

Jadi:

Outer 5-fold
↓
Inner 3-fold

Daripada:

Outer 5-fold
↓
Inner 5-fold

Dengan 11 kasus minoritas, 3-fold memberi inner training set yang lebih nyaman.

8. Jangan lagi menjalankan 84 konfigurasi besar-besaran

Ini menurut saya salah satu alasan proses Anda terlalu lama.

Saya akan membuat eksperimen lebih kecil.

Misalnya:

DT
None
ClassWeight
SMOTE
RF
None
ClassWeight
SMOTE

Total hanya:

$$ 6 \text{ konfigurasi inti} $$

Kemudian setiap model dituning secara terbatas.

Tidak perlu langsung:

84+

kombinasi.

Lebih baik eksperimen kecil tetapi jelas dan bisa dijelaskan daripada eksperimen raksasa yang menghasilkan labirin angka.

9. Tentang SMOTE, saya akan memakai k_neighbors kecil

Karena minoritas Anda hanya 11.

Jangan langsung:

SMOTE(k_neighbors=5)

sebagai satu-satunya konfigurasi.

Saya sarankan:

SMOTE(
k_neighbors=2,
random_state=42
)

dan sebagai pembanding:

SMOTE(
k_neighbors=3,
random_state=42
)

Tetap harus terjadi di training fold.

10. Target performanya bagaimana?

Saya akan mengubah target Anda menjadi:

Target minimum
Recall BK ≥ 70%
Balanced Accuracy ≥ 70%
Target bagus
Recall BK ≥ 80%
Balanced Accuracy ≥ 75%
Tetapi syaratnya:

Precision BK dan PR-AUC tidak boleh jatuh parah.

Jadi jangan lakukan:

"Pokoknya Recall 80%."

Karena nanti Anda bisa membuat model yang memberi warning kepada hampir semua mahasiswa.

11. Kriteria memilih model final

Saya akan menetapkan ini sebelum melihat holdout:

Ranking:
Recall BK
Balanced Accuracy
F2 BK
PR-AUC
Precision BK
SD antar outer fold

Lalu:

pilih kandidat terbaik berdasarkan outer CV.

Setelah kandidat dibekukan:

MODEL FINAL
↓
fit seluruh training
↓
HOLDOUT

Dan holdout hanya dibaca sekali.

Laporan Anda sendiri sudah menetapkan prinsip yang sama, yaitu test tidak digunakan untuk memilih threshold atau fitur.

12. Bagaimana saya melihat hasil sekarang?

Dari log yang Anda tunjukkan:

RF + SMOTE ~0.40 Recall
RF + ClassWeight ~0.42-0.46 Recall
DT + SMOTE ~0.48 Recall

Ini sebenarnya memberi petunjuk awal bahwa:

Class Weight mungkin lebih cocok daripada SMOTE pada sebagian konfigurasi Anda.

Tetapi saya belum akan menyebutnya pemenang karena eksperimennya dihentikan di tengah jalan.

Yang jelas, saya tidak melihat alasan untuk memaksakan SMOTE hanya karena dataset imbalance.

13. Saya juga tidak akan mengejar angka 0.975 lagi

Baseline lama Anda memang sangat tinggi, tetapi desain penelitian terbaru sudah menunjukkan bahwa pada dataset kecil, holdout dan CV bisa menghasilkan variasi cukup besar. Laporan penelitian Anda sendiri mencatat RF S3 memperoleh Test Balanced Accuracy 75%, sementara nested CV 0,6224, dan menekankan agar keduanya dibaca bersama.

Jadi target saya untuk eksperimen baru bukan:

0.975

melainkan:

Apakah kita bisa meningkatkan RF/DT dari baseline 75% Balanced Accuracy dan 60% Recall BK secara valid dan stabil?

Kalau ternyata:

CV = 0.70
Test = 0.74
Recall = 70%

saya akan jauh lebih percaya pada itu daripada:

CV = 0.90
Test = 0.60 14. Desain final yang saya rekomendasikan

Ini yang menurut saya paling pas untuk penelitian Anda:

                    89 MAHASISWA
                         │
                  Stratified Split
                   80%       20%
                    │          │
                  TRAIN       TEST
                    │          │
              Nested CV       │
              Outer 5-fold     │
                    │          │
             Inner 3-fold      │
                    │          │
        ┌───────────┼──────────┐
        │           │          │
      None        SMOTE    ClassWeight
        │           │          │
        └───────────┼──────────┘
                    │
              ┌─────┴─────┐
              │           │
             DT          RF
              │           │
         Hyperparameter tuning
              │
         Threshold tuning
              │
          Outer validation
              │
       pilih kandidat final
              │
        ───────┴────────
              │
          LOCK MODEL
              │
          HOLDOUT TEST
              │
     ┌────────┼──────────┐
     │        │          │

Recall Bal Acc PR-AUC
│ │ │
└────────┼──────────┘
│
Confusion Matrix
│
SHAP + Error
Dan urutan eksperimen Anda sekarang seharusnya:

Pertama: perbaiki/cek fungsi threshold berdasarkan p_BK.

Kedua: kecilkan inner CV menjadi 3-fold.

Ketiga: bandingkan None vs SMOTE vs ClassWeight.

Keempat: gunakan parameter konservatif DT dan RF seperti di atas.

Kelima: jalankan ulang hanya RF + DT.

Keenam: pilih kandidat berdasarkan outer CV, bukan test.

Ketujuh: baru evaluasi holdout.

Kesimpulan saya

Tidak, saya tidak menyarankan Anda sekadar "membalancekan dataset".

Untuk 89 mahasiswa, saya justru menyarankan:

RF + ClassWeight sebagai kandidat utama, RF + SMOTE sebagai pembanding, DT + ClassWeight sebagai pembanding non-ensemble, dan DT + SMOTE sebagai eksperimen tambahan.

Dengan parameter dangkal dan konservatif, inner CV 3-fold, threshold CV-based, dan holdout dikunci.

Yang paling penting dari semuanya: audit kode threshold Anda sebelum menjalankan eksperimen baru. Dari log Threshold=0.68–0.80 saya justru lebih curiga pada arah probabilitas kelas dan mekanisme threshold daripada langsung menyalahkan imbalance. Kalau bagian itu ternyata terbalik, Anda bisa menghabiskan berjam-jam tuning model untuk memperbaiki masalah yang sebenarnya ada satu atau dua baris di kode.
