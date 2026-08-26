1. Ubah fokus novelty

Saya tidak akan menjual novelty sebagai:

"menggunakan Random Forest dan TreeSHAP."

Itu bukan novelty yang kuat.

Novelty yang jauh lebih menarik adalah:

Leakage-safe early warning pada data praktikum dengan heterogenitas skema penilaian dan instrumen yang berbeda, menggunakan semantic feature alignment dan evaluasi robustness antar kelompok penilaian.

Secara konseptual penelitianmu menjadi:

Heterogeneous assessment data
↓
semantic harmonization
↓
temporal leakage-safe prediction
↓
early warning
↓
RF vs DT
↓
robustness AC vs BDE
↓
TreeSHAP

Ini jauh lebih "paper-worthy".

Literatur terbaru memang menyoroti pentingnya leakage-safe benchmark, temporal setting, heterogenitas feature schema, dan evaluasi lintas konteks.

2. Wajib lakukan ablation study

Ini menurut saya harus.

Sekarang kamu sudah punya:

S1 = Basic
S2 = Behavioral
S3 = Relational

Jangan hanya menyebut mana yang terbaik.

Tunjukkan kontribusi masing-masing kelompok fitur.

Misalnya:

Eksperimen Fitur
E0 Baseline minimal
S1 Basic
S2 S1 + Behavioral
S3 S2 + Relational

Kemudian laporkan perubahan:

ΔF1(S1 − E0)
ΔF1(S2 − S1)
ΔF1(S3 − S2)

Ini akan menjawab:

Apakah behavioral feature benar-benar menambah informasi?

Dalam hasilmu saat ini, S2 meningkatkan CV dari 0,742 → 0,754, tetapi test tetap 0,733. Itu sendiri sudah merupakan temuan yang menarik.

Literatur terbaru juga menekankan feature analysis dan ablation sebagai bagian dari benchmark yang lebih ketat.

3. Jangan hanya satu train-test split

Ini menurut saya salah satu peningkatan paling penting.

Saat ini:

80% train
20% test

dan satu test set hanya sekitar 24 mahasiswa.

Untuk paper, saya ingin kamu menambahkan Repeated Stratified K-Fold atau beberapa random seeds.

Contoh:

5-fold × 5 repeated seeds

atau minimal:

5-fold CV
seed = 42
123
2024
3407

Kemudian laporkan:

Mean F1
Std F1
95% CI

Bukan hanya:

F1 = 0,733.

Kenapa? Karena dengan dataset kecil, satu split sangat sensitif terhadap komposisi sampel. Bahkan studi terbaru mengenai student success menekankan bahwa perubahan konteks dapat mengubah stabilitas dan importance model.

4. Tambahkan confidence interval

Ini sangat bagus untuk publikasi.

Misalnya:

Random Forest S2
F1 = 0.733
95% CI = [0.68, 0.78]

Sekarang pembaca tahu bukan hanya titik estimasinya, tapi juga ketidakpastiannya.

Untuk dataset 120, ini jauh lebih meyakinkan daripada sekadar angka tunggal.

5. Bandingkan lebih dari dua algoritma, tetapi tetap terkontrol

Kalau targetmu publikasi, DT vs RF saja agak tipis.

Saya akan menambah paling banyak 2 baseline:

Logistic Regression
Random Forest
Gradient Boosting / XGBoost
Decision Tree

Tujuannya bukan membuat puluhan model.

Tujuannya menjawab:

apakah keunggulan RF memang karena ensemble tree, atau model lain dapat melakukan lebih baik?

Studi terbaru dalam student-performance prediction menggunakan berbagai keluarga model dan menunjukkan bahwa tidak ada satu algoritma yang selalu menang.

Tetapi jangan menambahkan model hanya untuk mengejar F1.

Saya akan memilih:

Logistic Regression → linear baseline
Decision Tree → interpretable tree baseline
Random Forest → ensemble
Gradient Boosting/XGBoost → boosted tree

Empat model cukup.

6. Lakukan statistical comparison, bukan cuma "F1 lebih tinggi"

Kalau misalnya:

RF F1 = 0.733
DT F1 = 0.593

jangan berhenti di situ.

Gunakan out-of-fold predictions dan lakukan uji statistik yang sesuai untuk performa model berpasangan, misalnya:

bootstrap confidence interval;
paired permutation test;
atau perbandingan pada out-of-fold predictions.

Ini membuat klaim:

"RF lebih baik"

menjadi jauh lebih kuat.

Paper terbaru yang melakukan benchmark leakage-safe juga memakai pengujian statistik untuk membandingkan model, bukan hanya ranking satu angka.

7. Lakukan robustness experiment AC vs BDE secara formal

Ini sebenarnya bisa menjadi kontribusi utama penelitianmu.

Sekarang:

AC F1 = 0.667
BDE F1 = 0.800

Jangan hanya menjadikannya tabel tambahan.

Jadikan research question tambahan atau secondary analysis:

Apakah performa model tetap konsisten pada dua skema penilaian yang berbeda?

Lalu laporkan:

Overall
AC
BDE

dengan:

F1;
recall;
precision;
specificity;
support;
confidence interval.

Dan jangan mengklaim bahwa perbedaan tersebut disebabkan asisten atau bobot, karena faktor-faktor itu confounded.

8. Bahkan lebih bagus: lakukan leave-group-out validation

Ini yang menurut saya bisa menaikkan kualitas penelitianmu secara signifikan.

Kamu punya:

Group AC
Group BDE

Coba eksperimen:

Train AC → Test BDE

dan:

Train BDE → Test AC

Bukan untuk model produksi, tetapi untuk menguji domain transfer.

Kalau hasilnya:

Train AC → BDE = rendah
Train BDE → AC = rendah

itu menunjukkan model sensitif terhadap assessment context.

Kalau tetap cukup bagus:

model lebih robust terhadap perbedaan skema penilaian.

Ini sangat menarik secara ilmiah.

9. Perkuat early-warning dengan beberapa prediction cutoff

Ini menurut saya sangat potensial.

Sekarang kamu punya satu cutoff:

sebelum Final.

Tetapi penelitian early warning yang kuat biasanya menanyakan:

Seberapa awal mahasiswa berisiko dapat dideteksi?

Literatur terbaru juga menekankan bahwa waktu prediksi merupakan bagian penting dari desain early-warning.

Kamu bisa membuat:

Stage 1 → setelah pertemuan 3
Stage 2 → setelah pertemuan 5
Stage 3 → setelah pertemuan 7

Kemudian:

Cutoff F1 Recall At-Risk
M3 ... ...
M5 ... ...
M7 ... ...

Ini jauh lebih kuat daripada hanya mengatakan:

"Kami memiliki early warning system."

Karena sekarang kamu bisa menjawab:

"Seberapa dini sistem dapat memberikan peringatan yang masih memiliki performa memadai?"

10. TreeSHAP jangan hanya dibuat sebagai gambar

Ini sudah kamu lakukan, tetapi untuk SINTA 2 saya akan tingkatkan.

Jangan hanya:

Beeswarm → feature importance.

Tambahkan:

Global
Mean |SHAP|
Direction
high Attendance → contribution toward Competent
low Attendance → contribution toward At-risk
Local

Pilih:

true positive;
true negative;
false positive;
false negative.

Penelitian XAI pendidikan terbaru juga menekankan bahwa penjelasan global saja bisa menyembunyikan perbedaan individual, sehingga instance-level explanation bernilai untuk pengambilan keputusan.

Ini cocok sekali dengan penelitianmu.

11. Tambahkan calibration

Karena kamu ingin early warning, probabilitas lebih berguna daripada sekadar kelas 0/1.

Misalnya model menghasilkan:

P(Belum Kompeten) = 0.87

Itu bisa dipakai sebagai:

High Risk

Tetapi jika:

P = 0.87

ternyata hanya benar 60% dari kasus serupa, probabilitasnya tidak terkalibrasi.

Jadi pertimbangkan:

Brier score;
reliability curve;
calibration curve.

Penelitianmu akan lebih dekat ke sistem decision support nyata.

12. Jangan oversampling secara sembarangan

Saya tidak otomatis menyarankan SMOTE.

Dengan hanya ±120 mahasiswa dan fitur yang sangat kontekstual, SMOTE dapat menghasilkan observasi sintetis yang secara akademik tidak realistis.

Lebih baik lakukan:

class_weight

atau threshold tuning terlebih dahulu.

Literatur memang menunjukkan oversampling dapat membantu dataset kecil/imbalanced, tetapi itu bukan otomatis pilihan terbaik untuk setiap setting.

13. Kekuatan terbesar kamu sebenarnya ada pada dataset lokal

Ini jangan disepelekan.

Kamu menggunakan data praktikum nyata dari:

Praktikum Logika Pemrograman

dengan:

rekam kehadiran;
TP;
respons;
laporan/asistensi;
dua skema penilaian;
dua kelompok asisten;
Final Individu;
aturan akademik nyata.

Jadi kontribusimu bukan model baru.

Kontribusinya lebih tepat:

methodological framework for leakage-safe early-warning prediction under heterogeneous assessment schemes in practical programming education.

Itu jauh lebih menarik.

14. Kamu perlu literature gap yang benar-benar tajam

Untuk target SINTA 2, jangan Bab II berisi:

RF adalah algoritma ensemble...
DT adalah algoritma tree...
SHAP adalah XAI...

Itu terlalu textbook.

Buat tabel penelitian terdahulu seperti:

Studi Dataset Target Timing Heterogeneous assessment Leakage control XAI Limitation

Kemudian tunjukkan:

mayoritas studi memprediksi performa berdasarkan dataset akademik standar atau agregat; sedikit yang mengevaluasi early-warning berdasarkan cutoff temporal yang ketat sekaligus mempertimbangkan heterogeneous scoring schemes.

Ini harus dibuktikan melalui systematic literature search, bukan asumsi.

Review terbaru memang menunjukkan prediksi performa mahasiswa sudah sangat luas, sementara penggunaan XAI dan kaitannya dengan inovasi pedagogis masih berkembang.

15. Saya akan mengubah RQ penelitianmu

Saat ini RQ-mu sudah cukup bagus, tetapi untuk paper saya akan naikkan menjadi:

RQ1

How accurately can student competency be predicted from pre-final practical activity data under a leakage-safe temporal protocol?

RQ2

How do Decision Tree, Random Forest, and Gradient Boosting compare under heterogeneous assessment schemes?

RQ3

Does behavioral feature engineering improve early-warning performance beyond basic activity features?

RQ4

How robust is the prediction performance across the AC and BDE assessment contexts?

RQ5

Which pre-final behavioral features contribute most to the model's predictions according to TreeSHAP?

Ini sudah membentuk paper yang jauh lebih kuat.

16. Struktur eksperimen yang saya rekomendasikan

Kalau saya yang mengerjakan penelitianmu, saya akan membuat:

EXPERIMENT 1
Baseline
────────────────
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting

EXPERIMENT 2
Feature ablation
────────────────
Basic
Basic + Behavioral
Basic + Behavioral + Relational

EXPERIMENT 3
Temporal robustness
────────────────
Cutoff M3
Cutoff M5
Cutoff M7
Pre-final

EXPERIMENT 4
Context robustness
────────────────
Overall
AC
BDE
Train AC → Test BDE
Train BDE → Test AC

EXPERIMENT 5
XAI
────────────────
Global SHAP
Beeswarm
Local TP/TN/FP/FN

EXPERIMENT 6
Probability
────────────────
Calibration
Risk threshold

Tidak perlu menjalankan semuanya dengan puluhan hyperparameter.

17. Dari semua itu, mana yang paling prioritas?

Kalau waktumu terbatas, saya akan urutkan:

Wajib

1. Leakage-safe preprocessing final

2. Repeated CV / confidence interval

3. Baseline model tambahan

4. Ablation S1/S2/S3

5. Formal AC vs BDE robustness

6. Confusion matrix + per-class metrics

7. TreeSHAP global + local

Sangat bagus untuk menaikkan kualitas paper

8. Multiple temporal cutoffs

9. Cross-group validation

10. Calibration

Target penelitianmu jangan lagi "menaikkan F1"

Ini poin terpenting.

Jangan mengejar:

F1 > 0.80

sebagai target buatan.

Targetmu harus:

membangun evaluasi early-warning yang valid, leakage-safe, robust terhadap heterogenitas penilaian, dan explainable.

Bisa saja hasil akhirnya:

RF = 0.733

tetapi kalau kamu bisa menunjukkan:

tidak ada leakage;
performa stabil pada repeated CV;
RF mengungguli baseline secara signifikan;
behavioral features memang memberikan kontribusi;
robustness AC/BDE dianalisis;
seberapa dini model dapat bekerja;
calibration/risk threshold valid;
SHAP konsisten;
hasil didukung literatur terbaru;

maka paper-mu jauh lebih kuat daripada paper dengan accuracy 90% tetapi metodologinya lemah.

Literatur terbaru bahkan menunjukkan bahwa leakage-safe evaluation dan domain shift dapat menyebabkan performa turun secara nyata ketika diuji pada data yang benar-benar independen; itu justru bagian penting dari validitas model.

Penilaian saya sekarang

Potensi metodologi: tinggi.
Novelty saat ini: sedang.
Kualitas eksperimen saat ini: cukup baik, tetapi belum cukup kuat untuk target SINTA 2.
Potensi setelah perbaikan: cukup kuat untuk ditargetkan ke jurnal SINTA 2, dengan catatan kualitas jurnal spesifik, scope, reviewer, dan novelty tetap tidak bisa dijamin hanya dari metodologi.
