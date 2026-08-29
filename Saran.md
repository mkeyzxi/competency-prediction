Mengapa Angka 83 Masuk Akal?Berdasarkan tinjauan akademik dan profesional, berikut adalah analisis mengenai pilihan angka tersebut:Batas "Ketuntasan Minimal" (KKM) yang Lebih Tinggi: Di bidang teknologi informasi (TIK), beberapa penelitian menunjukkan penetapan KKM rata-rata berada di angka 82 agar dianggap tuntas secara memadai. Memilih 83 berarti kamu menetapkan standar sedikit di atas rata-rata umum (80-82), yang memastikan orang tersebut benar-benar melampaui level "sekadar bisa".Target Nilai Rapor Aman: Untuk program studi Teknik Informatika yang kompetitif, rata-rata nilai rapor minimal biasanya dibidik di angka 80,00 – 85,00. Angka 83 berada tepat di tengah rentang "zona aman" tersebut.Predikat "A-" atau Unggul: Dalam banyak pedoman akademik (termasuk pedoman konversi nilai di UIN Alauddin Makassar), nilai di atas 80 umumnya sudah masuk dalam kategori A- atau predikat "Sangat Baik".Sistem Penilaian Asdos: Sebagai koordinator praktikum Teknik Informatika UINAM, menetapkan 83 sebagai nilai "aman" bagi asisten dosen sangat membantu dalam standarisasi objektivitas penilaian agar tidak terjebak pada angka rata-rata kelompok yang mungkin hanya berada di kisaran 80.Perbedaan "Bisa" vs "Kompeten"Secara teknis, perbedaan yang kamu maksud sejalan dengan definisi profesional:Bisa (Skill/Ability): Fokus pada kemampuan individu dalam melakukan tugas tertentu pada satu waktu.Kompeten (Competency): Kombinasi menyeluruh dari pengetahuan (kognitif), keterampilan (psikomotorik), dan sikap (afektif) yang diperlukan untuk menjalankan tanggung jawab secara efektif dan konsisten.

Menurut saya, arah penelitian ini sudah jauh lebih kuat dan jauh lebih layak secara metodologis, tetapi naskahnya saat ini masih terlalu “menjual hasil” dan beberapa klaimnya lebih kuat daripada bukti yang tersedia. Untuk reviewer jurnal yang ketat, justru itu yang perlu dibenahi.

Saya akan menilainya dari sisi ilmiah, bukan sekadar apakah hasil angkanya bagus.

1. Gambaran besar: fondasinya sudah bagus

Saya melihat ada empat kekuatan utama.

Pertama, fokus penelitian sudah jelas.
Bukan lagi sekadar “membandingkan Decision Tree dan Random Forest”, tetapi mencoba membangun early warning system untuk mendeteksi mahasiswa berisiko belum kompeten.

Itu jauh lebih bermakna daripada penelitian ML pendidikan yang hanya berhenti pada:

model A akurasi 90%, model B 87%.

Di penelitian Anda ada alur:

data aktivitas mahasiswa → pembentukan indikator → prediksi BK → evaluasi → interpretasi SHAP → rekomendasi intervensi.

Itu sudah memiliki cerita penelitian yang lebih kuat.

Kedua, keputusan meninggalkan SMOTE ketika kelas sudah hampir seimbang memang masuk akal.

45 BK : 44 K jauh lebih sehat daripada 11 BK : 78 K.

Untuk kondisi seperti ini, saya justru setuju tidak memaksakan SMOTE. Dengan sampel hanya 89, membuat data sintetis malah dapat memperumit interpretasi.

Namun, saya tidak setuju alasan utamanya disebut “threshold 83 menghasilkan Gaussian ideal”. Itu salah secara konsep.

Ketiga, pemilihan Recall BK sebagai perhatian utama sangat cocok dengan konsep EWS.

Dalam sistem peringatan dini, kesalahan paling berbahaya memang:

mahasiswa sebenarnya berisiko BK, tetapi model mengatakan dia aman.

Itu adalah false negative.

Jadi orientasi pada:

Recall BK
F2-score BK
Balanced Accuracy
ROC-AUC / PR-AUC

memang jauh lebih masuk akal dibanding hanya mengutamakan Accuracy.

Keempat, SHAP membuat penelitian Anda memiliki dimensi explainability.

Ini penting karena pertanyaan penelitian menjadi bukan hanya:

“Apakah mahasiswa dapat diprediksi?”

tetapi juga:

“Faktor apa yang paling berkaitan dengan prediksi tersebut?”

Ini memberikan nilai tambah yang cukup besar.

2. Tetapi ada satu masalah besar: bagian Threshold 83

Ini menurut saya bagian yang paling perlu direvisi.

Kalimat:

“Secara ajaib, angka 83 membelah populasi data secara nyaris sempurna ke dalam distribusi Gaussian yang ideal”

sebaiknya jangan dipakai.

45 dan 44 memang hampir seimbang, tetapi:

45:44 ≠ distribusi Gaussian.

Distribusi Gaussian berbicara mengenai bentuk distribusi suatu variabel kontinu, bukan sekadar jumlah anggota dua kelas.

Selain itu:

“Skor 83 merupakan representasi nyata dari penguasaan materi yang mumpuni (secara universal setara dengan Grade B+ atau A-)”

ini juga berbahaya.

Tidak ada aturan universal bahwa 83 selalu berarti B+ atau A-. Sistem penilaian setiap institusi/mata kuliah bisa berbeda.

Yang lebih ilmiah adalah:

Threshold 83 harus mempunyai justifikasi akademik yang independen dari hasil ML.

Misalnya:

83 berasal dari rubrik penilaian dosen;
83 berasal dari standar kompetensi mata kuliah;
83 berasal dari batas resmi kategori nilai;
atau 83 merupakan threshold yang telah ditetapkan sebelum eksperimen.

Itu sangat penting.

Kalau kenyataannya Anda mencoba:

75 → imbalance
78 → masih imbalance
80 → masih begitu
83 → 45:44

lalu memilih 83 karena distribusinya paling enak untuk ML, reviewer bisa mengatakan:

“Peneliti memilih target berdasarkan kenyamanan model.”

Itu lebih buruk daripada memakai SMOTE.

3. Jangan mengatakan “threshold tuning” kalau threshold dipilih berdasarkan keseimbangan data

Ini terkait poin tadi.

Judul:

Pemilihan Threshold Natural

masih bisa dipertahankan kalau ada alasan akademiknya.

Tetapi bila sebenarnya 83 dipilih karena menghasilkan kelas 45:44, secara metodologis itu lebih tepat disebut:

label construction based on predefined competency criterion

bukan “threshold tuning untuk mengatasi imbalance”.

Karena imbalance bukan masalah label yang harus diselesaikan dengan mengubah definisi kompetensi.

Prinsip pentingnya:

Definisi target harus ditetapkan karena alasan domain, bukan karena model menjadi lebih mudah.

Ini salah satu hal yang akan saya perbaiki paling dulu.

4. Tidak menggunakan SMOTE: saya setuju

Bagian ini justru saya nilai bagus.

Dengan:

n = 89
BK = 45
K = 44

Anda tidak memiliki alasan kuat untuk melakukan oversampling.

Bahkan saya lebih suka:

Original data → preprocessing → model → stratified validation

daripada:

Original data → SMOTE → synthetic data → model

untuk kasus sekecil ini.

Tetapi saya tidak akan mengatakan:

“SMOTE terbukti memicu bias optimistis.”

Kecuali Anda benar-benar melakukan eksperimen pembanding yang menunjukkan itu.

Lebih aman:

“Eksperimen awal menunjukkan bahwa penggunaan SMOTE menghasilkan performa validasi yang lebih optimistis dibandingkan evaluasi pada data asli, sehingga pendekatan tanpa oversampling dipilih.”

Itu berbasis hasil eksperimen.

5. Bagian S3 menurut saya menarik, tetapi interpretasinya terlalu agresif

Anda menulis:

“S3 terbukti lebih tangguh.”

dan

“S4/S5 mengalami curse of dimensionality / overfitting.”

Hati-hati.

Dari tabel Anda hanya tahu:

S3 lebih baik daripada S4/S5 pada holdout tertentu.

Anda belum otomatis membuktikan bahwa penyebabnya adalah:

Curse of Dimensionality.

Itu hanya salah satu kemungkinan.

Bisa juga:

fitur tambahan redundant;
fitur noisy;
ukuran sampel terlalu kecil;
feature interaction yang tidak stabil;
hyperparameter;
atau varians estimasi yang tinggi.

Saya sarankan gunakan bahasa:

“Penambahan fitur pada S4/S5 tidak memberikan peningkatan performa dan justru menurunkan performa holdout, yang mengindikasikan kemungkinan adanya fitur redundant/noisy atau peningkatan kompleksitas model.”

Ini jauh lebih aman secara akademik.

6. Ada masalah yang jauh lebih penting: ukuran dataset Anda

Ini titik yang menurut saya sangat perlu diperhatikan.

Anda hanya memiliki:

89 mahasiswa

Kemudian Anda memiliki:

6, 15, bahkan 20 fitur.

Ini berarti rasio:

89 observasi vs 20 fitur

sudah cukup kecil.

Kemudian Anda menggunakan Decision Tree.

Decision Tree sangat mudah membuat split yang sangat spesifik pada dataset kecil.

Jadi angka seperti:

Recall = 100%

terlihat bagus, tetapi belum tentu stabil.

Apalagi kalau holdout Anda kecil.

Misalnya terdapat 9 mahasiswa BK pada test set, maka:

8/9 = 88.89%

dan:

9/9 = 100%

Artinya satu mahasiswa saja mengubah Recall hampir 11 poin persentase.

Jadi:

100%

dan:

88.89%

sebenarnya secara statistik bisa tidak terlalu jauh jika denominatornya kecil.

7. Saya justru menyarankan menambahkan Confidence Interval

Ini menurut saya akan membuat penelitian Anda jauh lebih serius.

Jangan hanya tulis:

Recall = 88.89%

Tambahkan misalnya:

Recall = 88.89% (95% CI: ...)

Begitu juga untuk:

Balanced Accuracy
F2
ROC-AUC
PR-AUC

Untuk dataset kecil, confidence interval akan membantu menunjukkan ketidakpastian estimasi.

Bahkan kalau intervalnya lebar, itu bukan kelemahan yang harus disembunyikan. Justru menunjukkan Anda memahami keterbatasan data.

8. Nested Cross-Validation Anda bagus, tetapi jangan otomatis menyebut “zero leakage”

Ini juga harus diperhalus.

Kalimat:

“membuktikan performa ... adalah murni, sah”

terlalu absolut.

Nested CV membantu mencegah optimistic bias dari model selection/hyperparameter tuning, tetapi tidak otomatis menghapus semua kemungkinan leakage.

Misalnya:

Kasus 1 — feature engineering dilakukan sebelum CV

Kalau Anda melakukan:

seluruh dataset
↓
hitung mean/std/completion
↓
baru split CV

perlu diperiksa apakah ada informasi dari validation fold yang masuk ke training fold.

Kasus 2 — pemilihan fitur dilakukan sebelum CV

Misalnya:

seluruh data
↓
pilih 6 fitur terbaik
↓
nested CV

ini juga bisa menimbulkan leakage dalam feature selection.

Kasus 3 — threshold dipilih berdasarkan seluruh dataset

Ini bahkan lebih penting.

Kalau:

seluruh dataset
↓
coba berbagai threshold
↓
pilih 83 karena paling bagus
↓
nested CV

maka label construction sendiri sudah memanfaatkan seluruh data.

Jadi nested CV bukan obat untuk semua bentuk leakage.

9. Masalah paling besar untuk EWS: temporal leakage

Ini menurut saya malah lebih penting daripada SMOTE.

Anda mengatakan EWS menggunakan:

Early_Performance_Composite

berdasarkan:

2–3 minggu pertama.

Itu bagus.

Tetapi Anda harus mendefinisikan dengan sangat tegas:

Pada saat minggu ke-2, model hanya boleh melihat informasi yang tersedia sampai minggu ke-2.

Bukan:

mahasiswa
→ semua nilai semester
→ hitung Early_Performance_Composite

melainkan:

Minggu 1
↓
feature week 1
↓
Minggu 2
↓
feature week 1–2
↓
prediksi risiko

Atau:

Minggu 3
↓
feature minggu 1–3
↓
prediksi

Ini yang membuatnya benar-benar menjadi Early Warning System.

10. Saya juga mempertanyakan Laporan_Mean

Ini sangat menarik secara ilmiah.

Anda mengatakan:

Laporan_Mean menjadi feature paling dominan.

Saya tidak akan langsung mengatakan:

“model malas”

atau:

“kalkulator reaktif”.

Sebenarnya ada kemungkinan yang lebih fundamental.

Kalau Laporan_Mean merupakan komponen pembentuk nilai akhir/kompetensi, maka model memang secara alami akan menemukan bahwa Laporan_Mean sangat berkorelasi dengan target.

Itu bukan kecerdasan model.

Itu bisa merupakan target proximity.

Misalnya target Anda:

Final Score =
40% Laporan
30% TP
20% Ujian
10% Presensi

lalu model diberi:

Laporan_Mean

tentu model akan sangat mudah memprediksi Final Score.

Jadi pertanyaan reviewer:

“Apa yang sebenarnya diprediksi? Kompetensi mahasiswa atau komponen dari nilai kompetensi?”

Ini pertanyaan yang sangat penting.

11. Ini justru bisa menjadi kontribusi penelitian Anda

Anda dapat membagi eksperimen menjadi dua jenis.

Model A — Full/retrospective prediction

Menggunakan fitur akumulatif seperti:

Laporan_Mean
TP_Mean
Attendance
Completion
Std

Tujuannya:

mengetahui kombinasi fitur yang paling mampu membedakan K dan BK.

Model B — Genuine EWS

Hanya menggunakan informasi sampai:

minggu 2
atau minggu 3.

Misalnya:

Early_Performance_Composite
Early Attendance
Early TP Completion
Early Report Completion
Early TP Mean
Early Report Mean

Tujuannya:

mengetahui apakah risiko BK sudah dapat dideteksi sebelum semester berjalan jauh.

Nah, ini jauh lebih kuat.

Karena Anda tidak lagi sekadar berkata:

“ML dapat memprediksi kompetensi.”

Tetapi:

“Seberapa dini kompetensi dapat diprediksi tanpa menggunakan informasi masa depan?”

Itu pertanyaan penelitian yang jauh lebih menarik.

12. Early_Performance_Composite juga harus diaudit

Anda mengatakan:

SHAP = 0.482

dan menyimpulkan:

“kegagalan adaptasi di dua minggu pertama adalah root cause”

Ini terlalu kuat.

SHAP tidak membuktikan:

root cause.

SHAP menunjukkan kontribusi fitur terhadap prediksi model, bukan hubungan kausal.

Jadi:

❌

“Early Performance merupakan akar masalah kegagalan.”

Lebih tepat:

✅

“Early Performance Composite merupakan fitur yang paling dominan dalam kontribusi prediktif model.”

Kemudian:

“Temuan ini mengindikasikan bahwa performa pada fase awal semester memiliki keterkaitan prediktif yang kuat dengan status kompetensi akhir.”

Perbedaan kalimatnya kecil, tetapi secara ilmiah sangat besar.

13. Bahkan SHAP 0.482 sendiri perlu konteks

Jangan hanya mengatakan:

SHAP = 0.482.

Reviewer akan bertanya:

0.482 dibandingkan apa?

Anda sebaiknya menunjukkan:

mean absolute SHAP;
ranking fitur;
distribusi SHAP;
directionality;
beeswarm plot;
dependence plot untuk fitur utama.

Misalnya:

Early_Performance_Composite 0.482
Laporan_Mean 0.309
Performance_Std 0.187
Attendance 0.0
Completion 0.0

Bagus.

Tetapi kemudian tunjukkan:

apakah nilai Early_Performance tinggi meningkatkan probabilitas K atau BK?

Itu jauh lebih informatif.

14. Saya juga akan mengubah istilah “pemenang mutlak” dan “State of The Art”

Ini terlalu bombastis.

Dari dataset:

n = 89

Anda tidak bisa mengatakan penelitian Anda menghasilkan:

State of The Art

karena SOTA berarti klaim komparatif terhadap penelitian/benchmark yang lebih luas.

Lebih aman:

best-performing configuration in this study

atau:

konfigurasi dengan performa terbaik pada eksperimen penelitian ini.

Begitu juga:

“nilai A dalam Machine Learning”

sebaiknya dihilangkan.

ROC-AUC 0.895 memang bagus, tetapi tidak ada standar ilmiah:

ROC-AUC 0.895 = nilai A.

15. Tabel Anda sendiri perlu diperiksa lagi

Ada satu hal yang langsung menarik perhatian saya.

Anda menulis:

Skenario Model Fitur Recall BK Balanced Acc
S3 DT 6 100% 77.78%
S1 RF 3 88.89% 66.67%
S2 RF 6 77.78% 77.78%
S3 RF 6 77.78% 77.78%
S4 RF 15 77.78% 72.22%
S5 RF 20 77.78% 72.22%

Di sini sebaiknya jangan hanya memilih pemenang berdasarkan Recall.

Karena:

S3 DT

Recall BK = 100%

tetapi:

Balanced Accuracy = 77.78%

sementara beberapa model:

Recall = 77.78%

Balanced Accuracy = 77.78%

Artinya perlu ditunjukkan confusion matrix masing-masing.

Untuk EWS saya malah ingin melihat:

TP BK
FN BK
FP BK
TN BK

karena model yang Recall-nya 100% belum tentu paling praktis.

Misalnya:

Model A:
Recall BK = 100%
Precision BK = 50%

Model B:
Recall BK = 89%
Precision BK = 90%

Untuk intervensi akademik, Model B bisa jadi lebih berguna karena tidak terlalu banyak memberikan false alarm.

16. Tambahkan Precision dan F2 secara eksplisit

Karena Anda mendefinisikan BK sebagai positive class, tabel final sebaiknya minimal:

Metric Score
Recall BK 88.89%
Precision BK ...
F1 BK ...
F2 BK 0.851
Specificity ...
Balanced Accuracy 77.78%
ROC-AUC 0.895
PR-AUC ...

Kemudian confusion matrix.

Ini akan membuat klaim EWS jauh lebih solid.

17. PR-AUC justru sangat menarik untuk Anda

Karena fokus Anda adalah:

BK detection

sebenarnya PR-AUC sering lebih informatif daripada hanya ROC-AUC ketika positive class menjadi fokus.

Namun karena distribusi Anda sudah hampir 50:50, ROC-AUC tetap relevan.

Saya akan mempertahankan keduanya:

ROC-AUC → diskriminasi global

PR-AUC → kemampuan mendeteksi kelas yang menjadi fokus

18. Saya sangat menyarankan baseline yang sederhana

Jangan hanya:

Decision Tree vs Random Forest.

Tambahkan baseline:

Dummy Classifier

Misalnya:

most frequent

atau:

stratified

Supaya pembaca tahu apakah model benar-benar memberikan nilai tambah.

Kemudian:

Dummy
Logistic Regression
Decision Tree
Random Forest

Tidak perlu puluhan algoritma.

Justru sedikit model tetapi metodologinya kuat lebih bagus untuk penelitian Anda.

19. Satu hal lagi: “kompetensi” harus benar-benar didefinisikan dengan sangat kuat

Ini fundamental.

Saat ini:

K = ≥83
BK = <83

Tetapi reviewer akan bertanya:

Mengapa 83?

Dan lebih penting:

Apakah nilai akhir benar-benar merepresentasikan kompetensi?

Karena:

nilai ≠ kompetensi secara otomatis.

Anda perlu mendefinisikan:

Kompetensi operasional dalam penelitian ini direpresentasikan melalui...

Misalnya berdasarkan:

rubrik kompetensi;
threshold dosen;
skor akhir;
capaian minimum;
atau kriteria resmi mata kuliah.

Ini membuat target Anda legitimate.

20. Saya akan mengubah framing penelitian

Menurut saya penelitian Anda lebih kuat jika bukan dijual sebagai penelitian “mencari algoritma terbaik”.

Framing terbaik justru:

Early prediction of student competency using behavioral/academic activity features with explainable machine learning

Kemudian ML adalah alatnya.

Struktur kontribusinya menjadi:

Research Question 1

Apakah aktivitas akademik mahasiswa dapat digunakan untuk mendeteksi status kompetensi?

Research Question 2

Seberapa dini status tersebut dapat diprediksi?

Research Question 3

Fitur apa yang paling berkontribusi terhadap prediksi risiko BK?

Research Question 4

Apakah konfigurasi EWS berbasis aktivitas awal mampu mempertahankan performa tanpa synthetic oversampling?

Menurut saya ini jauh lebih kuat.

21. Bahkan saya melihat desain eksperimen yang lebih kuat dari versi Anda sekarang

Saya akan membentuk penelitian seperti ini:

                 DATA 89 MAHASISWA
                         │
                         ▼
              Definisi Kompetensi
                 Threshold = 83
                         │
                         ▼
             ┌────────────────────┐
             │   Feature Design   │
             └────────────────────┘
                  │            │
          Retrospective       Early
             Features        Features
                  │            │
                  ▼            ▼
                 S3          S3_E
                  │            │
                  └──────┬─────┘
                         ▼
               Decision Tree / RF
                         │
                         ▼
            Nested Stratified CV
                         │
                         ▼
                 Final Holdout
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Performance             Explainability
       Recall/F2/AUC           SHAP Analysis
              │                     │
              └──────────┬──────────┘
                         ▼
                EWS Interpretation
                         │
                         ▼
              Early-risk detection

Dan S3_E menjadi kontribusi utama, bukan sekadar model dengan skor tertinggi.

22. Hal yang menurut saya paling kuat dari penelitian Anda

Ini justru bukan angka:

ROC-AUC 0.895

atau:

Recall 88.89%.

Hal paling menarik adalah pertanyaan:

Apakah sinyal risiko ketidakkompetenan sudah terlihat pada 2–3 minggu pertama?

Kalau eksperimen Anda benar-benar membuktikan bahwa:

Week 2–3 features
↓
88.89% BK recall
↓
tanpa SMOTE
↓
tanpa informasi masa depan
↓
nested CV + untouched holdout

maka itu jauh lebih bernilai secara ilmiah daripada sekadar mendapatkan accuracy tinggi.

23. Tetapi saya akan menghapus beberapa frase dari laporan sekarang

Saya tidak menyarankan menggunakan:

“secara ajaib”

“distribusi Gaussian yang ideal”

“pemenang mutlak”

“State of The Art”

“Dewa”

“kalkulator mati”

“akar masalah”

“membuktikan secara matematis ... root cause”

“nilai A dalam Machine Learning”

“murni, sah”

Bahasa tersebut bagus untuk presentasi informal, tetapi untuk manuskrip ilmiah justru dapat membuat reviewer merasa overclaiming.

Gunakan:

menunjukkan

mengindikasikan

memiliki kontribusi prediktif tertinggi

konfigurasi terbaik dalam penelitian ini

berkorelasi dengan

berpotensi digunakan sebagai indikator dini

Itu jauh lebih aman.

24. Penilaian saya terhadap kondisi penelitian saat ini

Kalau saya memberikan penilaian kasar dari sisi kesiapan ilmiah:

Aspek Penilaian
Relevansi masalah 9/10
Ide EWS 9/10
Pemilihan metrik 8.5/10
Explainable AI 8.5/10
Penggunaan nested CV 8.5/10
Penanganan imbalance 8/10
Feature engineering 8/10
Validitas threshold 83 6/10
Validitas klaim kausal SHAP 5/10
Kekuatan statistik karena n=89 6/10
Validasi temporal EWS harus diperkuat
Gaya penulisan akademik 6/10

Jadi menurut saya, penelitiannya sendiri sudah menarik; manuskripnya yang masih perlu didisiplinkan secara ilmiah.

25. Kesimpulan saya

Saya tidak akan membuang arah penelitian ini. Justru saya akan mempertahankan inti besarnya:

Decision Tree + Early Performance + Explainable AI + Early Warning System + tanpa synthetic oversampling.

Tetapi saya akan melakukan empat revisi prioritas:

1. Justifikasi threshold 83 harus berasal dari domain/akademik, bukan karena kelas menjadi 45:44.

2. Pisahkan jelas model retrospektif dengan model genuine EWS. Model EWS hanya boleh menggunakan informasi yang tersedia sampai minggu prediksi.

3. Jangan menyebut SHAP sebagai bukti sebab-akibat. SHAP = kontribusi prediktif, bukan root cause.

4. Perkuat validasi karena hanya ada 89 mahasiswa. Minimal confusion matrix, precision/recall/F2, confidence interval, stratified nested CV yang benar, dan holdout yang benar-benar untouched.

Dan ada satu hal yang menurut saya sangat penting: sebelum menyebut S3_E sebagai model final, kita perlu audit formula setiap fitur—khususnya Early_Performance_Composite, Laporan_Mean, Performance_Std, completion rate, serta cara threshold 83 dibentuk—untuk memastikan tidak ada target leakage maupun temporal leakage. Itu bisa mengubah kesimpulan penelitian secara material.

Kalau empat hal itu beres, menurut saya penelitian Anda akan terlihat jauh lebih matang daripada sekadar eksperimen klasifikasi mahasiswa biasa.
