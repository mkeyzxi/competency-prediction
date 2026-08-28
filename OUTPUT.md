muhma@mkeyzxi MINGW64 /c/belajarku/Belajar ML/Logika-Algoritma (migrate/bdnr)
$ python src/build_dataset.py
Total Unique Students: 89
Class Distribution:
Competency_Label
0 45
1 44
Name: count, dtype: int64
Label Reason Distribution:
Label_Reason
Final_Score>=83 44
Final_Score<83 42
No_Final_Attendance 3
Name: count, dtype: int64
Exported processed datasets (V2).

muhma@mkeyzxi MINGW64 /c/belajarku/Belajar ML/Logika-Algoritma (migrate/bdnr)
$ python src/features.py
Features generated successfully (V2 - Incremental).

Berdasarkan hasil eksperimen terakhir yang kamu jalankan, skor paling tinggi untuk skenario S3 (khususnya varian S3_E) diraih oleh model DecisionTree.Berikut adalah rincian skor tertinggi pada tahap Cross-Validation (CV) dan evaluasi akhir:

1. Skor Tertinggi di Tahap Seleksi (CV)Model DecisionTree | S3_E menjadi kandidat terbaik dengan skor rata-rata:CV Recall BK: 0.7914 (79.1%)CV F2 BK: 0.7368CV Balanced Accuracy: 0.65002. Skor Tertinggi di Tahap Evaluasi Akhir (Final Holdout)Ketika model terbaik tersebut diuji pada data yang benar-benar baru (holdout set), performanya justru meningkat jauh lebih tinggi:Recall BK: 0.8889 (Skor deteksi kompetensi tertinggi)Balanced Accuracy: 0.7778F2 BK: 0.8511ROC-AUC: 0.8951
