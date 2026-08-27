1. Paling kritis: C1-S6

Di Dokumen 2 tercatat:

C1 + Random Forest + Feature Selection
CV Accuracy = 74,54 ± 8,10
CV Balanced Accuracy = 62,91 ± 11,01
Test Accuracy = 88%
Test Balanced Accuracy = 92,5%
TP=17, TN=5, FN=3, FP=0.

Dari confusion matrix tersebut, saya bisa menghitung:

Total test = 17 + 5 + 3 + 0 = 25

Accuracy:

$$ (17+5)/25 = 88\% $$

✅ 88% benar.

Tetapi Balanced Accuracy:

$$ \frac{Recall*{kelas1}+Recall*{kelas0}}{2} $$

Jika Belum Kompeten = 0 seperti yang didefinisikan dokumen:

$$ Recall*{kompeten}=17/(17+0)=100\% $$ $$ Recall*{belum\ kompeten}=5/(5+3)=62.5\% $$

maka:

$$ BA=(100+62.5)/2=\mathbf{81.25\%} $$

Bukan 92,5%.

Jadi ini prioritas audit nomor 1.

Ada kemungkinan confusion matrix-nya yang tercatat bukan confusion matrix yang menghasilkan 92,5%, atau angka Balanced Accuracy berasal dari prediksi lain. Kita tidak boleh menebak mana yang benar.

2. Audit seluruh C1-C_Full

Dokumen 2 mencatat:

Cutoff Model Test Acc Test Bal.Acc
C1 RF + FS 88% 92,5%
C2 DT 92% 87,5%
C2 RF + FS 84% 90%
C3 RF 88% 85%
C3 RF + FS 88% 85%
C4 RF + FS 84% 75%
C_Full RF + FS 88% 85%

Angka-angka ini ada di tabel hasil utama.

Saya ingin menghitung ulang setiap pasangan Accuracy dan Balanced Accuracy dari confusion matrix asli, bukan menerima tabel begitu saja.

Masalahnya, untuk C2-C_Full dokumen mengatakan “Tidak dirinci”, sehingga kita belum bisa memverifikasi Balanced Accuracy hanya dari PDF.

Jadi kita perlu mengambil hasil prediksi asli dari Python, bukan hanya laporan PDF.

3. Audit CV Mean ± SD

Dokumen menyebut:

C1 RF+FS: 74,54 ± 8,10 Accuracy; 62,91 ± 11,01 Balanced Accuracy
C2 DT: 75,13 ± 7,43; 65,45 ± 10,04
C2 RF+FS: 75,07 ± 7,80; 65,67 ± 12,41
C3 RF: 77,34 ± 7,03; 67,45 ± 11,84
C3 RF+FS: 75,28 ± 9,46
C4 RF+FS: 75,84 ± 8,31
C_Full RF+FS: 76,51 ± 11,04.

Yang perlu saya audit:

Apakah SD itu benar-benar berasal dari outer/repeated CV yang sama?

Karena kalau misalnya:

Accuracy berasal dari 25 fold,
Balanced Accuracy berasal dari split berbeda,
atau SD dihitung dengan ddof=0 sementara sebagian lain ddof=1,

maka angka tersebut tidak comparable.

Dokumen sendiri menyebut repeated/nested CV sebagai komponen penting penelitian.

Jadi saya ingin memastikan berapa fold, berapa repeat, dan dari fold mana setiap mean ± SD berasal.

4. Audit klaim “C1 adalah warning terbaik”

Ini penting secara ilmiah.

Dokumen menyatakan:

C1 sangat menarik karena 88% Accuracy dan 92,5% Balanced Accuracy.

Tetapi CV Balanced Accuracy C1 hanya:

62,91 ± 11,01%

sedangkan:

C3 RF = 67,45 ± 11,84%.

Jadi kita harus hati-hati.

Belum boleh langsung mengatakan:

“C1 adalah cutoff terbaik.”

Yang lebih tepat untuk sementara:

“C1 menunjukkan performa hold-out tertinggi pada konfigurasi yang diuji, tetapi kestabilan generalisasinya belum menunjukkan keunggulan yang jelas dibanding cutoff lainnya.”

Ini justru sangat menarik untuk artikel.

5. Audit klaim Accuracy 84–92% dan Balanced Accuracy 75–92,5%

Dokumen berulang kali menyatakan:

“hold-out Accuracy 84–92% dan Balanced Accuracy 75–92,5%.”

Secara aritmetika dari tabel memang demikian.

Tetapi setelah menemukan masalah C1, rentang tersebut harus diaudit ulang dari confusion matrix/prediction vector asli.

Ini penting karena angka 92,5% merupakan salah satu angka yang paling mungkin dikutip di abstrak.

Saya bahkan tidak akan memasukkan 92,5% ke abstrak sebelum audit selesai.

6. Audit confusion matrix C1 terhadap narasi recall

Dokumen mengatakan:

test memiliki 8 mahasiswa Belum Kompeten: TN=5 dan FN=3.

Ini menghasilkan:

$$ Recall\_{Belum Kompeten}=5/(5+3)=62.5\% $$

Tetapi pada tabel utama dokumen tidak menampilkan Recall.

Jadi nanti kita harus memastikan apakah yang dimaksud Recall Belum Kompeten memang 62,5%.

Ini penting karena di bagian lain penelitian menyatakan Recall Belum Kompeten adalah metrik kritis.

7. Audit F1 Macro

Ini malah belum tersedia lengkap pada tabel hasil utama Dokumen 2.

Padahal dokumen menyatakan F1 Macro sebagai salah satu metrik utama dan roadmap final menyebut:

Accuracy, Balanced Accuracy, F1 Macro, Recall Belum Kompeten dilaporkan.

Jadi saya ingin menghitung:

Precision
Recall
F1 per kelas
Macro F1

dari confusion matrix/prediction asli.

Untuk C1, misalnya, kita bisa menghitung F1 dari 17,5,3,0, lalu membandingkannya dengan angka yang nanti ada di script/output.

8. Audit denominator setiap metrik

Ini sering menjadi jebakan kecil tetapi mematikan saat reviewer membaca tabel.

Untuk setiap cutoff saya ingin memastikan:

jumlah test = jumlah TP + TN + FP + FN

dan:

Accuracy = (TP+TN)/N

Recall = TP/(TP+FN)

Specificity = TN/(TN+FP)

Balanced Accuracy = (Recall + Specificity)/2

Precision = TP/(TP+FP)

F1 = 2PR/(P+R)

Kalau satu saja tidak cocok, kita berhenti dan telusuri sumbernya.

9. Audit angka feature importance

Dokumen 2 menyebut pada iterasi sebelumnya:

TP_First2_Mean = 24,29%
Laporan_Max = 14,97%
Laporan_Mean = 6,10%
Respons_Std = 5,37%
Respons_Trend = 5,18%.

Ini juga perlu diaudit.

Sebab pada Dokumen 1 ranking dan angkanya berbeda:

TP_First2_Mean = 21,08%
Laporan_Max = 11,16%
Respons_Std = 7,64%
Absence_Count = 7,17%
Attendance_PreFinal_Rate = 6,63%.

Perbedaan ini belum tentu salah, karena model/iterasi bisa berbeda.

Tetapi dalam manuskrip final harus sangat jelas:

angka SHAP/importance berasal dari model mana, cutoff mana, feature set mana, dan split mana.

Jangan sampai reviewer bertanya:

“Why is TP_First2_Mean 24.29% in one section but 21.08% in another?”

dan kita baru menyadarinya setelah submit.

10. Audit ukuran sampel setiap tahap

Dokumen menggunakan:

P2 = 123 mahasiswa

dan test sekitar:

25 mahasiswa.

Ini harus konsisten dengan semua tabel.

Kalau test C1 memang 25:

17 + 5 + 3 + 0 = 25 ✅

Tapi saya ingin memastikan C2, C3, C4, C_Full juga menggunakan test population yang sama, bukan ada filtering/eligibility berbeda.

Karena penelitian menggunakan P0/P1/P2 dan C1-C4/C_Full.

11. Audit pemilihan model terhadap test set

Ini bukan sekadar audit angka, tetapi audit urutan eksperimen.

Dokumen mengatakan:

test tidak digunakan untuk feature selection, threshold selection, dan model selection.

Namun sekaligus:

C1 menghasilkan 92,5% sehingga disebut kandidat warning sangat menarik.

Saya ingin melihat kapan C1 dipilih.

Kalau semua cutoff dihitung ke test terlebih dahulu lalu C1 dipilih karena test paling tinggi, maka ada risiko:

test-set-driven model selection.

Bukan leakage dalam arti klasik seperti SMOTE sebelum split, tetapi tetap bisa menjadi selection bias / multiple comparison problem.

Ini sangat penting untuk SINTA 2.

12. Audit threshold

Dokumen menyebut threshold tuning sebagai bagian desain metodologi dan mengharuskan threshold tidak dipilih menggunakan test.

Saya ingin memastikan:

Apakah threshold C1/C2/... benar-benar 0,5?

atau:

Apakah threshold dioptimalkan?

Kalau dituning:

objective apa?
di inner CV?
berapa threshold?
apakah threshold berbeda antar cutoff?
apakah test prediction menggunakan threshold yang sudah locked?

Ini bisa mengubah seluruh confusion matrix.

Jadi daftar audit saya sebenarnya seperti ini
🔴 Prioritas sangat tinggi

1. C1-S6: confusion matrix ↔ Accuracy ↔ Balanced Accuracy

Karena saat ini 92,5% tidak cocok dengan TP=17, TN=5, FN=3, FP=0.

2. Semua confusion matrix C1-C_Full

Untuk memastikan seluruh Test Balanced Accuracy benar.

3. CV mean ± SD

Pastikan berasal dari fold/repeat yang benar-benar sama.

4. Recall Belum Kompeten

Karena ini metrik utama early-warning.

🟠 Prioritas tinggi

5. F1 Macro

Hitung ulang dari prediction asli.

6. Semua denominator N test

Pastikan konsisten 25 atau memang ada perubahan.

7. Threshold

Pastikan tidak disentuh test.

8. Model/cutoff selection

Pastikan C1 tidak dipilih karena melihat test.

🟡 Prioritas berikutnya

9. Feature importance / SHAP

Terutama perbedaan 24,29% vs 21,08%.

10. Semua angka dalam abstrak dan kesimpulan

Setelah angka utama dikunci, baru narasi final dibuat.

Dan saya ingin melakukan auditnya dari sumber paling bawah

Bukan dari PDF.

PDF hanya hasil laporan.

Yang paling ideal adalah kita audit langsung dari output/script eksperimen Python, karena dari sana saya bisa memeriksa:

y_test

y_pred

confusion_matrix

classification_report

balanced_accuracy_score

accuracy_score

f1_score

serta seluruh fold CV.

Dengan begitu kita bisa menjawab satu pertanyaan penting:

“Apakah 92,5% itu benar, atau hanya angka laporan yang salah?”

Saat ini, berdasarkan Dokumen 2 saja, saya belum akan menyatakan 92,5% benar. Yang sudah bisa saya pastikan adalah confusion matrix yang tertulis menghasilkan 81,25%, bukan 92,5%.

Dan ini sebenarnya kabar baik. Lebih baik kita menemukan baut longgar ini sekarang daripada reviewer yang menemukannya setelah submit. 🔍
