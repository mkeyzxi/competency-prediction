import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
import shutil

DOC_PATH = r"C:\belajarku\Belajar ML\Logika-Algoritma\Draft Laporan Litapdimas 2026 ver-2.0.docx"
REVISI_PATH = r"C:\belajarku\Belajar ML\Logika-Algoritma\Draft Laporan Litapdimas 2026 ver-2.0_revisi.docx"
BASE_DIR = r"C:\belajarku\Belajar ML\Logika-Algoritma"

def add_img(doc, img_rel, width_in=5.2):
    full_path = os.path.join(BASE_DIR, img_rel)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Missing image: {full_path}")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(2)
    pf.line_spacing = 1.0
    r = p.add_run()
    r.add_picture(full_path, width=Inches(width_in))
    return p

def add_cap(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.15
    pf.first_line_indent = None
    r = p.add_run(text)
    r.font.name = "Cambria"
    r.font.size = Pt(11)
    r.font.bold = False
    return p

def add_txt(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.first_line_indent = 540385  # ~0.59 in / 1.5 cm
    pf.space_before = Pt(0)
    pf.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = "Cambria"
    r.font.size = Pt(12)
    return p

def add_heading(doc, text, level=2, center=False):
    style_name = 'Normal' if level == 1 else f'Heading {level}'
    p = doc.add_paragraph(style=style_name)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.15 if center else 1.5
    r = p.add_run(text)
    r.font.name = "Cambria"
    r.font.size = Pt(12)
    r.font.bold = True
    return p

def rebuild_document():
    doc = docx.Document(r"C:\belajarku\Belajar ML\Logika-Algoritma\Draft Laporan Litapdimas 2026 ver-2.0.docx.bak")
    body = doc.element.body

    # 1. Update BAB I headings
    doc.paragraphs[22].text = "A. Latar Belakang"
    doc.paragraphs[40].text = "C. Rumusan Masalah"
    doc.paragraphs[45].text = "D. Tujuan Penelitian"
    doc.paragraphs[50].text = "E. Manfaat Penelitian"

    # 2. Add captions to existing images in BAB II and BAB III
    for i, child in enumerate(list(body)):
        if child.tag.endswith('p') and 'w:drawing' in child.xml:
            idx = body.index(child)
            prev_p = docx.text.paragraph.Paragraph(body[idx-1], doc).text
            if "proses prediksi dini dalam penelitian ini dapat digambarkan" in prev_p:
                cap = add_cap(doc, "Gambar 2.1. Alur Konseptual Prediksi Dini Kompetensi Praktikum Basis Data Non-Relasional.")
                child.addnext(cap._p)
                print("Added caption for Gambar 2.1")
            elif "alur desain penelitian dapat dirumuskan sebagai" in prev_p:
                cap = add_cap(doc, "Gambar 3.1. Alur Desain Penelitian Evaluasi Temporal dan Komparasi Model.")
                child.addnext(cap._p)
                print("Added caption for Gambar 3.1")
            elif "struktur pembagian data dapat digambarkan" in prev_p:
                cap = add_cap(doc, "Gambar 3.2. Struktur Pembagian Data Modeling (80%) dan Locked Holdout (20%) serta Nested Stratified CV.")
                child.addnext(cap._p)
                print("Added caption for Gambar 3.2")

    # 3. Locate BAB IV and BAB V
    bab4_start = None
    bab5_start = None
    for i, child in enumerate(body):
        if child.tag.endswith('p'):
            p = docx.text.paragraph.Paragraph(child, doc)
            if 'BAB IV' in p.text and bab4_start is None:
                bab4_start = child
            elif 'BAB V' in p.text and bab4_start is not None and bab5_start is None:
                bab5_start = child

    # Find the 5 tables in BAB IV
    tables = []
    in_b4 = False
    for child in list(body):
        if child == bab4_start:
            in_b4 = True
        elif child == bab5_start:
            in_b4 = False
        if in_b4 and child.tag.endswith('tbl'):
            tables.append(child)

    print(f"Found {len(tables)} tables in BAB IV")
    tbl1, tbl2, tbl3, tbl4, tbl5 = tables

    # Remove all non-table elements between bab4_start and bab5_start
    in_b4 = False
    to_remove = []
    for child in list(body):
        if child == bab4_start:
            in_b4 = True
        elif child == bab5_start:
            in_b4 = False
        if in_b4:
            if child not in tables:
                to_remove.append(child)

    print(f"Removing {len(to_remove)} old paragraphs from BAB IV...")
    for child in to_remove:
        body.remove(child)

    # INTERVAL 0: Before Table 4.1
    int0 = [
        add_heading(doc, "BAB IV", level=1, center=True),
        add_heading(doc, "HASIL DAN PEMBAHASAN", level=1, center=True),
        add_heading(doc, "A. Hasil Penelitian", level=2),
        add_heading(doc, "1. Karakteristik Data dan Rancangan Evaluasi", level=2),
        add_txt(doc, "Analisis menggunakan data dari 89 mahasiswa pada mata kuliah Praktikum Basis Data Non-Relasional, yang terdiri atas 45 mahasiswa berstatus Belum Kompeten (BK) dan 44 mahasiswa berstatus Kompeten. Evaluasi akhir dilakukan pada data uji terkunci (locked holdout set) berukuran 20% yang berisi 18 mahasiswa, dengan komposisi seimbang yaitu masing-masing 9 mahasiswa BK dan 9 mahasiswa Kompeten. Proses validasi internal menggunakan nested stratified cross-validation dengan 2 outer folds dan 5 inner folds guna mencegah terjadinya bias optimistik dan kebocoran informasi (data leakage). Untuk mengevaluasi potensi sistem peringatan dini (Early Warning System/EWS), kinerja model dievaluasi secara bertahap pada empat cutoff temporal: Minggu ke-1 (W1), Minggu ke-2 (W2), Minggu ke-3 (W3), serta Full Semester sebagai pembanding retrospektif."),
        add_heading(doc, "2. Hasil Evaluasi Temporal Sistem Peringatan Dini", level=2),
        add_txt(doc, "Hasil evaluasi pada locked holdout menunjukkan adanya pergeseran pola kinerja yang sangat substantif ketika informasi akademik dibatasi secara temporal. Evaluasi multi-cutoff memperlihatkan bagaimana trade-off antara kecepatan prediksi dan akurasi klasifikasi berkembang seiring bertambahnya minggu perkuliahan. Pada fase awal (W1), model mampu mendeteksi sebagian besar mahasiswa BK namun belum memiliki kemampuan diskriminasi terhadap mahasiswa Kompeten. Memasuki W2 dan W3, kemampuan deteksi risiko mencapai titik optimal. Rekapitulasi metrik performa EWS pada locked holdout disajikan pada Tabel 4.1."),
        add_cap(doc, "Tabel 4.1. Rekapitulasi metrik evaluasi EWS pada locked holdout.")
    ]
    for p in int0:
        tbl1.addprevious(p._p)

    # INTERVAL 1: Between Table 4.1 and Table 4.2
    int1 = [
        # Paragraf Penjelasan Tabel 4.1
        add_txt(doc, "Berdasarkan rekapitulasi metrik pada Tabel 4.1, terlihat perbedaan performa yang sangat nyata antarfase pengamatan. Pada fase awal (W1), sistem mencatatkan Recall BK sebesar 88,89%, namun nilai Specificity berada pada 0,00% dengan Balanced Accuracy 44,44%. Hal ini menunjukkan bahwa meskipun model mampu menangkap mayoritas mahasiswa berisiko sejak minggu pertama, model sama sekali belum mampu mengenali mahasiswa Kompeten sehingga seluruh kelas aman ikut tertandai. Perubahan signifikan terjadi pada cutoff Minggu ke-2 (W2), di mana Recall BK mencapai tingkat sempurna sebesar 100,00%, Specificity melonjak menjadi 55,56%, dan Balanced Accuracy meningkat tajam menjadi 77,78%. Angka-angka pada W2 ini identik dengan kinerja pada Minggu ke-3 (W3). Pada skenario Full Semester, Balanced Accuracy tetap berada pada 77,78%, di mana Specificity memang meningkat menjadi 66,67%, namun Recall BK justru mengalami penurunan menjadi 88,89%. Pola kuantitatif ini mengindikasikan bahwa titik W2 merupakan fase di mana akumulasi informasi awal telah memadai untuk mendeteksi seluruh mahasiswa Belum Kompeten tanpa adanya risiko kegagalan yang terlewat."),
        
        # Pengantar Gambar 4.1 -> Gambar 4.1 -> Caption -> Penjelasan Gambar 4.1
        add_txt(doc, "Untuk memperjelas lintasan dan dinamika perubahan metrik utama tersebut secara visual, Gambar 4.1 menyajikan grafik evolusi metrik Recall BK, Specificity, dan Balanced Accuracy pada setiap cutoff waktu."),
        add_img(doc, r"extracted_pdf_figures\page_4_img_1.png", width_in=5.2),
        add_cap(doc, "Gambar 4.1. Evolusi Recall BK, Specificity, dan Balanced Accuracy menurut Cutoff Waktu."),
        add_txt(doc, "Visualisasi pada Gambar 4.1 memperlihatkan karakteristik kurva non-monotonik yang sangat informatif bagi perancangan EWS. Kurva Recall BK mengalami lonjakan dari 88,89% pada W1 hingga mencapai puncaknya di 100% pada W2 dan W3, sebelum akhirnya terkoreksi ke 88,89% di akhir semester. Sementara itu, kurva Specificity merangkak naik secara bertahap dari 0% pada W1, ke 55,56% pada W2–W3, dan mencapai 66,67% pada Full Semester. Kurva Balanced Accuracy mencapai dataran tinggi (plateau) stabil sebesar 77,78% sejak W2 hingga akhir semester. Bentuk grafik ini menegaskan bahwa penambahan data praktikum setelah Minggu ke-2 tidak lagi memberikan peningkatan terhadap daya deteksi risiko, melainkan hanya menyaring mahasiswa kompeten dengan lebih presisi. Oleh karena itu, W2 menjadi kandidat cutoff paling efisien karena memaksimalkan jendela waktu intervensi bagi dosen sebelum perkuliahan berjalan terlalu jauh."),
        
        # Pengantar Gambar 4.2 -> Gambar 4.2 -> Caption -> Penjelasan Gambar 4.2
        add_txt(doc, "Selain ketiga metrik utama di atas, evaluasi sistem peringatan dini yang berorientasi pada mitigasi risiko memerlukan peninjauan terhadap metrik Precision, F2-score, dan koefisien korelasi Matthews (MCC) guna memastikan bahwa peringatan yang dihasilkan tidak bersifat acak dan tetap sensitif terhadap risiko."),
        add_img(doc, r"extracted_pdf_figures\page_5_img_2.png", width_in=5.2),
        add_cap(doc, "Gambar 4.2. Perbandingan Kinerja Lanjutan: Precision BK, F2-Score, dan Matthews Correlation Coefficient (MCC) pada Setiap Cutoff."),
        add_txt(doc, "Grafik pada Gambar 4.2 mengonfirmasi keandalan performa model melalui metrik pendukung. Skor F2, yang memberikan bobot dua kali lebih besar pada Recall dibanding Precision, melonjak dari 75,47% pada W1 menjadi 91,84% pada W2 dan W3, sebelum turun menjadi 85,11% pada Full Semester. Nilai F2 yang melampaui 91% pada W2 menegaskan keberhasilan model dalam mengutamakan perlindungan terhadap mahasiswa rentan (zero false negative). Pada saat yang sama, metrik Precision BK meningkat secara konsisten dari 47,06% pada W1 menjadi 69,23% pada W2–W3, dan 72,73% pada Full Semester. Peningkatan nilai MCC dari 0,00 pada W1 menjadi 0,57 pada W2–W3 dan 0,58 pada Full Semester menunjukkan adanya derajat asosiasi prediksi yang kuat dan stabil antara kelas prediksi model dengan label riil mahasiswa pada data uji terkunci."),
        
        # Heading A.3 dan Pengantar Tabel 4.2
        add_heading(doc, "3. Confusion Matrix pada Setiap Cutoff", level=2),
        add_txt(doc, "Distribusi klasifikasi pada setiap cutoff waktu memperlihatkan detail pergeseran antara prediksi benar dan salah. Untuk memahami secara konkret bagaimana mahasiswa diklasifikasikan ke dalam kategori True Positive, False Negative, False Positive, dan True Negative, hasil prediksi pada locked holdout (n=18) dipetakan ke dalam matriks kontingensi pada setiap cutoff temporal sebagaimana dirangkum pada Tabel 4.2."),
        add_cap(doc, "Tabel 4.2. Ringkasan confusion matrix pada setiap cutoff.")
    ]
    for p in int1:
        tbl2.addprevious(p._p)

    # INTERVAL 2: Between Table 4.2 and Table 4.3
    int2 = [
        # Paragraf Penjelasan Tabel 4.2
        add_txt(doc, "Berdasarkan data matriks kontingensi pada Tabel 4.2, pergeseran angka prediksi menggambarkan dinamika penyaringan mahasiswa di setiap tahapan waktu. Pada cutoff W1, model berhasil mendeteksi 8 mahasiswa BK, namun 1 mahasiswa BK terlewat (False Negative = 1), sementara seluruh 9 mahasiswa Kompeten salah diprediksi sebagai BK (False Positive = 9). Kondisi ini berubah secara substantif pada W2 dan W3, di mana seluruh 9 mahasiswa BK berhasil diklasifikasikan dengan tepat (True Positive = 9) tanpa ada satu pun mahasiswa berisiko yang terlewat (False Negative = 0). Di sisi lain, jumlah mahasiswa Kompeten yang keliru menerima peringatan berhasil ditekan dari 9 menjadi 4 mahasiswa (False Positive = 4), dengan 5 mahasiswa Kompeten berhasil dipilah secara benar (True Negative = 5). Pada skenario Full Semester, meskipun jumlah False Positive berkurang menjadi 3 orang, muncul kembali 1 kasus False Negative di mana mahasiswa BK terprediksi sebagai Kompeten. Perbandingan sel-sel matriks ini menunjukkan bahwa konfigurasi W2 dan W3 memberikan jaminan keamanan deteksi tertinggi bagi sistem peringatan dini."),
        
        # Pengantar Gambar 4.3 -> Gambar 4.3 -> Caption -> Penjelasan Gambar 4.3
        add_txt(doc, "Struktur klasifikasi model final terbaik (S3_E Decision Tree) pada locked holdout (n=18) di titik cutoff optimal tersebut divisualisasikan secara grafis melalui confusion matrix heatmap pada Gambar 4.3."),
        add_img(doc, r"results\final\cm_FINAL_S3_E_DecisionTree.png", width_in=4.2),
        add_cap(doc, "Gambar 4.3. Visualisasi Confusion Matrix Model Final EWS (S3_E Decision Tree) pada Locked Holdout (n=18)."),
        add_txt(doc, "Visualisasi confusion matrix pada Gambar 4.3 memperlihatkan kontras yang tegas antara kuadran prediksi benar dan salah pada model final. Sel True Positive (kiri-atas) terisi penuh dengan 9 mahasiswa (100% sensitivitas), dan sel False Negative (kanan-atas) bernilai 0 mutlak. Keberadaan 4 mahasiswa pada sel False Positive (kiri-bawah) dan 5 mahasiswa pada True Negative (kanan-bawah) mencerminkan toleransi sistem yang wajar. Dalam arsitektur EWS akademik, ketiadaan False Negative adalah parameter keberhasilan terpenting, karena sistem bertugas sebagai jaring pengaman agar tidak ada mahasiswa yang tertinggal, sementara keberadaan 4 false alarm dinilai masih berada dalam batas kapasitas pendampingan dosen di laboratorium."),
        
        # Heading A.4 dan Pengantar Tabel 4.3
        add_heading(doc, "4. Jumlah Mahasiswa yang Mendapat Peringatan dan Efisiensi Intervensi", level=2),
        add_txt(doc, "Penerapan EWS di lingkungan akademik menuntut efisiensi beban kerja dosen dan asisten praktikum. Oleh karena itu, jumlah mahasiswa yang ditandai menerima alarm peringatan harus dianalisis berdampingan dengan jumlah mahasiswa berisiko yang terlewat pada setiap cutoff pengamatan. Ringkasan perbandingan tersebut disajikan pada Tabel 4.3."),
        add_cap(doc, "Tabel 4.3. Jumlah peringatan dan jenis kesalahan klasifikasi.")
    ]
    for p in int2:
        tbl3.addprevious(p._p)

    # INTERVAL 3: Between Table 4.3 and Table 4.4
    int3 = [
        # Paragraf Penjelasan Tabel 4.3
        add_txt(doc, "Tabel 4.3 menunjukkan evaluasi beban intervensi operasional yang dihadapi oleh tim pengajar. Pada cutoff W1, sistem memberikan sinyal peringatan kepada 17 dari 18 mahasiswa (94,44% dari total sampel uji). Secara praktis, beban tindak lanjut yang mencakup hampir seluruh kelas ini sangat tidak efisien bagi dosen, terlebih masih terdapat 1 kasus mahasiswa berisiko yang terlewat. Memasuki cutoff W2 dan W3, volume peringatan berkurang secara signifikan menjadi 13 mahasiswa, sementara kasus risiko yang terlewat (False Negative) berhasil dieliminasi sepenuhnya menjadi 0. Pada skenario Full Semester, jumlah mahasiswa yang diperingatkan memang turun lebih lanjut menjadi 11 orang, namun penurunan ini harus dibayar mahal dengan kembalinya 1 kasus False Negative. Angka-angka ini membuktikan bahwa penundaan prediksi hingga akhir semester tidak sebanding dengan risiko membiarkan mahasiswa bermasalah tanpa intervensi."),
        
        # Pengantar Gambar 4.4 -> Gambar 4.4 -> Caption -> Penjelasan Gambar 4.4
        add_txt(doc, "Keseimbangan antara volume peringatan dan tingkat keparahan risiko kegagalan divisualisasikan secara komparatif melalui diagram batang pada Gambar 4.4."),
        add_img(doc, r"extracted_pdf_figures\page_5_img_1.png", width_in=5.2),
        add_cap(doc, "Gambar 4.4. Jumlah Mahasiswa Penerima Peringatan (Alarm) dan Kasus Risiko Terlewat (False Negative) pada Setiap Cutoff."),
        add_txt(doc, "Diagram batang pada Gambar 4.4 memperjelas trade-off operasional antara beban bimbingan pengajar (batang biru) dan tingkat risiko mahasiswa terlewat (batang oranye). Penurunan tajam jumlah alarm dari 17 pada W1 menjadi 13 pada W2 mencerminkan peningkatan efisiensi penyaringan sebesar 23,5%, sekaligus menghilangkan batang oranye (FN = 0). Munculnya kembali batang oranye pada Full Semester menegaskan bahaya mengandalkan model akhir semester yang terlalu konservatif. Dengan demikian, visualisasi ini memperkuat simpulan bahwa W2 merupakan titik kompromi paling rasional antara daya tampung intervensi dosen dan keselamatan akademik mahasiswa."),
        
        # Heading A.5 dan Pengantar Tabel 4.4
        add_heading(doc, "5. Hasil Evaluasi Baseline Retrospektif", level=2),
        add_txt(doc, "Sebelum menerapkan sistem deteksi bertahap, penelitian menguji performa model retrospektif menggunakan seluruh data semester penuh pada lima skenario fitur (S1 hingga S5) untuk Decision Tree dan Random Forest sebagai tolok ukur (benchmark). Ringkasan hasil evaluasi disajikan pada Tabel 4.4."),
        add_cap(doc, "Tabel 4.4. Ringkasan hasil evaluasi baseline retrospektif.")
    ]
    for p in int3:
        tbl4.addprevious(p._p)

    # INTERVAL 4: Between Table 4.4 and Table 4.5
    int4 = [
        # Paragraf Penjelasan Tabel 4.4
        add_txt(doc, "Berdasarkan rekapitulasi pada Tabel 4.4, konfigurasi S3 (6 fitur: Mean + Completion + Variability) menghasilkan kinerja terbaik pada data uji terkunci. Model Decision Tree S3 berhasil meraih Test Balanced Accuracy sebesar 77,78% dan Test Recall BK 100,00%, mengungguli Skenario S1 (Test BA 66,67%) dan Skenario S2 (Test BA 72,22%). Pada algoritma Random Forest, Skenario S3 juga mencatatkan Test BA 77,78% dan membukukan nilai PR-AUC tertinggi di antara seluruh model sebesar 0,892. Namun demikian, penambahan fitur lebih lanjut pada Skenario S4 (15 fitur) dan Skenario S5 (20 fitur) tidak mampu meningkatkan performa holdout, di mana nilai Test BA mengalami penurunan atau stagnasi. Temuan tabulasi ini menunjukkan bahwa ruang fitur yang memadukan tiga dimensi perilaku belajar (performa rata-rata, ketuntasan tugas, dan konsistensi) telah optimal untuk data berskala kecil (n=89)."),
        
        # Pengantar Gambar 4.5 -> Gambar 4.5 -> Caption -> Penjelasan Gambar 4.5
        add_txt(doc, "Perbandingan stabilitas antara hasil validasi silang bersarang (Nested CV) dan pengujian pada data uji terkunci (Locked Holdout) untuk kelima skenario retrospektif tersebut divisualisasikan pada Gambar 4.5."),
        add_img(doc, r"extracted_pdf_figures\page_3_img_1.png", width_in=5.2),
        add_cap(doc, "Gambar 4.5. Perbandingan Balanced Accuracy antara Validasi Silang (Nested CV) dan Locked Holdout pada Lima Konfigurasi Retrospektif."),
        add_txt(doc, "Grafik pada Gambar 4.5 memperlihatkan perbandingan performa antara estimasi CV (batang biru) dan performa locked holdout (batang oranye). Terlihat jelas bahwa performa holdout mencapai puncaknya pada konfigurasi S3 baik untuk Decision Tree maupun Random Forest. Kesenjangan antara CV dan holdout pada S1–S3 mencerminkan variabilitas estimasi yang wajar pada sampel kecil, namun kestabilan keunggulan S3 di atas skenario lain mengonfirmasi kekokohan seleksi fitur berbasis perilaku. Gambar ini juga membuktikan secara grafis bahwa memperbanyak atribut (S4 dan S5) justru menurunkan efektivitas model akibat meningkatnya kompleksitas hipotesis yang tidak didukung oleh volume sampel yang cukup."),
        
        # Pengantar Gambar 4.6 -> Gambar 4.6 -> Caption -> Penjelasan Gambar 4.6
        add_txt(doc, "Sebagai pembanding langsung terhadap confusion matrix model EWS W2, Gambar 4.6 menyajikan confusion matrix dari model baseline retrospektif S3 Decision Tree yang menggunakan data satu semester penuh."),
        add_img(doc, r"results\confusion_matrix\cm_Baseline_S3_DecisionTree.png", width_in=4.2),
        add_cap(doc, "Gambar 4.6. Confusion Matrix Baseline Retrospektif S3 Decision Tree."),
        add_txt(doc, "Analisis terhadap Gambar 4.6 memperlihatkan hasil yang sangat menarik: distribusi matriks pada model retrospektif satu semester penuh persis sama dengan model temporal W2, yaitu TP=9, FN=0, FP=4, dan TN=5. Kesamaan identik ini memberikan bukti empiris yang kokoh bahwa kemampuan klasifikasi model pada Minggu ke-2 telah menyamai kemampuan diskriminasi data satu semester penuh. Fakta ini menggugurkan keraguan bahwa prediksi dini akan mengorbankan akurasi secara drastis, sekaligus menjadi landasan ilmiah terkuat untuk mengimplementasikan EWS pada awal perkuliahan."),
        
        # Heading A.6 dan Pengantar Tabel 4.5
        add_heading(doc, "6. Ketidakpastian Estimasi Kinerja", level=2),
        add_txt(doc, "Dokumen penelitian melaporkan analisis ketidakpastian non-parametrik menggunakan metode bootstrap (B=1000 iterasi) untuk mengukur rentang interval kepercayaan 95% dari estimasi metrik kinerja Full Semester pada locked holdout berukuran kecil (n=18). Hasil estimasi dan interval ketidakpastian tersebut dirangkum pada Tabel 4.5."),
        add_cap(doc, "Tabel 4.5. Estimasi kinerja Full Semester dan interval ketidakpastian bootstrap (B=1000).")
    ]
    for p in int4:
        tbl5.addprevious(p._p)

    # INTERVAL 5: After Table 4.5 until BAB V
    int5 = [
        # Paragraf Penjelasan Tabel 4.5
        add_txt(doc, "Data pada Tabel 4.5 memperlihatkan estimasi titik beserta interval kepercayaan bootstrap 95% untuk evaluasi model. Nilai Recall BK sebesar 88,89% memiliki rentang interval antara 62,50% hingga 100,00%; Balanced Accuracy sebesar 77,78% memiliki rentang antara 56,15% hingga 95,45%; dan F2 BK sebesar 85,11% memiliki rentang antara 62,46% hingga 98,04%. Rentang interval yang cukup lebar ini merupakan representasi jujur dari variabilitas sampling pada dataset berukuran kecil. Hal terpenting dari temuan ini adalah batas bawah interval kepercayaan Balanced Accuracy (56,15%) tetap berada secara konsisten di atas baseline tebakan acak (dummy classifier sebesar 50,00%). Ini membuktikan bahwa kinerja model tidak terjadi secara kebetulan semata, melainkan memiliki validitas prediktif yang nyata."),
        
        # Pengantar Gambar 4.7 -> Gambar 4.7 -> Caption -> Penjelasan Gambar 4.7
        add_txt(doc, "Untuk menelaah relevansi metrik evaluasi pada kondisi data kecil dan membuktikan mengapa Accuracy konvensional tidak memadai untuk sistem peringatan dini, Gambar 4.7 menyajikan perbandingan antara Accuracy standar dan Balanced Accuracy di setiap cutoff temporal."),
        add_img(doc, r"extracted_pdf_figures\page_6_img_1.png", width_in=5.2),
        add_cap(doc, "Gambar 4.7. Perbandingan Accuracy dan Balanced Accuracy pada Locked Holdout untuk Setiap Cutoff Temporal."),
        add_txt(doc, "Visualisasi Gambar 4.7 mempertegas pentingnya menggunakan Balanced Accuracy sebagai indikator evaluasi utama. Pada cutoff W1, kurva Accuracy dan Balanced Accuracy sama-sama berada di level 44,44% karena kegagalan model mengenali mahasiswa kompeten. Namun mulai W2 hingga Full Semester, kedua garis metrik berhimpit secara sempurna pada angka 77,78%. Dalam konteks ini, Balanced Accuracy memberikan penilaian yang objektif karena memperhitungkan rata-rata sensitivitas dari kedua kelas secara setara. Grafik ini menunjukkan bahwa peningkatan jumlah data dari W2 ke Full Semester tidak lagi meningkatkan skor Balanced Accuracy, sehingga mempertegas posisi W2 sebagai cutoff evaluasi paling optimal."),
        
        # Heading A.7 dan SHAP Explainability
        add_heading(doc, "7. Temuan Interpretabilitas Model (Explainable AI / SHAP)", level=2),
        add_txt(doc, "Penerapan Explainable AI menggunakan SHAP (SHapley Additive exPlanations) memberikan pemahaman mendalam mengenai mekanisme internal model klasifikasi dalam menetapkan keputusan prediksi. Analisis ini mengungkap kontribusi masing-masing fitur prediktor terhadap probabilitas mahasiswa menjadi Belum Kompeten atau Kompeten, sekaligus melacak bagaimana representasi fitur bergeser ketika sistem beralih dari model retrospektif ke model temporal."),
        
        # Pengantar Gambar 4.8 -> Gambar 4.8 -> Caption -> Penjelasan Gambar 4.8
        add_txt(doc, "Kontribusi relatif setiap fitur prediktor dalam model final EWS S3_E Decision Tree dianalisis melalui tingkat kepentingan fitur global (global feature importance) sebagaimana divisualisasikan pada Gambar 4.8."),
        add_img(doc, r"results\shap\global_importance_FINAL_S3_E_DecisionTree.png", width_in=4.8),
        add_cap(doc, "Gambar 4.8. Tingkat Kepentingan Fitur Global (Global Feature Importance) pada Model Final EWS S3_E Decision Tree."),
        add_txt(doc, "Gambar 4.8 mengilustrasikan bahwa fitur Early_Performance_Composite mendominasi struktur keputusan model final secara mutlak dengan bobot kepentingan melebihi 60%, disusul oleh fitur ketuntasan tugas awal dan rasio kehadiran. Tingginya bobot fitur komposit ini membuktikan bahwa kualitas performa pada dua penugasan praktikum awal membawa informasi paling bernilai dalam memisahkan mahasiswa kompeten dari mahasiswa yang berisiko belum kompeten."),
        
        # Pengantar Gambar 4.9 -> Gambar 4.9 -> Caption -> Penjelasan Gambar 4.9
        add_txt(doc, "Untuk memahami bagaimana arah nilai fitur memengaruhi prediksi status kompetensi pada model retrospektif yang menggunakan data satu semester penuh, Gambar 4.9 menampilkan distribusi nilai SHAP (beeswarm plot) pada baseline S3."),
        add_img(doc, r"results\shap\beeswarm_Baseline_S3_DecisionTree.png", width_in=4.8),
        add_cap(doc, "Gambar 4.9. Distribusi Nilai SHAP (Beeswarm Plot) pada Model Baseline Retrospektif S3."),
        add_txt(doc, "Pada Gambar 4.9, sebaran titik nilai SHAP menunjukkan bahwa fitur akumulatif Laporan_Mean menjadi pembeda utama. Titik-titik biru (nilai fitur rendah) terkonsentrasi di sebelah kiri garis nol dengan nilai SHAP negatif yang besar, menandakan bahwa nilai laporan yang buruk mendorong model secara kuat ke arah klasifikasi Belum Kompeten. Sebaliknya, titik merah (nilai tinggi) mendorong ke arah Kompeten. Meskipun hubungan ini sangat logis, ketergantungan pada nilai laporan kumulatif membuat model ini bersifat reaktif dan tidak dapat difungsikan sebagai sistem peringatan dini di awal semester."),
        
        # Pengantar Gambar 4.10 -> Gambar 4.10 -> Caption -> Penjelasan Gambar 4.10
        add_txt(doc, "Perubahan mendasar terjadi ketika model diadaptasi ke dalam paradigma sistem peringatan dini dengan pembatasan cutoff temporal, sebagaimana diperlihatkan pada beeswarm plot model final EWS S3_E pada Gambar 4.10."),
        add_img(doc, r"results\shap\beeswarm_FINAL_S3_E_DecisionTree.png", width_in=4.8),
        add_cap(doc, "Gambar 4.10. Distribusi Nilai SHAP (Beeswarm Plot) pada Model Final EWS S3_E."),
        add_txt(doc, "Distribusi SHAP pada Gambar 4.10 memperlihatkan pergeseran arsitektur prediktif yang radikal: fokus utama model berpindah seutuhnya ke fitur Early_Performance_Composite. Mahasiswa dengan capaian tugas awal yang rendah (titik biru) terdorong kuat ke area prediksi risiko Belum Kompeten. Hal ini membuktikan bahwa algoritma pembelajaran mesin mampu menangkap sinyal kerentanan belajar mahasiswa sejak pertemuan awal, sehingga memberikan dasar empiris bagi pengajar untuk memulai intervensi tanpa perlu menunggu akumulasi nilai akhir."),

        # BAGIAN B: PEMBAHASAN HASIL PENELITIAN
        add_heading(doc, "B. Pembahasan Hasil Penelitian", level=2),
        add_heading(doc, "1. Evaluasi Temporal bagi Sistem Peringatan Dini", level=2),
        add_txt(doc, "Hasil penelitian ini menegaskan bahwa nilai ilmiah utama sistem peringatan dini (EWS) tidak terletak pada perburuan satu angka metrik tertinggi di akhir semester, melainkan pada pemahaman terhadap perubahan karakter prediksi ketika jendela informasi akademik dibatasi secara temporal. Sebagaimana diperlihatkan pada Gambar 4.1 dan Gambar 4.2, evaluasi retrospektif memang membuktikan bahwa data akhir semester mampu membedakan mahasiswa Kompeten dan Belum Kompeten dengan baik. Namun, dalam konteks operasional pendidikan tinggi, model retrospektif memiliki kelemahan mendasar: ia baru menghasilkan kesimpulan ketika waktu intervensi pengajar sudah habis. Sebaliknya, EWS sejati harus bersedia mengorbankan sebagian kelengkapan data demi kecepatan keputusan, sehingga menciptakan jendela intervensi yang cukup panjang bagi dosen untuk memberikan bimbingan."),
        add_txt(doc, "Kondisi ekstrem pada cutoff W1 menunjukkan bahwa mengandalkan satu pertemuan saja tidak memadai karena tingginya laju false alarm (Specificity 0%), sehingga beban intervensi mencakup hampir seluruh kelas. Namun, ketersediaan data hingga Minggu ke-2 (W2) membawa perubahan dramatis: Recall BK mencapai 100% dan Balanced Accuracy mencapai 77,78%. Yang paling krusial, penambahan data hingga Minggu ke-3 (W3) tidak memberikan tambahan nilai prediktif apa pun. Pola ini membuktikan bahwa sinyal risiko kompetensi mahasiswa praktikum Logika-Algoritma dan Basis Data Non-Relasional telah terkonsentrasi secara padat pada dua minggu pertama, sehingga menempatkan W2 sebagai titik intervensi yang paling rasional dan strategis."),

        add_heading(doc, "2. Trade-off antara False Alarm dan Risiko yang Terlewat", level=2),
        add_txt(doc, "Salah satu kontribusi paling aplikatif dari penelitian ini adalah pengungkapan trade-off asimetris antara false alarm (False Positive) dan risiko kegagalan yang terlewat (False Negative), sebagaimana divisualisasikan pada Gambar 4.3 dan Gambar 4.4. Dalam konteks sistem peringatan dini akademik, biaya (cost) dari kedua jenis kesalahan tersebut sangat tidak setara. Terjadinya False Negative merupakan kegagalan sistem yang paling fatal, karena mahasiswa yang sesungguhnya membutuhkan pendampingan justru luput dari pemantauan hingga akhirnya benar-benar gagal. Sebaliknya, False Positive (mahasiswa kompeten yang keliru ditandai) hanya menghasilkan konsekuensi ringan berupa monitoring awal, konsultasi singkat, atau konfirmasi progres tugas oleh asisten praktikum."),
        add_txt(doc, "Pada cutoff W2 dan W3, sistem berhasil mencapai False Negative = 0 (Recall BK 100%), dengan jumlah peringatan sebanyak 13 mahasiswa dari total 18 mahasiswa uji. Nilai F2-score yang mencapai 91,84% secara matematis membuktikan keberpihakan model pada minimasi false negative. Penurunan jumlah peringatan dari 17 pada W1 menjadi 13 pada W2 memperlihatkan bahwa beban kerja dosen telah tereduksi secara signifikan ke tingkat yang dapat dikelola (manageable workload), sembari tetap menjamin perlindungan 100% terhadap seluruh mahasiswa berisiko."),

        add_heading(doc, "3. Balanced Accuracy Lebih Informatif daripada Accuracy Saja", level=2),
        add_txt(doc, "Penggunaan metrik evaluasi yang tepat merupakan prasyarat mutlak dalam mengevaluasi EWS. Gambar 4.7 memperlihatkan bahwa pada cutoff W1, nilai Accuracy standar sebesar 44,44% menyamarkan fakta kritis bahwa spesifisitas model bernilai 0% (seluruh mahasiswa kompeten salah diklasifikasikan). Sebaliknya, Balanced Accuracy secara eksplisit menghitung rata-rata recall dari kedua kelas, sehingga mampu memberikan gambaran performa yang jujur dan tidak bias."),
        add_txt(doc, "Kebutuhan akan metrik seimbang ini semakin nyata ketika membandingkan hasil model baseline retrospektif. Sebagai contoh, Random Forest S1 menghasilkan PR-AUC yang sangat tinggi (0,858) namun memiliki Balanced Accuracy yang lebih rendah (66,67%), sedangkan Decision Tree S3 menghasilkan Balanced Accuracy 77,78% dengan Recall BK 100%. Perbedaan ini membuktikan bahwa satu model mungkin unggul dalam memeringkat probabilitas risiko (ranking capability), namun model lain lebih unggul dalam menetapkan batas keputusan kelas operasional (operating threshold). Oleh karena itu, evaluasi EWS harus mengintegrasikan Balanced Accuracy, Recall BK, F2, dan confusion matrix secara komprehensif, bukan bersandar pada akurasi konvensional semata."),

        add_heading(doc, "4. Perbandingan Karakteristik Decision Tree dan Random Forest", level=2),
        add_txt(doc, "Perbandingan antara Decision Tree dan Random Forest pada penelitian ini menyajikan wawasan berharga mengenai trade-off antara interpretabilitas dan performa ensemble pada sampel data terbatas (n=89). Sebagaimana ditunjukkan pada Gambar 4.5 dan Tabel 4.4, Random Forest pada skenario S3 unggul dalam metrik pemeringkatan probabilitas dengan PR-AUC mencapai 0,892 (dibandingkan Decision Tree sebesar 0,738). Namun demikian, pada tingkat keputusan klasifikasi locked holdout, Decision Tree S3 berhasil menyamai Balanced Accuracy Random Forest (77,78%) dan bahkan mencatatkan Recall BK yang lebih sempurna (100% berbanding 77,78%)."),
        add_txt(doc, "Keunggulan substansial Decision Tree terletak pada transparansi pohon keputusannya (white-box model). Struktur pemisahan cabang Decision Tree dapat secara langsung dikonversikan menjadi aturan operasional yang intuitif bagi pengajar praktikum, misalnya: jika nilai Early_Performance_Composite berada di bawah ambang batas tertentu, mahasiswa otomatis masuk daftar prioritas pendampingan. Hal ini menjadikan Decision Tree sangat unggul secara pedagogis dibandingkan Random Forest yang bersifat black-box, terutama ketika sistem dirancang untuk dioperasikan langsung oleh tenaga pengajar di laboratorium komputer."),

        add_heading(doc, "5. Interpretasi SHAP dan Makna Indikator Performa Awal", level=2),
        add_txt(doc, "Integrasi Explainable AI melalui analisis SHAP pada Gambar 4.8, Gambar 4.9, dan Gambar 4.10 berhasil menjembatani kesenjangan antara prediksi matematis dan intuisi pedagogis. Analisis SHAP mengungkap bahwa dominasi fitur Laporan_Mean pada model retrospektif secara efektif digantikan oleh Early_Performance_Composite pada model EWS. Temuan ini memberikan makna substantif: mahasiswa yang mengalami kesulitan pada praktikum logika-algoritma dan basis data non-relasional biasanya telah menunjukkan hambatan pemahaman sintaks, pemodelan dokumen JSON, atau logika query sejak penugasan pertama dan kedua."),
        add_txt(doc, "Meskipun demikian, penelitian ini secara tegas menekankan bahwa kontribusi SHAP harus dibaca sebagai asosiasi prediktif, bukan hubungan sebab-akibat (kausalitas). Nilai tugas pendahuluan atau laporan awal yang rendah merupakan indikator atau 'gejala' adanya hambatan belajar, bukan penyebab tunggal ketidakkompetenan mahasiswa. Pemahaman non-kausal ini sangat penting agar intervensi pengajar tidak hanya berfokus pada mengejar perbaikan angka nilai semata, melainkan menyasar akar kesulitan konseptual yang dialami mahasiswa."),

        add_heading(doc, "6. Implikasi Praktis bagi Intervensi Akademik", level=2),
        add_txt(doc, "Berdasarkan hasil pengujian pada cutoff W2, sistem menandai 13 dari 18 mahasiswa: sembilan mahasiswa Belum Kompeten dan empat mahasiswa Kompeten. Oleh sebab itu, sistem lebih tepat ditempatkan sebagai alat triase atau penyaringan awal, bukan sebagai mekanisme penetapan keputusan akademik otomatis. Peringatan dini menjadi pemicu bagi pengajar untuk memeriksa progres belajar, mengidentifikasi kesulitan spesifik, dan menawarkan bantuan yang proporsional."),
        add_txt(doc, "Untuk mengoperasionalkan sistem peringatan dini ini ke dalam tindakan bimbingan personal, pengajar memerlukan penjelasan di tingkat individu. Gambar 4.11 menyajikan visualisasi penjelasan lokal (local SHAP waterfall plot) untuk salah satu kasus mahasiswa True Positive yang berhasil dideteksi berisiko pada locked holdout."),
        add_img(doc, r"results\shap\local_TP_FINAL_S3_E_DecisionTree.png", width_in=4.8),
        add_cap(doc, "Gambar 4.11. Analisis Penjelasan Lokal (Local SHAP Waterfall Plot) pada Kasus Mahasiswa True Positive (Terdeteksi Berisiko Sejak Dini)."),
        add_txt(doc, "Visualisasi waterfall plot pada Gambar 4.11 memperlihatkan secara transparan bagaimana probabilitas risiko individual terbentuk. Dimulai dari nilai ekspektasi dasar model E[f(X)] = 0,500, batang merah menunjukkan kontribusi fitur yang meningkatkan risiko ke arah kelas Belum Kompeten. Nilai Early_Performance_Composite mahasiswa yang rendah memberikan dorongan positif terbesar (+0,41), disusul oleh rendahnya tingkat penyelesaian tugas. Informasi transparan ini memungkinkan dosen dan asisten laboratorium mengetahui akar masalah spesifik mahasiswa bersangkutan—apakah karena kesulitan logika query atau kelalaian presensi—sehingga intervensi remedial dapat dirancang secara presisi dan terarah sebelum evaluasi tengah semester."),
        add_txt(doc, "Bagi pengajar, visualisasi lokal seperti pada Gambar 4.11 mengubah peringatan sistem dari sekadar 'label risiko abstrak' menjadi diagnosis yang dapat ditindaklanjuti secara konkret. Dosen dan asisten praktikum dapat membedakan tindakan: mahasiswa dengan risiko kuat diberikan bimbingan intensif dan latihan mandiri, sementara mahasiswa yang berada di ambang batas dipantau perkembangannya. Pendekatan ini memastikan sistem berfungsi sebagai instrumen pendukung pembelajaran (learning analytics tool), bukan sebagai vonis penentu nasib mahasiswa."),

        add_heading(doc, "7. Kontribusi Metodologis dan Kebaruan", level=2),
        add_txt(doc, "Kontribusi metodologis pertama adalah memperlakukan cutoff temporal sebagai bagian eksplisit dari evaluasi, sehingga pertanyaan tentang seberapa dini prediksi dapat dilakukan diuji secara langsung. Kedua, penelitian menghubungkan metrik yang sensitif terhadap risiko dengan jumlah mahasiswa yang menerima peringatan dan jumlah false negative. Ketiga, feature engineering dan SHAP digunakan untuk menghubungkan kinerja prediksi dengan indikator yang lebih mudah ditafsirkan. Keempat, nested cross-validation, locked holdout, dan bootstrap digunakan untuk mengurangi risiko pembacaan berlebihan terhadap satu estimasi kinerja."),
        add_txt(doc, "Kontribusi tersebut sebaiknya dirumuskan sebagai kontribusi kerangka evaluasi dan temuan empiris pada dataset yang digunakan. Karena ukuran data terbatas dan evaluasi eksternal belum dilakukan, hasil ini belum cukup untuk menyatakan bahwa pendekatan yang sama akan memberikan kinerja serupa pada semua mata kuliah, semester, atau institusi."),

        add_heading(doc, "8. Keterbatasan Penelitian dan Arah Lanjutan", level=2),
        add_txt(doc, "Penelitian memiliki beberapa keterbatasan. Pertama, data berjumlah 89 mahasiswa dan locked holdout hanya 18 mahasiswa, sehingga satu atau dua prediksi dapat mengubah nilai metrik secara berarti. Kedua, data latih dan uji berasal dari cohort yang sama; generalisasi ke semester, pengajar, atau kelas lain belum diuji secara eksternal. Ketiga, ambang nilai 83 merupakan kriteria operasional penelitian dan validitasnya perlu diperiksa terhadap rubrik resmi serta konsistensi penilai. Keempat, variabel seperti keterlibatan belajar, kualitas interaksi, dan faktor kontekstual lain belum tercakup apabila tidak tersedia sebagai data numerik. Kelima, interpretasi SHAP tidak dapat dianggap sebagai bukti kausal."),
        add_txt(doc, "Penelitian lanjutan sebaiknya menguji cutoff W2 pada cohort baru, menggunakan cohort semester berikutnya sebagai external test set, memeriksa kalibrasi threshold berdasarkan kebutuhan intervensi, dan mengevaluasi apakah intervensi yang dipandu oleh hasil model benar-benar memperbaiki capaian mahasiswa. Pengembangan selanjutnya lebih bermanfaat jika diarahkan pada kestabilan dan dampak penggunaan model daripada sekadar menambah jumlah algoritma."),

        add_heading(doc, "9. Sintesis Pembahasan", level=2),
        add_txt(doc, "Secara keseluruhan, hasil pada dataset ini menunjukkan bahwa data hingga W2 telah menghasilkan Recall BK 100% dan Balanced Accuracy 77,78%, tanpa peningkatan metrik yang sama pada W3. Full Semester meningkatkan specificity menjadi 66,67%, tetapi Recall BK turun menjadi 88,89% dan Balanced Accuracy tetap 77,78%. Pola ini mendukung W2 sebagai kandidat cutoff awal yang layak diuji lebih lanjut. Kesimpulan tersebut bersifat lokal dan perlu direplikasi pada cohort lain. Nilai utama penelitian terletak pada pengaitan waktu prediksi, kemampuan mendeteksi risiko, jumlah peringatan, dan interpretasi indikator model sebagai dasar dukungan akademik yang lebih dini.")
    ]
    for p in int5:
        bab5_start.addprevious(p._p)

    # Save to REVISI_PATH first
    doc.save(REVISI_PATH)
    print(f"Successfully saved to: {REVISI_PATH}")

    # Also try to copy over DOC_PATH
    try:
        shutil.copy2(REVISI_PATH, DOC_PATH)
        print(f"Successfully overwritten original: {DOC_PATH}")
    except Exception as e:
        print(f"Could not overwrite {DOC_PATH} immediately (file may be locked by Word): {e}")

if __name__ == "__main__":
    rebuild_document()
