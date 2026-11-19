# -*- coding: utf-8 -*-
"""
Perbaikan menyeluruh 'Draft Laporan Litapdimas 2026 ver-2.1_NEW - Copy.docx'
-> output baru: 'Draft Laporan Litapdimas 2026 ver-2.2.docx'

Cakupan:
  1. Kunci konfigurasi validasi: 5 outer folds x 5 repetisi (25 iterasi) + 3 inner folds,
     populasi 86 (42 BK + 44 K), latih 68 (33 BK + 35 K), holdout 18 (9 BK + 9 K).
  2. Hapus duplikasi gambar diagram alir (pertahankan Gambar 3.1 resmi, hapus duplikat 3.2),
     samakan caption, ubah intro H menjadi rujukan silang.
  3. Hapus paragraf pengantar lama Subbab A.3; tambah kalimat penegas threshold (A.3 & A.5).
  4. Caption Gambar 4.1 memuat semua metrik grafik (tanpa MCC).
  5. Bersihkan sisa LaTeX ($n=18$, tanda kurung kosong, spasi ganda), typo, kata nyasar.
"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import docx
from docx.oxml.ns import qn

SRC = 'Draft Laporan Litapdimas 2026 ver-2.1_NEW - Copy.docx'
DST = 'Draft Laporan Litapdimas 2026 ver-2.2.docx'

doc = docx.Document(SRC)

# ---------------------------------------------------------------
# Kunci referensi elemen SEBELUM mutasi (hindari gesernya indeks)
# ---------------------------------------------------------------
els = {i: p._element for i, p in enumerate(doc.paragraphs)}

def get_text(el):
    return ''.join(t.text or '' for t in el.findall('.//' + qn('w:t')))

def set_text(el, new_text):
    """Tulis teks baru mempertahankan format run pertama; kosongkan run lain."""
    rs = el.findall('.//' + qn('w:r'))
    first = None
    for r in rs:
        for t in r.findall(qn('w:t')):
            if first is None:
                first = t
                t.text = new_text
                t.set(qn('xml:space'), 'preserve')
            else:
                t.text = ''

def delete(el):
    el.getparent().remove(el)

def edit(i, subs, ensure_period=False):
    """Terapkan substitusi (old, new) pada teks paragraf ke-i lalu tulis ulang."""
    txt = get_text(els[i])
    for old, new in subs:
        if old not in txt:
            raise SystemExit(f'GAGAL: pola tidak ditemukan di paragraf [{i}]: {old!r}\nTeks: {txt}')
        txt = txt.replace(old, new)
    if ensure_period:
        txt = txt.rstrip()
        if not txt.endswith('.'):
            txt += '.'
    set_text(els[i], txt)

log = []

# ---------------------------------------------------------------
# 1) Kajian Terdahulu & Research Gap: kata nyasar
# ---------------------------------------------------------------
edit(134, [('serta pengembangan kurikulum. ScienceDirect Hal tersebut',
            'serta pengembangan kurikulum. Hal tersebut')])
edit(146, [('sebagai kompetensi. CSED Dengan demikian',
            'sebagai kompetensi. Dengan demikian')])
log += [(134, 'kata nyasar "ScienceDirect" dihapus'),
        (146, 'kata nyasar "CSED" dihapus')]

# ---------------------------------------------------------------
# 2) BAB III - kunci konfigurasi validasi (Desain Penelitian)
# ---------------------------------------------------------------
edit(158, [(
    'Validasi nested menggunakan 2 outer folds × 5 inner folds (dengan 5 kali pengulangan/repetisi '
    'sehingga menghasilkan total 25 putaran evaluasi) serta 3 inner folds untuk tuning hyperparameter, '
    'dipadukan dengan 18 mahasiswa (20%) sebagai data locked holdout yang dibekukan untuk pengujian akhir.',
    'Validasi nested menggunakan 5 outer folds × 5 repetisi (total 25 iterasi evaluasi) serta '
    '3 inner folds untuk tuning hyperparameter, dipadukan dengan 18 mahasiswa (20%) sebagai data '
    'locked holdout yang dibekukan untuk pengujian akhir.')])
log.append((158, 'protokol nested CV: 5 outer folds × 5 repetisi + 3 inner folds'))

# Caption Gambar 3.1 (gambar resmi dipertahankan di posisi ini)
edit(161, [(
    'Gambar 3.1. Skema Nested Stratified Cross-Validation (5 outer fold × 5 inner fold) '
    'untuk evaluasi kinerja model EWS.',
    'Gambar 3.1. Alur Desain Penelitian: Pembagian 86 Mahasiswa menjadi Data Pengembangan dan '
    '18 Locked Holdout (9 BK dan 9 K), pelaksanaan Nested Stratified Cross-Validation dengan '
    '5 outer folds × 5 repetisi (25 iterasi evaluasi) serta 3 inner folds untuk tuning hyperparameter, '
    'dan evaluasi akhir model terpilih pada locked holdout.')])
log.append((161, 'caption Gambar 3.1 disamakan dengan protokol terkunci'))

# ---------------------------------------------------------------
# 3) Typo populasi & pembagian data
# ---------------------------------------------------------------
edit(167, [('sebanyak 86 orangterdiri atas 4 kelas praktikum.',
            'sebanyak 86 orang yang terdiri atas 4 kelas praktikum.')])
edit(241, [('terbagi dalam 80% (68 orang) data training18 orang) ditetapkan sebagai locked holdout',
            'terbagi dalam 80% (68 orang) data training dan 20% (18 orang) ditetapkan sebagai locked holdout')])
log += [(167, '"86 orangterdiri" -> "86 orang yang terdiri"'),
        (241, '"data training18 orang)" -> "data training dan 20% (18 orang)"')]

# ---------------------------------------------------------------
# 4) BAB IV - pengembangan model (Metodologi)
# ---------------------------------------------------------------
edit(223, [(
    'dengan 2 outer folds × 5 inner folds (5 repetisi = 25 iterasi evaluasi) untuk optimasi parameter '
    'pada data latih (n=68)',
    'dengan 5 outer folds × 5 repetisi (25 iterasi evaluasi) dan 3 inner folds untuk tuning hyperparameter, '
    'untuk optimasi parameter pada data latih (n = 68)')])
log.append((223, 'protokol nested CV dikunci + "(n=68)" -> "(n = 68)"'))

# ---------------------------------------------------------------
# 5) Hapus duplikat Gambar 3.2 (gambar & caption), intro H jadi rujukan silang
# ---------------------------------------------------------------
edit(242, [(
    'Secara konseptual, struktur pembagian data dapat digambarkan sebagai berikut:',
    'Secara konseptual, struktur pembagian data dan skema validasi nested yang digunakan dalam '
    'penelitian ini telah dirangkum secara utuh dalam alur desain penelitian pada Gambar 3.1.')])
delete(els[243])  # gambar duplikat (versi resolusi rendah)
delete(els[244])  # caption 'Gambar 3.2 ... (2 outer × 5 inner fold)'
log += [(242, 'intro Subbab H -> rujukan silang ke Gambar 3.1'),
        (243, 'gambar duplikat dihapus (hanya Gambar 3.1 resmi dipertahankan)'),
        (244, 'caption duplikat Gambar 3.2 dihapus')]

# ---------------------------------------------------------------
# 6) Ringkasan BAB IV - kunci konfigurasi
# ---------------------------------------------------------------
edit(268, [(
    'dengan 2 outer folds × 5 inner folds (5 repetisi = 25 iterasi evaluasi) guna mencegah',
    'dengan 5 outer folds × 5 repetisi (25 iterasi evaluasi) dan 3 inner folds untuk tuning '
    'hyperparameter guna mencegah')])
log.append((268, 'protokol nested CV dikunci (ringkasan)'))

# ---------------------------------------------------------------
# 7) LaTeX mentah: $n=18$
# ---------------------------------------------------------------
edit(270, [('($n=18$)', '(n = 18)')])
edit(279, [('($n=18$)', '(n = 18)')])
log += [(270, '"($n=18$)" -> "(n = 18)"'), (279, '"($n=18$)" -> "(n = 18)"')]

# ---------------------------------------------------------------
# 8) Caption Gambar 4.1: semua metrik grafik, tanpa MCC
# ---------------------------------------------------------------
edit(275, [(
    'Gambar 4.1. Evolusi Recall BK, Specificity, dan Balanced Accuracy menurut Cutoff Waktu '
    'pada Locked Holdout (n=18).',
    'Gambar 4.1. Perbandingan Recall BK, Precision BK, F2 BK, Specificity, dan Balanced Accuracy '
    'menurut Cutoff Waktu pada Locked Holdout (n = 18).')])
log.append((275, 'caption 4.1: Recall BK, Precision BK, F2 BK, Specificity, BA (tanpa MCC)'))

# ---------------------------------------------------------------
# 9) Hapus paragraf pengantar duplikat Subbab A.3
# ---------------------------------------------------------------
delete(els[278])
log.append((278, 'paragraf pengantar lama A.3 dihapus (mengulang Tabel 4.2)'))

# ---------------------------------------------------------------
# 10) Caption 4.2 & 4.3: spasi "n = 18" + tanda baca
# ---------------------------------------------------------------
edit(285, [('(n=18)', '(n = 18)')], ensure_period=True)
edit(301, [('(n=18)', '(n = 18)')], ensure_period=True)
log += [(285, 'caption 4.2 dirapikan'), (301, 'caption 4.3 dirapikan')]

# ---------------------------------------------------------------
# 11) Kalimat penegas threshold (A.3 dan A.5)
# ---------------------------------------------------------------
edit(287, [(
    'sebelum evaluasi tengah semester.',
    'sebelum evaluasi tengah semester. Perlu ditegaskan bahwa perbedaan ambang keputusan (threshold) '
    'antara model EWS W2 (t = 0,20) dan model Baseline S3 (t = 0,52) merupakan hasil penguncian dari '
    'optimasi inner fold pada data latih (n = 68), bukan nilai yang disesuaikan secara manual pada '
    'data locked holdout (n = 18).')])
edit(302, [(
    'tanpa mengorbankan sensitivitas deteksi risiko.',
    'tanpa mengorbankan sensitivitas deteksi risiko. Perlu ditegaskan pula bahwa kesamaan matriks '
    'tersebut dicapai dengan ambang keputusan (threshold) yang berbeda, yaitu t = 0,20 pada model '
    'EWS W2 dan t = 0,52 pada model Baseline S3, yang keduanya dikunci dari hasil optimasi inner '
    'fold pada data latih (n = 68), bukan disesuaikan secara manual pada data locked holdout (n = 18).')])
log += [(287, 'kalimat penegas threshold ditambahkan (A.3)'),
        (302, 'kalimat penegas threshold ditambahkan (A.5)')]

# ---------------------------------------------------------------
# 12) Rapikan spasi ganda sisa
# ---------------------------------------------------------------
edit(296, [('Tabel 4.4.  ', 'Tabel 4.4. ')])

# ---------------------------------------------------------------
# 13) Sisa LaTeX Bootstrap: kurung kosong & iterasi
# ---------------------------------------------------------------
edit(307, [('yang terbatas (), evaluasi', 'yang terbatas (n = 18), evaluasi'),
           ('metode bootstrap sebanyak  iterasi', 'metode bootstrap sebanyak B = 1000 iterasi')])
edit(309, [('(B=1000)', '(B = 1000)')])
log += [(307, '"()" -> "(n = 18)", "bootstrap sebanyak B = 1000 iterasi"'),
        (309, '"(B=1000)" -> "(B = 1000)"')]

# ---------------------------------------------------------------
# 14) Sisa LaTeX SHAP: nilai kosong & kurung kosong (Gambar 4.4/4.5)
# ---------------------------------------------------------------
edit(318, [('terdorong kuat ke sisi kanan hingga nilai SHAP  menuju prediksi BK',
            'terdorong kuat ke sisi kanan hingga nilai SHAP positif menuju prediksi BK'),
           ('berada di sisi kiri () menuju kelas Kompeten',
            'berada di sisi kiri (nilai SHAP negatif) menuju kelas Kompeten')])
edit(323, [('dorongan kuat ke sisi negatif ( hingga ) yang menahan',
            'dorongan kuat ke sisi negatif yang menahan'),
           ('bergeser ke sisi positif () untuk memicu',
            'bergeser ke sisi positif untuk memicu')])
log += [(318, 'sisa LaTeX SHAP Gambar 4.4 dibersihkan'),
        (323, 'sisa LaTeX SHAP Gambar 4.5 dibersihkan, paragraf utuh')]

# ---------------------------------------------------------------
# Simpan
# ---------------------------------------------------------------
doc.save(DST)
print(f' tersimpan: {DST}')
print()
for i, note in log:
    print(f'  [{i}] {note}')
