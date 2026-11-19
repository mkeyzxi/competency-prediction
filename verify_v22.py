# -*- coding: utf-8 -*-
"""Verifikasi menyeluruh Draft ver-2.2."""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import docx
from docx.oxml.ns import qn

doc = docx.Document('Draft Laporan Litapdimas 2026 ver-2.2.docx')
print('Paragraf:', len(doc.paragraphs), '| Tabel:', len(doc.tables))

alltexts = [p.text for p in doc.paragraphs]
for tb in doc.tables:
    for r in tb.rows:
        for c in r.cells:
            alltexts.append(c.text)

DOLAR = '$'
BACKS = '\\'
bad = {
    'tanda dolar': DOLAR,
    'backslash': BACKS,
    '"2 outer"': '2 outer',
    '"5 inner"': '5 inner',
    '"5 outer fold × 5 inner"': '5 outer fold × 5 inner',
    '"ScienceDirect"': 'ScienceDirect',
    '"CSED"': 'CSED',
    '"orangterdiri"': 'orangterdiri',
    '"training18"': 'training18',
    '"explainable Al"': 'explainable Al',
    '"Explainable Al"': 'Explainable Al',
    '"Gambar 3.2"': 'Gambar 3.2',
    '"n=18"': 'n=18',
    '"n=68"': 'n=68',
    '"B=1000"': 'B=1000',
    'kurung kosong "()"': '()',
}
ok = True
for name, pat in bad.items():
    hits = [f'[{i}] {t[:95]}' for i, t in enumerate(alltexts) if pat in t]
    if hits:
        ok = False
        print(f'MASIH ADA {name}:')
        for h in hits:
            print('   ' + h)
print('SEMUA BERSIH' if ok else 'ADA SISA!')
print()

n_img = sum(1 for p in doc.paragraphs for b in p._element.iter()
            if b.tag.endswith('}blip') and b.get(qn('r:embed')))
print('Jumlah gambar tersisip:', n_img)
print()

# teks akhir paragraf-paragraf yang diedit (posisi bergeser -1 setelah 3 penghapusan sebelumnya)
print('=== Posisi final paragraf kunci ===')
for i, p in enumerate(doc.paragraphs):
    t = p.text
    if (t.startswith('Validasi nested menggunakan')
            or t.startswith('Gambar 3.1.')
            or t.startswith('Secara konseptual, struktur pembagian')
            or t.startswith('Proses pengembangan model menggunakan nested')
            or t.startswith('Analisis menggunakan data dari 86')
            or t.startswith('Gambar 4.1.')
            or t.startswith('Distribusi klasifikasi pada setiap cutoff')
            or t.startswith('Gambar 4.2.')
            or t.startswith('Visualisasi pada Gambar 4.2')
            or t.startswith('Gambar 4.3.')
            or t.startswith('Sebaran kuadran pada Gambar 4.3')
            or t.startswith('Mengingat ukuran sampel')
            or t.startswith('Tabel 4.5.')
            or t.startswith('Pada Gambar 4.4')
            or t.startswith('Distribusi pada Gambar 4.5')
            or t.startswith('Populasi dalam penelitian')
            or t.startswith('Pembagian data dalam penelitian')
            or 'Subbab A' in t and t.startswith('3. Confusion')):
        print(f'[{i}] {t}')
        print()
