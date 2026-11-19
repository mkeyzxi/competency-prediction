"""Pemeriksaan menyeluruh dokumen Copy: LaTeX, diagram teks, soft-break, outer fold, tanda baca."""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import docx
from docx.oxml.ns import qn

doc = docx.Document('Draft Laporan Litapdimas 2026 ver-2.1_NEW - Copy.docx')

print('=== 1) Dolar / backslash / LaTeX di paragraf ===')
for i, p in enumerate(doc.paragraphs):
    t = p.text
    if '$' in t or '\\' in t:
        print(f'[{i}] {t}')
        print()

print('=== 2) Karakter diagram teks (arrow/box) di paragraf ===')
found = False
for i, p in enumerate(doc.paragraphs):
    t = p.text
    if any(c in t for c in ['\u2192', '\u2193', '\u2514', '\u250c', '\u2502', '\u2500', '\u2550']):
        print(f'[{i}] {t[:150]}')
        found = True
if not found:
    print('(tidak ada)')

print()
print('=== 3) Soft line-break (w:br / w:cr) ===')
cnt = 0
for i, p in enumerate(doc.paragraphs):
    brs = p._element.findall('.//' + qn('w:br'))
    crs = p._element.findall('.//' + qn('w:cr'))
    if brs or crs:
        print(f'[{i}] br={len(brs)} cr={len(crs)} text="{p.text.strip()[:90]}"')
        cnt += 1
if cnt == 0:
    print('(tidak ada)')

print()
print('=== 4) outer/inner fold (paragraf & tabel) ===')
for i, p in enumerate(doc.paragraphs):
    tl = p.text.lower()
    if 'outer fold' in tl or 'inner fold' in tl:
        print(f'P[{i}] {p.text[:140]}')
for ti, tb in enumerate(doc.tables):
    seen = set()
    for r in tb.rows:
        for c in r.cells:
            tl = c.text.lower()
            if ('outer fold' in tl or 'inner fold' in tl) and c.text[:60] not in seen:
                seen.add(c.text[:60])
                print(f'TABEL[{ti}] {c.text[:140]}')

print()
print('=== 5) Tail repr: caption 4.2/4.3 & kalimat (FP = 4, TN = 5) ===')
for i in [283, 285, 301, 302]:
    print(f'[{i}] ...{doc.paragraphs[i].text[-100:]!r}')

print()
print('=== 6) Paragraf dengan tanda kurung kosong "()" atau spasi ganda mencurigakan ===')
import re
for i, p in enumerate(doc.paragraphs):
    t = p.text
    if '()' in t or '  ' in t.replace('\t', ' '):
        flags = []
        if '()' in t:
            flags.append('EMPTY-PARENS')
        print(f'[{i}] {"+".join(flags)} {t[:120]}')

print()
print('=== 7) n= / B= / t=0 tanpa spasi (konsistensi) ===')
for i, p in enumerate(doc.paragraphs):
    t = p.text
    for pat in ['n=', 'n=1', 'B=', 't=0', 'TN=', 'FP=', 'FN=', 'TP=']:
        if pat in t:
            print(f'[{i}] {t[:130]}')
            break
