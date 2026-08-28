import pandas as pd
xl = pd.ExcelFile('data/raw/DBNR.xlsx')
c = 0
for s in ['A', 'B', 'C', 'D', 'E']:
    df = pd.read_excel(xl, s, header=None)
    data = df.iloc[3:].copy()
    for i, row in data.iterrows():
        nama = str(row[1]).lower() if pd.notna(row[1]) else ''
        nim = str(row[0]).lower() if pd.notna(row[0]) else ''
        
        # exclude asisten
        is_asisten = False
        # Asisten are Ahmad Syahid, Muhammad Alif Hardhy Anugrah, Muh. Anwar Syafriawan (from my earlier findings)
        # Let's just check if their names are exactly these, or if 'asisten' is in the name.
        if 'anugrah' in nama and 'alif' in nama: is_asisten = True
        if 'syafriawan' in nama: is_asisten = True
        if 'syahid' in nama and 'ahmad' in nama: is_asisten = True
        
        # Also let's check if the row has any non-null data at all
        has_data = row.notna().any()
        
        if has_data and not is_asisten:
            c += 1
            if not nama and not nim:
                print(f"Row {i} in {s} has NO NIM and NO NAMA but has data: \n{row}")

print(f"Total valid rows (excluding known asisten): {c}")
