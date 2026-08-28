import pandas as pd
import re

xl = pd.ExcelFile('data/raw/DBNR.xlsx')
students = []

for s in ['A', 'B', 'C', 'D', 'E']:
    df = pd.read_excel(xl, s, header=None)
    data = df.iloc[3:].copy()
    
    col_final = df.columns[-3]
    col_total = df.columns[-2]
    
    for _, row in data.iterrows():
        nim = str(row[0]).strip() if pd.notna(row[0]) else ''
        nama = str(row[1]).strip() if pd.notna(row[1]) else ''
        
        # Check if the row is effectively empty
        if not row.notna().any():
            continue
            
        # Is it an asisten?
        is_asisten = False
        nama_lower = nama.lower()
        nim_lower = nim.lower()
        
        if 'asisten' in nama_lower or 'asisten' in nim_lower:
            is_asisten = True
            
        # Hardcode known asisten names from the sheets
        if 'anugrah' in nama_lower and 'alif' in nama_lower: is_asisten = True
        if 'syafriawan' in nama_lower: is_asisten = True
        if 'syahid' in nama_lower and 'ahmad' in nama_lower: is_asisten = True
        
        if not is_asisten and nama:
            students.append({
                'NIM': nim,
                'Nama': nama,
                'Kelas': s,
                'FINAL': row[col_final],
                'TOTAL_NILAI': row[col_total]
            })

df_s = pd.DataFrame(students)
print("Before dedup:", len(df_s))

# Find duplicated NIMs
valid_nims = df_s[df_s['NIM'] != '']
dups = valid_nims[valid_nims.duplicated('NIM', keep=False)]
if len(dups) > 0:
    print("Duplicates:\n", dups[['NIM', 'Nama', 'Kelas']])
    
df_s = df_s.drop_duplicates(subset=['NIM'], keep='first')
print("After dedup by NIM:", len(df_s))
