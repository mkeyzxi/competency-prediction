import pandas as pd

xl = pd.ExcelFile('data/raw/DBNR.xlsx')
students = []

for s in ['A', 'B', 'C', 'D', 'E']:
    df = pd.read_excel(xl, s, header=None)
    data = df.iloc[3:].copy()
    
    col_final = df.columns[-3]
    col_total = df.columns[-2]
    
    for _, row in data.iterrows():
        nim = row[0]
        nama = row[1]
        if pd.notna(nama):
            if 'asisten' in str(nama).lower() or 'asisten' in str(nim).lower():
                continue
                
            students.append({
                'NIM': str(nim).strip() if pd.notna(nim) else '',
                'Nama': str(nama).strip(),
                'Kelas': s,
                'FINAL': row[col_final]
            })

df_s = pd.DataFrame(students)
print("Total valid student names (no asisten):", len(df_s))
df_s = df_s.drop_duplicates(subset=['Nama'])
print("Total after NAMA deduplication:", len(df_s))
print(df_s['Kelas'].value_counts())
