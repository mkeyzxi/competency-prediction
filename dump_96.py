import pandas as pd
xl = pd.ExcelFile('data/raw/DBNR.xlsx')
for s in ['A', 'B', 'C', 'D', 'E']:
    df = pd.read_excel(xl, s, header=None)
    data = df.iloc[3:].copy()
    for _, row in data.iterrows():
        if pd.notna(row[0]) and pd.notna(row[1]):
            print(f"{s} | NIM: {row[0]} | NAMA: {row[1]}")
