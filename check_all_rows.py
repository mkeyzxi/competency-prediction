import pandas as pd
xl = pd.ExcelFile('data/raw/DBNR.xlsx')
c = 0
for s in ['A', 'B', 'C', 'D', 'E']:
    df = pd.read_excel(xl, s, header=None)
    data = df.iloc[3:].copy()
    for _, row in data.iterrows():
        # Check if row has any non-NaN values
        if row.notna().any():
            nim = str(row[0]) if pd.notna(row[0]) else ''
            nama = str(row[1]) if pd.notna(row[1]) else ''
            
            # exclude Asisten? Let's print everything that is not an Asisten
            if 'asisten' not in nama.lower() and 'asisten' not in nim.lower():
                print(f"[{s}] NIM: {nim} | NAMA: {nama}")
                c += 1
print('Total:', c)
