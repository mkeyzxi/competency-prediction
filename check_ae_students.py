import pandas as pd
xl = pd.ExcelFile('data/raw/DBNR.xlsx')
total_students = 0
for s in ['A', 'B', 'C', 'D', 'E']:
    df = pd.read_excel(xl, s, header=None)
    data = df.iloc[3:].copy()
    valid = data[data[0].notna() & data[1].notna()]
    total_students += len(valid)
    print(f"Sheet {s}: {len(valid)} students")
print("Total A-E:", total_students)
