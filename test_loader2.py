import pandas as pd

xls = pd.ExcelFile('data/raw/data gabungan testing.xlsx')

for sheet in ['A', 'B']:
    df = pd.read_excel(xls, sheet_name=sheet, header=None)
    print(f"\n--- Sheet: {sheet} ---")
    print(f"Shape: {df.shape}")
    
    # Let's print rows 0, 1, 2, 3 to see the headers
    for i in range(4):
        print(f"Row {i}: {df.iloc[i].values.tolist()[:45]}")
