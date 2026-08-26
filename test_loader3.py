import pandas as pd
import numpy as np

xls = pd.ExcelFile('data/raw/data gabungan testing.xlsx')

for sheet in ['A', 'C']:
    df = pd.read_excel(xls, sheet_name=sheet, header=None)
    print(f"\n--- Sheet {sheet} ---")
    print(df.iloc[0:4, 37:45].values)

for sheet in ['B', 'D', 'E']:
    df = pd.read_excel(xls, sheet_name=sheet, header=None)
    print(f"\n--- Sheet {sheet} ---")
    print(df.iloc[0:4, 40:48].values)
