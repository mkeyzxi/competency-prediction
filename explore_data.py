import pandas as pd

files = [
    'data/raw/PENILAIAN LOGIKA PEMROGRAMAN KELAS A DAN C .xlsx', 
    'data/raw/PENILAIAN LOGIKA PMEROGRAMAN KELAS B, D, DAN E.xlsx'
]

for f in files:
    print(f'\n--- {f} ---')
    try:
        xls = pd.ExcelFile(f)
        for sheet in xls.sheet_names:
            print(f'\nSheet: {sheet}')
            df = pd.read_excel(xls, sheet_name=sheet, nrows=5)
            print('Columns:', df.columns.tolist())
    except Exception as e:
        print(e)
