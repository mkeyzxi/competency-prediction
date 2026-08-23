import pandas as pd

files = [
    'data/raw/PENILAIAN LOGIKA PEMROGRAMAN KELAS A DAN C .xlsx', 
    'data/raw/PENILAIAN LOGIKA PMEROGRAMAN KELAS B, D, DAN E.xlsx'
]

for f in files:
    print(f'\n{"="*50}\n--- {f} ---\n{"="*50}')
    try:
        xls = pd.ExcelFile(f)
        for sheet in xls.sheet_names:
            print(f'\n>>> Sheet: {sheet}')
            df = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=10)
            print(df.to_string())
    except Exception as e:
        print(e)
