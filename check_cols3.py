import pandas as pd
ac_path = "data/raw/PENILAIAN LOGIKA PEMROGRAMAN KELAS A DAN C .xlsx"
bde_path = "data/raw/PENILAIAN LOGIKA PMEROGRAMAN KELAS B, D, DAN E.xlsx"

print("AC:")
df_ac = pd.read_excel(ac_path, sheet_name="A", header=None, nrows=10)
for i, row in df_ac.iterrows():
    print(f"Row {i}:", row.tolist()[:10])

print("\nBDE:")
df_bde = pd.read_excel(bde_path, sheet_name="KELAS B", header=None, nrows=10)
for i, row in df_bde.iterrows():
    print(f"Row {i}:", row.tolist()[:10])
