import pandas as pd
bde_path = "data/raw/PENILAIAN LOGIKA PMEROGRAMAN KELAS B, D, DAN E.xlsx"
df_bde = pd.read_excel(bde_path, sheet_name="KELAS B", header=4)
print(df_bde.columns.tolist()[:30])
print(df_bde.head(3))
