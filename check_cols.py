import pandas as pd

ac_path = "data/raw/PENILAIAN LOGIKA PEMROGRAMAN KELAS A DAN C .xlsx"
bde_path = "data/raw/PENILAIAN LOGIKA PMEROGRAMAN KELAS B, D, DAN E.xlsx"

print("AC:")
df_ac = pd.read_excel(ac_path, sheet_name="A", skiprows=4)
print(df_ac.columns.tolist()[:30])

print("\nBDE:")
df_bde = pd.read_excel(bde_path, sheet_name="KELAS B", skiprows=4)
print(df_bde.columns.tolist()[:30])

df_final = pd.read_excel(bde_path, sheet_name="PENILAIAN_UAS")
print("\nFINAL:")
print(df_final.columns.tolist())
