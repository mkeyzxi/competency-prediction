import pandas as pd
ac_path = "data/raw/PENILAIAN LOGIKA PEMROGRAMAN KELAS A DAN C .xlsx"
print("AC (skiprows=3):")
df_ac = pd.read_excel(ac_path, sheet_name="A", skiprows=3)
print(df_ac.columns.tolist()[:30])
print(df_ac.iloc[0].tolist()[:30])
