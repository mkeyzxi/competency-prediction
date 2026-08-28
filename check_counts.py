import pandas as pd
import json

xl = pd.ExcelFile('data/raw/DBNR.xlsx')

report = {}
total_ae = 0

for s in ['A', 'B', 'C', 'D', 'E']:
    df = pd.read_excel(xl, s, header=None)
    data = df.iloc[3:].copy()
    valid = data[data[0].notna() & data[1].notna()]
    report[s] = len(valid)
    total_ae += len(valid)
    
df_final = pd.read_excel(xl, 'FINAL', header=0)
valid_final = df_final[df_final.iloc[:, 0].notna()] # Col 0 is Nama

report['Total_AE'] = total_ae
report['Total_FINAL'] = len(valid_final)

# Let's see the first column of FINAL
report['FINAL_kelas_counts'] = valid_final['Kelas'].value_counts().to_dict()

print(json.dumps(report, indent=4))
