import pandas as pd
import numpy as np

ac_path = "data/raw/PENILAIAN LOGIKA PEMROGRAMAN KELAS A DAN C .xlsx"
bde_path = "data/raw/PENILAIAN LOGIKA PMEROGRAMAN KELAS B, D, DAN E.xlsx"

def load_ac(path):
    all_data = []
    for sheet in ["A", "C"]:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        # Data starts from row 3
        df = df.iloc[3:].copy()
        
        extracted = pd.DataFrame()
        extracted['NIM'] = df[1]
        extracted['Nama'] = df[2]
        extracted['Class'] = sheet
        extracted['Scoring_Scheme'] = 'AC'
        extracted['Assistant_Group'] = 'GROUP AC'
        
        # Kehadiran 1-6 from excel (cols 3 to 8) mapped to M2-M7 later
        for i in range(6):
            extracted[f'Kehadiran_raw_{i+1}'] = df[3+i]
            
        # Laporan 1-7
        for i in range(7):
            extracted[f'Laporan_{i+1}'] = df[15+i]
            
        # TP 1-6
        for i in range(6):
            extracted[f'TP_{i+1}'] = df[23+i]
            
        # Respons 1-6
        for i in range(6):
            extracted[f'Respons_{i+1}'] = df[30+i]
            
        # Final Individu
        extracted['Final_Individu'] = df[37]
        
        all_data.append(extracted)
    return pd.concat(all_data, ignore_index=True)

def load_bde(path):
    all_data = []
    for sheet in ["KELAS B", "KELAS D", "KELAS E"]:
        df = pd.read_excel(path, sheet_name=sheet, header=4)
        
        # Drop rows where NIM is NaN
        df = df.dropna(subset=['NIM']).copy()
        
        extracted = pd.DataFrame()
        extracted['NIM'] = df['NIM']
        extracted['Nama'] = df['NAMA']
        extracted['Class'] = sheet.replace('KELAS ', '')
        extracted['Scoring_Scheme'] = 'BDE'
        extracted['Assistant_Group'] = 'GROUP BDE'
        
        # Kehadiran 1-8 (cols 2 to 9)
        for i in range(8):
            extracted[f'Kehadiran_raw_{i+1}'] = df.iloc[:, 2+i]
            
        # TP + Respons (cols 24 to 30) -> 7 items
        for i in range(7):
            extracted[f'TP_Respons_{i+1}'] = df.iloc[:, 24+i]
            
        # Laporan (cols 32 to 38) -> 7 items
        for i in range(7):
            extracted[f'Laporan_{i+1}'] = df.iloc[:, 32+i]
            
        all_data.append(extracted)
    return pd.concat(all_data, ignore_index=True)

df_ac = load_ac(ac_path)
df_bde = load_bde(bde_path)

print(df_ac.shape)
print(df_bde.shape)
