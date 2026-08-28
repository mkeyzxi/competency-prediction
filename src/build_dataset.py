import pandas as pd
import numpy as np
import os
import re

def to_float(v):
    if pd.isna(v): return np.nan
    if isinstance(v, str):
        v = v.replace(',', '.')
        try: return float(v)
        except: return np.nan
    return float(v)

def main():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    
    xl = pd.ExcelFile('data/raw/DBNR.xlsx')
    
    students = []
    activities = []
    
    for sheet_name in ['A', 'B', 'C', 'D', 'E']:
        df = pd.read_excel(xl, sheet_name, header=None)
        
        header0 = df.iloc[0].fillna('').astype(str).tolist()
        header1 = df.iloc[1].fillna('').astype(str).tolist()
        
        col_nim = 0
        col_nama = 1
        
        col_final = df.columns[-3]
        
        idx_tot_hadir = -1
        idx_tot_lap = -1
        idx_tot_tp = -1
        
        for i, val in enumerate(header0):
            val_upper = val.upper()
            if 'TOTAL NILAI HADIR' in val_upper: idx_tot_hadir = i
            elif 'TOTAL NILAI LAPORAN' in val_upper: idx_tot_lap = i
            elif 'TOTAL NILAI TP' in val_upper: idx_tot_tp = i
            
        kehadiran_cols = list(range(2, idx_tot_hadir)) if idx_tot_hadir > 2 else []
        if kehadiran_cols and 'total' in header1[kehadiran_cols[-1]].lower():
            kehadiran_cols = kehadiran_cols[:-1]
            
        laporan_cols = list(range(idx_tot_hadir + 1, idx_tot_lap)) if idx_tot_lap > -1 else []
        tp_cols = list(range(idx_tot_lap + 1, idx_tot_tp)) if idx_tot_tp > -1 else []
        
        data = df.iloc[3:].copy()
        for idx, row in data.iterrows():
            nim = str(row[col_nim]).strip() if pd.notna(row[col_nim]) else ''
            nama = str(row[col_nama]).strip() if pd.notna(row[col_nama]) else ''
            
            if not row.notna().any() or not nama:
                continue
                
            final_score = to_float(row[col_final])
            
            if pd.isna(final_score):
                final_status = "Tidak Mengikuti Final"
                label_reason = "No_Final_Attendance"
                comp_label = 0
            elif final_score >= 83:
                final_status = "Mengikuti Final"
                label_reason = "Final_Score>=83"
                comp_label = 1
            else:
                final_status = "Mengikuti Final"
                label_reason = "Final_Score<83"
                comp_label = 0
                
            students.append({
                'NIM': nim,
                'Nama': nama,
                'Kelas': sheet_name,
                'Final_Score': final_score,
                'Final_Status': final_status,
                'Label_Reason': label_reason,
                'Competency_Label': comp_label
            })
            
            for i, c in enumerate(kehadiran_cols):
                val = row[c]
                if pd.notna(val):
                    activities.append({
                        'Nama': nama,
                        'Kelas': sheet_name,
                        'Activity_Type': 'Attendance',
                        'Activity_ID': f'P{i+1}',
                        'Time_Index': i+1,
                        'Score': float(val),
                        'Available': 1
                    })
                    
            for i, c in enumerate(laporan_cols):
                val = row[c]
                if pd.notna(val):
                    activities.append({
                        'Nama': nama,
                        'Kelas': sheet_name,
                        'Activity_Type': 'Laporan',
                        'Activity_ID': f'L{i+1}',
                        'Time_Index': i+1,
                        'Score': float(val),
                        'Available': 1
                    })
                    
            for i, c in enumerate(tp_cols):
                val = row[c]
                if pd.notna(val):
                    activities.append({
                        'Nama': nama,
                        'Kelas': sheet_name,
                        'Activity_Type': 'TP',
                        'Activity_ID': f'TP{i+1}',
                        'Time_Index': i+1,
                        'Score': float(val),
                        'Available': 1
                    })
                    
    df_students = pd.DataFrame(students)
    df_students = df_students.drop_duplicates(subset=['Nama'])
    
    print(f"Total Unique Students: {len(df_students)}")
    print("Class Distribution:\n", df_students['Competency_Label'].value_counts())
    print("Label Reason Distribution:\n", df_students['Label_Reason'].value_counts())
    
    assert len(df_students) == 89
    # Removed hardcoded assertions for specific class counts as the threshold was changed to 83
    
    df_students.to_csv('data/processed/students_master.csv', index=False)
    
    df_act = pd.DataFrame(activities)
    df_act.to_csv('data/processed/activities_long.csv', index=False)
    
    print("Exported processed datasets (V2).")

if __name__ == "__main__":
    main()
