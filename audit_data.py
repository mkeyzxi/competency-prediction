import pandas as pd
import numpy as np
import json

def load_data():
    xl = pd.ExcelFile('data/raw/DBNR.xlsx')
    sheets = ['A', 'B', 'C', 'D', 'E']
    
    all_students = []
    
    for sheet in sheets:
        df = pd.read_excel(xl, sheet, header=None)
        
        # Determine column indices dynamically
        row0 = df.iloc[0].fillna('').astype(str).tolist()
        row1 = df.iloc[1].fillna('').astype(str).tolist()
        
        # Start reading from row 3 (index 2), as row 0 and 1 are headers and row 2 might be empty or spacer
        data = df.iloc[3:].copy()
        
        for idx, row in data.iterrows():
            nim = row[0]
            nama = row[1]
            if pd.isna(nama) and pd.isna(nim):
                continue
                
            student = {
                'NIM': nim,
                'Nama': nama,
                'Kelas': sheet
            }
            all_students.append(student)
            
    df_students = pd.DataFrame(all_students)
    
    df_final = pd.read_excel(xl, 'FINAL', header=0)
    
    # Audit info
    audit = {
        'total_sheets_A_E': len(df_students),
        'total_final': len(df_final),
        'missing_nim_A_E': int(df_students['NIM'].isna().sum()),
        'missing_nim_FINAL': int(df_final['NIM'].isna().sum()),
        'unique_kelas_final': df_final['Kelas'].unique().tolist(),
        'missing_nama_A_E': int(df_students['Nama'].isna().sum()),
        'missing_nama_FINAL': int(df_final['Nama'].isna().sum()),
    }
    
    # Clean names for joining
    def clean_name(n):
        if pd.isna(n): return ''
        return str(n).strip().lower()
        
    df_students['Nama_clean'] = df_students['Nama'].apply(clean_name)
    df_final['Nama_clean'] = df_final['Nama'].apply(clean_name)
    
    # Join on Nama_clean and Kelas
    merged = pd.merge(df_students, df_final, on=['Nama_clean', 'Kelas'], how='outer', indicator=True)
    audit['merged_both'] = int((merged['_merge'] == 'both').sum())
    audit['merged_left_only'] = int((merged['_merge'] == 'left_only').sum())
    audit['merged_right_only'] = int((merged['_merge'] == 'right_only').sum())
    
    # Dump missing matches
    left_only = merged[merged['_merge'] == 'left_only'][['Nama_x', 'Kelas']]
    right_only = merged[merged['_merge'] == 'right_only'][['Nama_y', 'Kelas']]
    
    audit['unmatched_A_E'] = left_only.to_dict('records')
    audit['unmatched_FINAL'] = right_only.to_dict('records')
    
    with open('audit_report.json', 'w') as f:
        json.dump(audit, f, indent=4)
        
if __name__ == "__main__":
    load_data()
