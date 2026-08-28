import pandas as pd
import json

def get_89_students():
    xl = pd.ExcelFile('data/raw/DBNR.xlsx')
    students = []
    
    for s in ['A', 'B', 'C', 'D', 'E']:
        df = pd.read_excel(xl, s, header=None)
        data = df.iloc[3:].copy()
        
        # columns based on our raw_sample discovery:
        # We know col 0 is NIM, col 1 is NAMA.
        # The last three columns are FINAL, TOTAL NILAI, BOBOT
        col_final = df.columns[-3]
        col_total = df.columns[-2]
        col_bobot = df.columns[-1]
        
        for _, row in data.iterrows():
            nim = row[0]
            nama = row[1]
            if pd.notna(nim) and pd.notna(nama):
                final_val = row[col_final]
                total_val = row[col_total]
                
                students.append({
                    'NIM': str(nim).strip(),
                    'Nama': str(nama).strip(),
                    'Kelas': s,
                    'FINAL': final_val,
                    'TOTAL_NILAI': total_val
                })
                
    df_s = pd.DataFrame(students)
    print("Total before any filter:", len(df_s))
    
    # Deduplicate by NIM
    df_s = df_s.drop_duplicates(subset=['NIM'])
    print("Total after NIM deduplication:", len(df_s))
    
    # Check those with actual FINAL values
    has_final = df_s[df_s['FINAL'].notna()]
    print("Total with non-null FINAL:", len(has_final))
    
    # Wait, in raw_sample for Sheet D, someone had FINAL=0 and BOBOT='E'
    # Maybe we should check if some students don't have bobot or final?
    
    # Let's count students per class
    print("Class distribution after dedup:\n", df_s['Kelas'].value_counts())
    
    # Wait, maybe there's an exact 89 by removing missing names/NIMs from a specific list?
    # Let's print out the duplicates if any
    all_nims = [s['NIM'] for s in students]
    dups = pd.Series(all_nims).value_counts()
    print("Duplicates:\n", dups[dups > 1])
    
if __name__ == '__main__':
    get_89_students()
