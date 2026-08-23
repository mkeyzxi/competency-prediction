import pandas as pd
import os
import numpy as np
from src.utils import load_config

def load_ac_data(path):
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
        
        # Drop empty rows
        extracted = extracted.dropna(subset=['NIM']).copy()
        valid_indices = extracted.index
        df = df.loc[valid_indices]
        
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

def load_bde_data(path):
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

def load_final_uas(path):
    df_final = pd.read_excel(path, sheet_name="PENILAIAN_UAS")
    df_final = df_final.dropna(subset=['NIM']).copy()
    
    # Clean NIM
    df_final['NIM'] = df_final['NIM'].astype(str).str.strip().str.replace('.0', '', regex=False)
    
    # Handle duplicates according to rules:
    # If identical, drop 1. If different, keep highest 'final'
    df_final = df_final.sort_values(by='final', ascending=False)
    df_final = df_final.drop_duplicates(subset=['NIM'], keep='first')
    
    return df_final

def load_and_clean_data(config_path: str = 'configs/data_config.yaml'):
    """
    Load data from raw excel files according to PRD v1.1.
    Merges AC and BDE schemes with PENILAIAN_UAS.
    """
    config = load_config(config_path)
    
    ac_path = config['data']['ac_path']
    bde_path = config['data']['bde_path']
    
    df_ac = load_ac_data(ac_path)
    df_bde = load_bde_data(bde_path)
    
    # Combine activity datasets
    df_activity = pd.concat([df_ac, df_bde], ignore_index=True)
    
    # Clean NIM on activity dataset
    df_activity['NIM'] = df_activity['NIM'].astype(str).str.strip().str.replace('.0', '', regex=False)
    
    # Remove obvious non-student rows if any snuck in
    mask_valid = ~df_activity['NIM'].str.contains('Asisten|Bobot|Nilai|NIM|nan', case=False, na=False)
    df_activity = df_activity[mask_valid].copy()
    
    # Load and clean Final UAS
    df_final = load_final_uas(bde_path)
    
    # Merge activity with Final UAS (Final UAS acts as truth for Final Individu for BDE, and attendance)
    # df_final columns: NIM, final, nilai flowchart, nilai kodingan
    df_merged = pd.merge(df_activity, df_final, on='NIM', how='left')
    
    # Set Final_Individu for BDE from the UAS 'final' column
    bde_mask = df_merged['Scoring_Scheme'] == 'BDE'
    df_merged.loc[bde_mask, 'Final_Individu'] = df_merged.loc[bde_mask, 'final']
    
    # Save the interim data
    os.makedirs('data/interim', exist_ok=True)
    df_merged.to_csv('data/interim/combined_data.csv', index=False)
    
    return df_merged

if __name__ == "__main__":
    df = load_and_clean_data()
    print("Data loaded successfully. Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample AC:", df[df['Scoring_Scheme'] == 'AC'].head(1))
    print("Sample BDE:", df[df['Scoring_Scheme'] == 'BDE'].head(1))
