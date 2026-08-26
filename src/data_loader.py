import pandas as pd
import os
import numpy as np
from src.utils import load_config

def load_sheet_data(df, sheet):
    # Data starts from row 3 (0-indexed index 3)
    df = df.iloc[3:].copy()
    
    extracted = pd.DataFrame()
    extracted['NIM'] = df[1]
    extracted['Nama'] = df[2]
    extracted['Class'] = sheet
    extracted['Scoring_Scheme'] = 'UNIFIED'
    
    # Extract NIM early for cleaning
    extracted['NIM'] = extracted['NIM'].astype(str).str.strip().str.replace('.0', '', regex=False)
    
    # Drop rows where NIM is NaN or empty early to prevent processing empty rows
    extracted = extracted[~extracted['NIM'].str.contains('nan', case=False, na=False)].copy()
    extracted = extracted.dropna(subset=['NIM']).copy()
    valid_indices = extracted.index
    df = df.loc[valid_indices]
    
    # Kehadiran 1-10
    for i in range(10):
        extracted[f'Kehadiran_raw_{i+1}'] = pd.to_numeric(df[3+i], errors='coerce').fillna(0)
        
    if sheet in ['A', 'C']:
        # Laporan 1-7 (col 15-21)
        for i in range(7):
            extracted[f'Laporan_{i+1}'] = pd.to_numeric(df[15+i], errors='coerce').fillna(0)
        extracted['Laporan_8'] = np.nan
        
        # TP 1-6 (col 23-28)
        for i in range(6):
            extracted[f'TP_{i+1}'] = pd.to_numeric(df[23+i], errors='coerce').fillna(0)
        extracted['TP_7'] = np.nan
        extracted['TP_8'] = np.nan
        
        # Respons 1-6 (col 30-35)
        for i in range(6):
            extracted[f'Respons_{i+1}'] = pd.to_numeric(df[30+i], errors='coerce').fillna(0)
        extracted['Respons_7'] = np.nan
        extracted['Respons_8'] = np.nan
        
        extracted['Final_Individu'] = df[37]
        
    else: # B, D, E
        # Laporan 1-8 (col 15-22)
        for i in range(8):
            extracted[f'Laporan_{i+1}'] = pd.to_numeric(df[15+i], errors='coerce').fillna(0)
            
        # TP 1-8 (col 24-31)
        for i in range(8):
            extracted[f'TP_{i+1}'] = pd.to_numeric(df[24+i], errors='coerce').fillna(0)
            
        # Respons 1-8 (col 33-40)
        for i in range(8):
            extracted[f'Respons_{i+1}'] = pd.to_numeric(df[33+i], errors='coerce').fillna(0)
            
        extracted['Final_Individu'] = df[44]
        
    return extracted

def load_and_clean_data(config_path: str = 'configs/data_config.yaml'):
    """
    Load data from merged excel file for classes A-E.
    """
    config = load_config(config_path)
    
    merged_path = config['data']['merged_path']
    sheets = config['data']['sheets']
    
    all_data = []
    
    for sheet in sheets:
        df_sheet_raw = pd.read_excel(merged_path, sheet_name=sheet, header=None)
        df_sheet_extracted = load_sheet_data(df_sheet_raw, sheet)
        all_data.append(df_sheet_extracted)
            
    df_merged = pd.concat(all_data, ignore_index=True)
    
    # Clean NIM
    df_merged['NIM'] = df_merged['NIM'].astype(str).str.strip().str.replace('.0', '', regex=False)
    
    # Remove obvious non-student rows if any snuck in
    mask_valid = ~df_merged['NIM'].str.contains('Asisten|Bobot|Nilai|NIM|nan', case=False, na=False)
    df_merged = df_merged[mask_valid].copy()
    
    df_merged['Final_Individu'] = pd.to_numeric(df_merged['Final_Individu'], errors='coerce')
    
    # Save the interim data
    os.makedirs('data/interim', exist_ok=True)
    df_merged.to_csv('data/interim/combined_data.csv', index=False)
    
    return df_merged

if __name__ == "__main__":
    df = load_and_clean_data()
    print("Data loaded successfully. Shape:", df.shape)
    print("Sample A:", df[df['Class'] == 'A'].head(1))
    print("Sample B:", df[df['Class'] == 'B'].head(1))
