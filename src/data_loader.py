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
    if sheet in ['A', 'C']:
        extracted['Final_Individu'] = df[37]
    else:
        extracted['Final_Individu'] = np.nan
        
    return extracted

def load_and_clean_data(config_path: str = 'configs/data_config.yaml'):
    """
    Load data from merged excel file for classes A-E.
    """
    config = load_config(config_path)
    
    merged_path = config['data']['merged_path']
    sheets = config['data']['sheets']
    
    all_data = []
    uas_list = []
    
    for sheet in sheets:
        df_sheet_raw = pd.read_excel(merged_path, sheet_name=sheet, header=None)
        df_sheet_extracted = load_sheet_data(df_sheet_raw, sheet)
        all_data.append(df_sheet_extracted)
        
        if sheet in ['B', 'D', 'E']:
            uas = pd.DataFrame()
            uas['NIM'] = df_sheet_raw.iloc[3:][44]
            uas['final_uas'] = df_sheet_raw.iloc[3:][45]
            uas_list.append(uas)
            
    df_merged = pd.concat(all_data, ignore_index=True)
    
    # Clean NIM
    df_merged['NIM'] = df_merged['NIM'].astype(str).str.strip().str.replace('.0', '', regex=False)
    
    # Remove obvious non-student rows if any snuck in
    mask_valid = ~df_merged['NIM'].str.contains('Asisten|Bobot|Nilai|NIM|nan', case=False, na=False)
    df_merged = df_merged[mask_valid].copy()
    
    # Process Master UAS
    master_uas = pd.concat(uas_list, ignore_index=True)
    master_uas['NIM'] = master_uas['NIM'].astype(str).str.strip().str.replace('.0', '', regex=False)
    master_uas = master_uas[~master_uas['NIM'].str.contains('nan', case=False, na=False)]
    master_uas = master_uas.dropna(subset=['NIM'])
    
    master_uas['final_uas'] = pd.to_numeric(master_uas['final_uas'], errors='coerce')
    master_uas = master_uas.sort_values(by='final_uas', ascending=False)
    master_uas = master_uas.drop_duplicates(subset=['NIM'], keep='first')
    
    # Merge and update Final_Individu
    df_merged = pd.merge(df_merged, master_uas, on='NIM', how='left')
    bde_mask = df_merged['Class'].isin(['B', 'D', 'E'])
    df_merged.loc[bde_mask, 'Final_Individu'] = df_merged.loc[bde_mask, 'final_uas']
    df_merged = df_merged.drop(columns=['final_uas'])
    
    # Save the interim data
    os.makedirs('data/interim', exist_ok=True)
    df_merged.to_csv('data/interim/combined_data.csv', index=False)
    
    return df_merged

if __name__ == "__main__":
    df = load_and_clean_data()
    print("Data loaded successfully. Shape:", df.shape)
    print("Sample A:", df[df['Class'] == 'A'].head(1))
    print("Sample B:", df[df['Class'] == 'B'].head(1))
