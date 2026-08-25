import pandas as pd
import numpy as np
from src.utils import load_config
import os

def check_presence(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return 1.0 if val > 0 else 0.0
    val = str(val).strip()
    return 1.0 if val and val.lower() not in ['0', '0.0', 'nan', 'none'] else 0.0

def clean_attendance_value(val):
    if pd.isna(val):
        return 0.0
    try:
        num = float(val)
        if num > 0 and num < 1:
            return 0.5
        elif num >= 1:
            return 1.0
        else:
            return 0.0
    except:
        return 0.0

def preprocess_data(df, config_path='configs/attendance_mapping.yaml'):
    # 1. Reconstruct 10 attendances uniformly for all classes
    for i in range(1, 11):
        raw_col = f'Kehadiran_raw_{i}'
        df[f'Attendance_{i}'] = df[raw_col].apply(clean_attendance_value)
    
    # 2. Calculate absence count
    attendance_cols = [f'Attendance_{i}' for i in range(1, 11)]
    # Count sessions where attendance == 0
    df['absence_count'] = (df[attendance_cols] == 0.0).sum(axis=1)
    
    # 3. Create Flags (As per user clarification)
    # hadir < 3 -> absen > 7
    df['Early_Exit_Flag'] = (df['absence_count'] > 7).astype(int)
    # absen >= 4 -> tidak lanjut
    df['Attendance_Ineligible_Flag'] = (df['absence_count'] >= 4).astype(int)
    
    # 4. Target Labeling
    # Convert Final_Individu to numeric
    df['Final_Individu'] = pd.to_numeric(df['Final_Individu'], errors='coerce')
    
    # Exclude missing finals from target? The PRD says "Masukkan quality issue; perbaiki dari sumber resmi atau keluarkan dari modelling."
    # We will mark them and drop from the modeling 'eligible' dataset.
    df['Competency_Label'] = (df['Final_Individu'] >= 75).astype(int)
    df['Competency_Name'] = df['Competency_Label'].map({1: 'Kompeten', 0: 'Belum Kompeten'})
    
    # Save excluded and eligible
    os.makedirs('data/processed', exist_ok=True)
    
    # Excluded: missing final or ineligible (Early Exit is subsumed by Ineligible since >7 is >=4)
    excluded_mask = df['Final_Individu'].isna() | (df['Attendance_Ineligible_Flag'] == 1)
    
    df_excluded = df[excluded_mask].copy()
    df_eligible = df[~excluded_mask].copy()
    
    df_excluded.to_csv('data/processed/excluded.csv', index=False)
    df_eligible.to_csv('data/processed/eligible.csv', index=False)
    df.to_csv('data/processed/master_clean.csv', index=False)
    
    return df_eligible

if __name__ == "__main__":
    df_interim = pd.read_csv('data/interim/combined_data.csv')
    df_clean = preprocess_data(df_interim)
    print(f"Master clean shape: {df_interim.shape}")
    print(f"Eligible shape: {df_clean.shape}")
    print("Columns:", df_clean.columns.tolist())
