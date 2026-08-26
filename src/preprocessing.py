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
    df['Final_Individu'] = pd.to_numeric(df['Final_Individu'], errors='coerce')
    
    # Missing final mask
    missing_final = df['Final_Individu'].isna()
    
    df['Competency_Label'] = (df['Final_Individu'] >= 75).astype(int)
    df['Competency_Name'] = df['Competency_Label'].map({1: 'Kompeten', 0: 'Belum Kompeten'})
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    # P0: Raw Valid (hanya keluarkan missing final)
    df_p0 = df[~missing_final].copy()
    
    # P1: Eligible (tanpa Early Exit)
    df_p1 = df_p0[df_p0['Early_Exit_Flag'] == 0].copy()
    
    # P2: Strict Eligible (tanpa Attendance Ineligible)
    df_p2 = df_p0[df_p0['Attendance_Ineligible_Flag'] == 0].copy()
    
    # Audit trail
    audit_data = [
        {"Stage": "Raw Data", "Count": len(df)},
        {"Stage": "Missing Final (Excluded)", "Count": missing_final.sum()},
        {"Stage": "P0 (Raw Valid)", "Count": len(df_p0)},
        {"Stage": "Early Exit (Excluded from P1)", "Count": (df_p0['Early_Exit_Flag'] == 1).sum()},
        {"Stage": "P1 (Eligible)", "Count": len(df_p1)},
        {"Stage": "Attendance Ineligible (Excluded from P2)", "Count": (df_p0['Attendance_Ineligible_Flag'] == 1).sum()},
        {"Stage": "P2 (Strict Eligible)", "Count": len(df_p2)}
    ]
    pd.DataFrame(audit_data).to_csv('results/reports/population_audit.csv', index=False)
    
    df_excluded = df[missing_final].copy()
    df_excluded.to_csv('data/processed/excluded.csv', index=False)
    
    df_p0.to_csv('data/processed/population_P0.csv', index=False)
    df_p1.to_csv('data/processed/population_P1.csv', index=False)
    df_p2.to_csv('data/processed/population_P2.csv', index=False)
    
    # Simpan master untuk compatibility dengan script yang belum terupdate
    df_p0.to_csv('data/processed/eligible.csv', index=False)
    df.to_csv('data/processed/master_clean.csv', index=False)
    
    return df_p0, df_p1, df_p2

if __name__ == "__main__":
    df_interim = pd.read_csv('data/interim/combined_data.csv')
    df_p0, df_p1, df_p2 = preprocess_data(df_interim)
    print(f"Master clean shape: {df_interim.shape}")
    print(f"P0 Raw Valid: {df_p0.shape}")
    print(f"P1 Eligible: {df_p1.shape}")
    print(f"P2 Strict Eligible: {df_p2.shape}")
    print("Columns:", df_p0.columns.tolist())
