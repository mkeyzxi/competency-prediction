import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame, cutoff_session='PreFinal') -> pd.DataFrame:
    df = df.copy()
    
    # Determine the index limits based on cutoff_session
    if cutoff_session == 'M3':
        idx_max = 3
    elif cutoff_session == 'M5':
        idx_max = 5
    elif cutoff_session == 'M7':
        idx_max = 7
    elif cutoff_session == 'PreFinal':
        idx_max = 8
    else:
        raise ValueError(f"Unknown cutoff_session: {cutoff_session}")
        
    pre_final_cols = [f'Attendance_{i}' for i in range(1, idx_max + 1)]
    
    # Task columns logic
    task_idx_max = min(idx_max, 6) # Max TP and Respons is 6
    lap_idx_max = min(idx_max, 7) # Max Laporan is 7
    
    tp_cols = [f'TP_{i}' for i in range(1, task_idx_max + 1)]
    respon_cols = [f'Respons_{i}' for i in range(1, task_idx_max + 1)]
    laporan_cols = [f'Laporan_{i}' for i in range(1, lap_idx_max + 1)]
    
    all_raw_cols = [f'Attendance_{i}' for i in range(1, 11)] + \
                   [f'TP_{i}' for i in range(1, 7)] + \
                   [f'Respons_{i}' for i in range(1, 7)] + \
                   [f'Laporan_{i}' for i in range(1, 8)]
                   
    for col in all_raw_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Compute Means and Rates
    df['Attendance_PreFinal_Rate'] = df[pre_final_cols].fillna(0).mean(axis=1) if pre_final_cols else 0.0
    
    df['TP_Response_Source'] = 'SEPARATE'
    
    df['Laporan_Mean'] = df[laporan_cols].fillna(0).mean(axis=1) if laporan_cols else 0.0
    df['TP_Mean'] = df[tp_cols].fillna(0).mean(axis=1) if tp_cols else 0.0
    df['Respons_Mean'] = df[respon_cols].fillna(0).mean(axis=1) if respon_cols else 0.0
    
    df['Laporan_Completion_Rate'] = (df[laporan_cols].fillna(0) > 0).mean(axis=1) if laporan_cols else 0.0
    df['TP_Completion_Rate'] = (df[tp_cols].fillna(0) > 0).mean(axis=1) if tp_cols else 0.0
    df['Respons_Completion_Rate'] = (df[respon_cols].fillna(0) > 0).mean(axis=1) if respon_cols else 0.0
    
    df['Respons_TP_Gap'] = df['Respons_Mean'] - df['TP_Mean']
    
    return df
