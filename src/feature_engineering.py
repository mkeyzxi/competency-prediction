import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Define column groups
    attendance_cols = [f'Attendance_{i}' for i in range(1, 11)]
    tp_cols = [f'TP_{i}' for i in range(1, 7)]
    respon_cols = [f'Respons_{i}' for i in range(1, 7)]
    tp_respons_cols = [f'TP_Respons_{i}' for i in range(1, 8)]
    laporan_cols = [f'Laporan_{i}' for i in range(1, 8)]
    
    # Ensure numeric
    all_raw_cols = attendance_cols + tp_cols + respon_cols + tp_respons_cols + laporan_cols
    for col in all_raw_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    bde_mask = df['Scoring_Scheme'] == 'BDE'
    ac_mask = df['Scoring_Scheme'] == 'AC'
    
    # Pre-Final Attendance Rate
    # AC: 2-7. BDE: 1-8.
    df['Attendance_PreFinal_Rate'] = np.nan
    
    # AC Attendance PreFinal (Mean of Attendance_2 to Attendance_7)
    ac_pre_final_cols = [f'Attendance_{i}' for i in range(2, 8)]
    df.loc[ac_mask, 'Attendance_PreFinal_Rate'] = df.loc[ac_mask, ac_pre_final_cols].fillna(0).mean(axis=1)
    
    # BDE Attendance PreFinal (Mean of Attendance_1 to Attendance_8)
    bde_pre_final_cols = [f'Attendance_{i}' for i in range(1, 9)]
    df.loc[bde_mask, 'Attendance_PreFinal_Rate'] = df.loc[bde_mask, bde_pre_final_cols].fillna(0).mean(axis=1)
    
    # Compute basic means (S1)
    # Fill missing values with 0 before computing mean, because missing = not submitted (score 0)
    
    # Laporan is common for both
    df['Laporan_Mean'] = df[laporan_cols].fillna(0).mean(axis=1)
    
    # Initialize all scheme-specific columns as NaN (Not Applicable)
    for col in tp_cols + respon_cols + tp_respons_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    # AC Specific
    df['TP_Mean'] = np.nan
    df['Respons_Mean'] = np.nan
    df.loc[ac_mask, 'TP_Mean'] = df.loc[ac_mask, tp_cols].fillna(0).mean(axis=1)
    df.loc[ac_mask, 'Respons_Mean'] = df.loc[ac_mask, respon_cols].fillna(0).mean(axis=1)
    
    # BDE Specific
    df['TP_Respons_Mean'] = np.nan
    df.loc[bde_mask, 'TP_Respons_Mean'] = df.loc[bde_mask, tp_respons_cols].fillna(0).mean(axis=1)
    
    # Compute completion rates (S2)
    # Completion = Proportion of non-null and non-zero values (since 0 = not submitted)
    df['Laporan_Completion_Rate'] = (df[laporan_cols].fillna(0) > 0).mean(axis=1)
    
    df['TP_Completion_Rate'] = np.nan
    df['Respons_Completion_Rate'] = np.nan
    df.loc[ac_mask, 'TP_Completion_Rate'] = (df.loc[ac_mask, tp_cols].fillna(0) > 0).mean(axis=1)
    df.loc[ac_mask, 'Respons_Completion_Rate'] = (df.loc[ac_mask, respon_cols].fillna(0) > 0).mean(axis=1)
    
    df['TP_Respons_Completion_Rate'] = np.nan
    df.loc[bde_mask, 'TP_Respons_Completion_Rate'] = (df.loc[bde_mask, tp_respons_cols].fillna(0) > 0).mean(axis=1)
    
    # Compute relational features (S3)
    df['Respons_TP_Gap'] = np.nan
    df.loc[ac_mask, 'Respons_TP_Gap'] = df.loc[ac_mask, 'Respons_Mean'] - df.loc[ac_mask, 'TP_Mean']
    
    return df
