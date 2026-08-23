import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Define column groups based on actual excel shape
    kehadiran_cols = [f'Kehadiran_{i}' for i in range(1, 11)]
    tp_cols = [f'TP_{i}' for i in range(1, 7)]
    respon_cols = [f'Respon_{i}' for i in range(1, 7)]
    laporan_cols = [f'Laporan_{i}' for i in range(1, 8)]
    
    # Ensure numeric
    for col in kehadiran_cols + tp_cols + respon_cols + laporan_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute basic means (S1)
    # Fill missing values with 0 before computing mean, because missing = not submitted (score 0)
    df['Attendance_Rate'] = df[kehadiran_cols].fillna(0).mean(axis=1)
    
    df['TP_Mean'] = df[tp_cols].fillna(0).mean(axis=1)
    df['Respons_Mean'] = df[respon_cols].fillna(0).mean(axis=1)
    df['Laporan_Mean'] = df[laporan_cols].fillna(0).mean(axis=1)
    
    # Compute completion rates (S2)
    # Proportion of non-null values
    df['TP_Completion_Rate'] = df[tp_cols].notnull().mean(axis=1)
    df['Respons_Completion_Rate'] = df[respon_cols].notnull().mean(axis=1)
    df['Laporan_Completion_Rate'] = df[laporan_cols].notnull().mean(axis=1)
    
    # Compute relational features (S3)
    df['Respons_TP_Gap'] = df['Respons_Mean'] - df['TP_Mean']
    
    return df
