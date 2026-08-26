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
            
    # ================================================================
    # S1: BASIC FEATURES (Mean & Attendance Rate)
    # ================================================================
    df['Attendance_PreFinal_Rate'] = df[pre_final_cols].fillna(0).mean(axis=1) if pre_final_cols else 0.0
    
    df['TP_Response_Source'] = 'SEPARATE'
    
    df['Laporan_Mean'] = df[laporan_cols].fillna(0).mean(axis=1) if laporan_cols else 0.0
    df['TP_Mean'] = df[tp_cols].fillna(0).mean(axis=1) if tp_cols else 0.0
    df['Respons_Mean'] = df[respon_cols].fillna(0).mean(axis=1) if respon_cols else 0.0
    
    # ================================================================
    # S2: COMPLETION RATES
    # ================================================================
    df['Laporan_Completion_Rate'] = (df[laporan_cols].fillna(0) > 0).mean(axis=1) if laporan_cols else 0.0
    df['TP_Completion_Rate'] = (df[tp_cols].fillna(0) > 0).mean(axis=1) if tp_cols else 0.0
    df['Respons_Completion_Rate'] = (df[respon_cols].fillna(0) > 0).mean(axis=1) if respon_cols else 0.0
    
    # Keep Respons_TP_Gap for backward compatibility (not used in new S3)
    df['Respons_TP_Gap'] = df['Respons_Mean'] - df['TP_Mean']
    
    # ================================================================
    # S3: PERFORMANCE VOLATILITY (universal, replaces Respons_TP_Gap in registry)
    # ================================================================
    # Combine all score columns into one array per student and compute std.
    # This is independent of whether TP and Respons are separate or combined.
    all_score_cols = [c for c in tp_cols + respon_cols + laporan_cols if c in df.columns]
    if all_score_cols:
        df['Performance_Volatility'] = df[all_score_cols].fillna(0).std(axis=1, ddof=0)
    else:
        df['Performance_Volatility'] = 0.0
    
    # ================================================================
    # S4: TEMPORAL / STATISTICAL FEATURES
    # ================================================================
    
    # --- Standard Deviation, Min, Max ---
    def compute_stats(cols, prefix):
        """Compute _Std, _Min, _Max for a group of score columns."""
        if not cols:
            df[f'{prefix}_Std'] = 0.0
            df[f'{prefix}_Min'] = 0.0
            df[f'{prefix}_Max'] = 0.0
            return
        filled = df[cols].fillna(0)
        df[f'{prefix}_Std'] = filled.std(axis=1, ddof=0)
        df[f'{prefix}_Min'] = filled.min(axis=1)
        df[f'{prefix}_Max'] = filled.max(axis=1)
    
    compute_stats(tp_cols, 'TP')
    compute_stats(respon_cols, 'Respons')
    compute_stats(laporan_cols, 'Laporan')
    
    # --- Recent Performance (Last 2 sessions) ---
    def compute_last2_mean(cols, prefix):
        """Mean of the last 2 available sessions."""
        if not cols or len(cols) < 2:
            df[f'{prefix}_Last2_Mean'] = df[cols].fillna(0).mean(axis=1) if cols else 0.0
            return
        last2 = cols[-2:]
        df[f'{prefix}_Last2_Mean'] = df[last2].fillna(0).mean(axis=1)
    
    compute_last2_mean(tp_cols, 'TP')
    compute_last2_mean(respon_cols, 'Respons')
    compute_last2_mean(laporan_cols, 'Laporan')
    
    # --- Trend (Second Half Mean - First Half Mean) ---
    def compute_trend(cols, prefix):
        if not cols or len(cols) < 2:
            df[f'{prefix}_First_Half_Mean'] = 0.0
            df[f'{prefix}_Second_Half_Mean'] = 0.0
            df[f'{prefix}_Trend'] = 0.0
            return
        
        mid = len(cols) // 2
        first_half = cols[:mid]
        second_half = cols[mid:]
        
        df[f'{prefix}_First_Half_Mean'] = df[first_half].fillna(0).mean(axis=1)
        df[f'{prefix}_Second_Half_Mean'] = df[second_half].fillna(0).mean(axis=1)
        df[f'{prefix}_Trend'] = df[f'{prefix}_Second_Half_Mean'] - df[f'{prefix}_First_Half_Mean']
        
    compute_trend(tp_cols, 'TP')
    compute_trend(respon_cols, 'Respons')
    compute_trend(laporan_cols, 'Laporan')
    
    # --- Late vs Early Gap (same as Trend but named explicitly) ---
    df['TP_LateEarly_Gap'] = df['TP_Trend']
    df['Respons_LateEarly_Gap'] = df['Respons_Trend']
    df['Laporan_LateEarly_Gap'] = df['Laporan_Trend']
    
    # --- Activity Score (composite) ---
    df['Activity_Score'] = (
        0.25 * df['TP_Mean'] + 
        0.25 * df['Respons_Mean'] + 
        0.25 * df['Laporan_Mean'] + 
        0.25 * df['Attendance_PreFinal_Rate'] * 100
    )
    
    # ================================================================
    # S5: TREE-SPECIFIC FEATURES
    # ================================================================
    
    # --- First 2 sessions mean (complement to Last2_Mean) ---
    def compute_first2_mean(cols, prefix):
        """Mean of the first 2 available sessions."""
        if not cols or len(cols) < 2:
            df[f'{prefix}_First2_Mean'] = df[cols].fillna(0).mean(axis=1) if cols else 0.0
            return
        first2 = cols[:2]
        df[f'{prefix}_First2_Mean'] = df[first2].fillna(0).mean(axis=1)
    
    compute_first2_mean(tp_cols, 'TP')
    compute_first2_mean(respon_cols, 'Respons')
    compute_first2_mean(laporan_cols, 'Laporan')
    
    # --- Attendance granularity ---
    att_filled = df[pre_final_cols].fillna(0) if pre_final_cols else pd.DataFrame()
    if not att_filled.empty:
        df['Absence_Count'] = (att_filled == 0).sum(axis=1)
        df['Partial_Attendance_Count'] = ((att_filled > 0) & (att_filled < 1)).sum(axis=1)
    else:
        df['Absence_Count'] = 0
        df['Partial_Attendance_Count'] = 0
    
    # --- Global Performance aggregates (across ALL task types) ---
    if all_score_cols:
        all_scores_filled = df[all_score_cols].fillna(0)
        df['Performance_Mean'] = all_scores_filled.mean(axis=1)
        df['Performance_Std'] = all_scores_filled.std(axis=1, ddof=0)
        
        # Late performance: mean of last 2 cols of each category, averaged
        late_cols = []
        for cols_group in [tp_cols, respon_cols, laporan_cols]:
            if len(cols_group) >= 2:
                late_cols.extend(cols_group[-2:])
            elif cols_group:
                late_cols.extend(cols_group)
        if late_cols:
            df['Performance_Late_Mean'] = df[late_cols].fillna(0).mean(axis=1)
        else:
            df['Performance_Late_Mean'] = 0.0
        
        # Overall performance trend: late mean - early mean
        early_cols = []
        for cols_group in [tp_cols, respon_cols, laporan_cols]:
            if len(cols_group) >= 2:
                early_cols.extend(cols_group[:2])
            elif cols_group:
                early_cols.extend(cols_group)
        if early_cols:
            df['Performance_Early_Mean'] = df[early_cols].fillna(0).mean(axis=1)
        else:
            df['Performance_Early_Mean'] = 0.0
        
        df['Performance_Trend'] = df['Performance_Late_Mean'] - df['Performance_Early_Mean']
        
    else:
        df['Performance_Mean'] = 0.0
        df['Performance_Std'] = 0.0
        df['Performance_Late_Mean'] = 0.0
        df['Performance_Early_Mean'] = 0.0
        df['Performance_Trend'] = 0.0
        
    # ================================================================
    # S6: DOMAIN-ORIENTED FEATURES (Improvement, Consistency, etc.)
    # ================================================================
    
    # 1. Improvement (Late vs Early difference explicitly named)
    df['Laporan_Improvement'] = df['Laporan_Trend']
    df['Respons_Improvement'] = df['Respons_Trend']
    df['TP_Improvement'] = df['TP_Trend']
    
    # 2. Early vs Late Means
    df['Laporan_EarlyMean'] = df['Laporan_First_Half_Mean']
    df['Laporan_LateMean'] = df['Laporan_Second_Half_Mean']
    
    df['Respons_EarlyMean'] = df['Respons_First_Half_Mean']
    df['Respons_LateMean'] = df['Respons_Second_Half_Mean']

    df['TP_EarlyMean'] = df['TP_First_Half_Mean']
    df['TP_LateMean'] = df['TP_Second_Half_Mean']
    
    # 3. Overall features
    # Consistency is the inverse of volatility (std)
    # Using negative std as consistency score, or 1 / (std + 1)
    df['Overall_Consistency'] = 1.0 / (df['Performance_Std'] + 1.0)
    
    df['Overall_Recent_Mean'] = df['Performance_Late_Mean']
    df['Overall_Trend'] = df['Performance_Trend']
    
    return df
