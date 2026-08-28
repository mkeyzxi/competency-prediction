import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_features():
    df_act = pd.read_csv('data/processed/activities_long.csv')
    df_students = pd.read_csv('data/processed/students_master.csv')
    
    cutoffs = {
        'C1': {'Attendance': 2, 'Laporan': 1, 'TP': 1},
        'C2': {'Attendance': 4, 'Laporan': 2, 'TP': 2},
        'C3': {'Attendance': 5, 'Laporan': 3, 'TP': 3},
        'C_Full': {'Attendance': 7, 'Laporan': 4, 'TP': 4}
    }
    
    for c_name, c_limits in cutoffs.items():
        df_c = df_act.copy()
        mask = (
            ((df_c['Activity_Type'] == 'Attendance') & (df_c['Time_Index'] <= c_limits['Attendance'])) |
            ((df_c['Activity_Type'] == 'Laporan') & (df_c['Time_Index'] <= c_limits['Laporan'])) |
            ((df_c['Activity_Type'] == 'TP') & (df_c['Time_Index'] <= c_limits['TP']))
        )
        df_c = df_c[mask]
        
        features = []
        for _, student in df_students.iterrows():
            nama = student['Nama']
            st_data = df_c[df_c['Nama'] == nama]
            
            att = st_data[st_data['Activity_Type'] == 'Attendance']
            lap = st_data[st_data['Activity_Type'] == 'Laporan']
            tp = st_data[st_data['Activity_Type'] == 'TP']
            
            feat = {
                'Nama': nama, 
                'Kelas': student['Kelas'], 
                'Competency_Label': student['Competency_Label'],
                'Final_Score': student['Final_Score'],
                'Final_Status': student['Final_Status'],
                'Label_Reason': student['Label_Reason']
            }
            
            def safe_mean(df, col='Score'): return df[col].mean() if not df.empty else np.nan
            def safe_std(df, col='Score'): return df[col].std(ddof=0) if len(df) > 1 else (0.0 if len(df) == 1 else np.nan)
            def safe_min(df, col='Score'): return df[col].min() if not df.empty else np.nan
            def safe_max(df, col='Score'): return df[col].max() if not df.empty else np.nan
            def completion(df, max_avail): return len(df) / max_avail if max_avail > 0 else np.nan
            def trend(df): 
                if len(df) > 1:
                    return np.polyfit(range(len(df)), df['Score'], 1)[0]
                return 0.0
                
            # S1
            feat['Attendance_PreFinal_Rate'] = completion(att, c_limits['Attendance'])
            feat['TP_Mean'] = safe_mean(tp)
            feat['Laporan_Mean'] = safe_mean(lap)
            
            # S2
            feat['Absence_Count'] = len(att[att['Score'] == 0]) if not att.empty else np.nan
            feat['TP_Completion_Rate'] = completion(tp, c_limits['TP'])
            feat['Laporan_Completion_Rate'] = completion(lap, c_limits['Laporan'])
            
            # S3
            feat['TP_Std'] = safe_std(tp)
            feat['Laporan_Std'] = safe_std(lap)
            perf = pd.concat([tp, lap])
            feat['Performance_Std'] = safe_std(perf)
            
            # S4
            feat['TP_First2_Mean'] = safe_mean(tp.head(2))
            feat['Laporan_First2_Mean'] = safe_mean(lap.head(2))
            feat['TP_Last2_Mean'] = safe_mean(tp.tail(2))
            feat['Laporan_Last2_Mean'] = safe_mean(lap.tail(2))
            feat['TP_Trend'] = trend(tp)
            feat['Laporan_Trend'] = trend(lap)
            
            # S5
            feat['TP_Min'] = safe_min(tp)
            feat['TP_Max'] = safe_max(tp)
            feat['Laporan_Min'] = safe_min(lap)
            feat['Laporan_Max'] = safe_max(lap)
            feat['Performance_Late_Mean'] = safe_mean(perf.tail(4))
            
            # EWS Incremental Features
            def max_absence_streak(att_df):
                streak, max_streak = 0, 0
                for score in att_df['Score']:
                    if score == 0:
                        streak += 1
                        max_streak = max(max_streak, streak)
                    else:
                        streak = 0
                return max_streak
                
            feat['Absence_Streak_Max'] = max_absence_streak(att)
            
            if len(perf) >= 4:
                overall_std = safe_std(perf)
                early_mean = safe_mean(perf.head(len(perf)//2))
                late_mean = safe_mean(perf.tail(len(perf) - len(perf)//2))
                decline = early_mean - late_mean
                feat['Performance_Decline_Flag'] = 1 if (overall_std > 0 and decline > overall_std) else 0
            else:
                feat['Performance_Decline_Flag'] = 0
                
            feat['Zero_Score_Count'] = len(perf[perf['Score'] == 0])
            
            if len(perf) >= 4:
                mid = len(perf) // 2
                first_half = perf.head(mid)
                second_half = perf.tail(len(perf) - mid)
                fh_comp = len(first_half[first_half['Score'] > 0]) / len(first_half)
                sh_comp = len(second_half[second_half['Score'] > 0]) / len(second_half)
                feat['Cumulative_Completion_Trend'] = sh_comp - fh_comp
            else:
                feat['Cumulative_Completion_Trend'] = 0.0
                
            early_perf = pd.concat([tp.head(2), lap.head(2)])
            feat['Early_Performance_Composite'] = safe_mean(early_perf)
            
            features.append(feat)
            
        df_feat = pd.DataFrame(features)
        
        from src.feature_registry import get_features
        base_cols = ['Nama', 'Kelas', 'Competency_Label', 'Final_Score', 'Final_Status', 'Label_Reason']
        scenarios_to_export = ['S1', 'S2', 'S3', 'S4', 'S5', 'S3_A', 'S3_B', 'S3_C', 'S3_D', 'S3_E', 'S3_EWS']
        
        for sc in scenarios_to_export:
            feats = get_features(sc)
            cols = base_cols + feats
            cols = [c for c in cols if c in df_feat.columns]
            df_feat[cols].to_csv(f'data/features/{c_name}_{sc}.csv', index=False)
        
    print("Features generated successfully (V2 - Incremental).")

if __name__ == '__main__':
    create_features()
