FEATURE_REGISTRY = {
    "S1": [
        "Attendance_PreFinal_Rate",
        "TP_Mean",
        "Laporan_Mean",
    ],
    "S2": [
        "Attendance_PreFinal_Rate",
        "TP_Mean",
        "Laporan_Mean",
        "Absence_Count",
        "TP_Completion_Rate",
        "Laporan_Completion_Rate",
    ],
    "S3": [
        "Attendance_PreFinal_Rate",
        "TP_Mean",
        "Laporan_Mean",
        "TP_Completion_Rate",
        "Laporan_Completion_Rate",
        "Performance_Std",
    ],
    "S4": [
        "Attendance_PreFinal_Rate", "TP_Mean", "Laporan_Mean",
        "Absence_Count", "TP_Completion_Rate", "Laporan_Completion_Rate",
        "TP_Std", "Laporan_Std", "Performance_Std",
        "TP_First2_Mean", "Laporan_First2_Mean",
        "TP_Last2_Mean", "Laporan_Last2_Mean",
        "TP_Trend", "Laporan_Trend",
    ],
    "S5": [
        "Attendance_PreFinal_Rate", "TP_Mean", "Laporan_Mean",
        "Absence_Count", "TP_Completion_Rate", "Laporan_Completion_Rate",
        "TP_Std", "Laporan_Std", "Performance_Std",
        "TP_First2_Mean", "Laporan_First2_Mean",
        "TP_Last2_Mean", "Laporan_Last2_Mean",
        "TP_Trend", "Laporan_Trend",
        "TP_Min", "TP_Max", "Laporan_Min", "Laporan_Max",
        "Performance_Late_Mean",
    ],
    # ============================
    # INCREMENTAL EWS EXPERIMENTS
    # S3 + 1 feature at a time
    # ============================
    "S3_A": [
        # S3 + Absence_Streak_Max
        "Attendance_PreFinal_Rate",
        "TP_Mean", "Laporan_Mean",
        "TP_Completion_Rate", "Laporan_Completion_Rate",
        "Performance_Std",
        "Absence_Streak_Max",
    ],
    "S3_B": [
        # S3 + Performance_Decline_Flag
        "Attendance_PreFinal_Rate",
        "TP_Mean", "Laporan_Mean",
        "TP_Completion_Rate", "Laporan_Completion_Rate",
        "Performance_Std",
        "Performance_Decline_Flag",
    ],
    "S3_C": [
        # S3 + Zero_Score_Count
        "Attendance_PreFinal_Rate",
        "TP_Mean", "Laporan_Mean",
        "TP_Completion_Rate", "Laporan_Completion_Rate",
        "Performance_Std",
        "Zero_Score_Count",
    ],
    "S3_D": [
        # S3 + Cumulative_Completion_Trend
        "Attendance_PreFinal_Rate",
        "TP_Mean", "Laporan_Mean",
        "TP_Completion_Rate", "Laporan_Completion_Rate",
        "Performance_Std",
        "Cumulative_Completion_Trend",
    ],
    "S3_E": [
        # S3 + Early_Performance_Composite
        "Attendance_PreFinal_Rate",
        "TP_Mean", "Laporan_Mean",
        "TP_Completion_Rate", "Laporan_Completion_Rate",
        "Performance_Std",
        "Early_Performance_Composite",
    ],
    "S3_EWS": [
        # S3 + best combination (to be determined after incremental testing)
        "Attendance_PreFinal_Rate",
        "TP_Mean", "Laporan_Mean",
        "TP_Completion_Rate", "Laporan_Completion_Rate",
        "Performance_Std",
        "Absence_Streak_Max",
        "Performance_Decline_Flag",
        "Zero_Score_Count",
        "Cumulative_Completion_Trend",
        "Early_Performance_Composite",
    ],
}

def get_features(scenario: str):
    return FEATURE_REGISTRY.get(scenario, [])

