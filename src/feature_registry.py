FEATURE_REGISTRY = {
    "S1": [
        "Attendance_PreFinal_Rate",
        "TP_Mean",
        "Respons_Mean",
        "Laporan_Mean",
    ],
    "S2": [
        "Attendance_PreFinal_Rate",
        "TP_Mean",
        "Respons_Mean",
        "Laporan_Mean",
        "TP_Completion_Rate",
        "Respons_Completion_Rate",
        "Laporan_Completion_Rate",
    ],
    "S3": [
        "Attendance_PreFinal_Rate",
        "TP_Mean",
        "Respons_Mean",
        "Laporan_Mean",
        "TP_Completion_Rate",
        "Respons_Completion_Rate",
        "Laporan_Completion_Rate",
        "Performance_Volatility",
    ],
    "S4": [
        # Basic
        "Attendance_PreFinal_Rate",
        "TP_Mean",
        "Respons_Mean",
        "Laporan_Mean",
        # Completion
        "TP_Completion_Rate",
        "Respons_Completion_Rate",
        "Laporan_Completion_Rate",
        # Volatility
        "Performance_Volatility",
        # Statistical
        "TP_Std",
        "Respons_Std",
        "Laporan_Std",
        "TP_Min",
        "TP_Max",
        "Respons_Min",
        "Respons_Max",
        "Laporan_Min",
        "Laporan_Max",
        # Temporal
        "TP_Trend",
        "Respons_Trend",
        "Laporan_Trend",
        # Recent Performance
        "TP_Last2_Mean",
        "Respons_Last2_Mean",
        "Laporan_Last2_Mean",
        # Late vs Early Gap
        "TP_LateEarly_Gap",
        "Respons_LateEarly_Gap",
        "Laporan_LateEarly_Gap",
        # Composite
        "Activity_Score",
    ],
    "S5": [
        # ===== Per-Category Statistics =====
        # TP
        "TP_Mean",
        "TP_Std",
        "TP_Min",
        "TP_Max",
        "TP_Last2_Mean",
        "TP_First2_Mean",
        "TP_Trend",
        # Respons
        "Respons_Mean",
        "Respons_Std",
        "Respons_Min",
        "Respons_Max",
        "Respons_Last2_Mean",
        "Respons_Trend",
        # Laporan
        "Laporan_Mean",
        "Laporan_Std",
        "Laporan_Min",
        "Laporan_Max",
        "Laporan_Last2_Mean",
        "Laporan_Trend",
        # ===== Attendance =====
        "Attendance_PreFinal_Rate",
        "Absence_Count",
        "Partial_Attendance_Count",
        # ===== Completion =====
        "TP_Completion_Rate",
        "Respons_Completion_Rate",
        "Laporan_Completion_Rate",
        # ===== Global Performance (cross-category) =====
        "Performance_Mean",
        "Performance_Std",
        "Performance_Late_Mean",
        "Performance_Trend",
    ],
    "S6": [
        # Domain-Oriented Features
        "Laporan_Improvement",
        "Respons_Improvement",
        "TP_Improvement",
        "Laporan_EarlyMean",
        "Laporan_LateMean",
        "Respons_EarlyMean",
        "Respons_LateMean",
        "TP_EarlyMean",
        "TP_LateMean",
        "Overall_Consistency",
        "Overall_Recent_Mean",
        "Overall_Trend",
        # Core performance
        "Performance_Mean",
        "Laporan_Mean",
        "Respons_Mean",
        "TP_Mean",
        "Laporan_Std",
        "Respons_Std",
        "TP_Std",
        "Laporan_Completion_Rate",
        "Attendance_PreFinal_Rate"
    ]
}

def get_features(scenario: str):
    return FEATURE_REGISTRY.get(scenario, [])
