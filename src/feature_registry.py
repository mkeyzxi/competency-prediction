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
        "Respons_TP_Gap",
    ]
}

def get_features(scenario: str):
    return FEATURE_REGISTRY.get(scenario, [])
