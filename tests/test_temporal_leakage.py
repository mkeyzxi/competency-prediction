import pandas as pd
import os
import sys

def test_temporal_leakage():
    print("Running Temporal Leakage Check...")
    
    # Read one of the processed feature files
    file_path = 'data/processed/featured_S3.csv'
    if not os.path.exists(file_path):
        print("Data not found.")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    columns = set(df.columns)
    
    # These columns MUST NOT be in the features
    forbidden_cols = [
        'Final_Individu', 'Final_Kelompok', 'nilai flowchart', 'nilai kodingan', 
        'NILAI_AKHIR', 'Predikat', 'Attendance_9', 'Attendance_10', 'Attendance_8'
    ]
    
    leakage_found = False
    for col in forbidden_cols:
        if col in columns:
            print(f"FAIL: Temporal Leakage detected! Column '{col}' is present.")
            leakage_found = True
            
    if not leakage_found:
        print("PASS: No temporal leakage found in feature columns.")
        
    # Also verify that structural NaN behavior is intact in S1/S2/S3
    print("\nRunning Structural Imputation Sanity Check...")
    ac_mask = df['Scoring_Scheme'] == 'AC'
    bde_mask = df['Scoring_Scheme'] == 'BDE'
    
    if df[ac_mask]['TP_Respons_Mean'].notna().sum() > 0:
        print("FAIL: AC has non-NaN values in TP_Respons_Mean")
        leakage_found = True
    else:
        print("PASS: AC correctly uses NaN for TP_Respons_Mean")
        
    if df[bde_mask]['TP_Mean'].notna().sum() > 0:
        print("FAIL: BDE has non-NaN values in TP_Mean")
        leakage_found = True
    else:
        print("PASS: BDE correctly uses NaN for TP_Mean")
        
    if df[bde_mask]['Respons_TP_Gap'].notna().sum() > 0:
        print("FAIL: BDE has non-NaN values in Respons_TP_Gap")
        leakage_found = True
    else:
        print("PASS: BDE correctly uses NaN for Respons_TP_Gap")
        
    if leakage_found:
        sys.exit(1)

if __name__ == "__main__":
    test_temporal_leakage()
