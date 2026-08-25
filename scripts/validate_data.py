import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_clean_data
from src.data_validation import validate_data
from src.class_analysis import run_class_analysis
import json

def main():
    print("Loading data...")
    df = load_and_clean_data()
    print(f"Data loaded successfully. Total rows: {len(df)}")
    
    print("\n--- AUDIT MISSING VALUES PER KELAS ---")
    
    for cls in ['A', 'B', 'C', 'D', 'E']:
        cls_mask = df['Class'] == cls
        print(f"\nTotal Siswa {cls}: {cls_mask.sum()}")
        print(f"Missing TP_1 di Kelas {cls}: {df.loc[cls_mask, 'TP_1'].isna().sum()}")
        print(f"Missing Respons_1 di Kelas {cls}: {df.loc[cls_mask, 'Respons_1'].isna().sum()}")
        print(f"Missing Laporan_1 di Kelas {cls}: {df.loc[cls_mask, 'Laporan_1'].isna().sum()}")
        print(f"Missing Final_Individu di Kelas {cls}: {df.loc[cls_mask, 'Final_Individu'].isna().sum()}")
    
    print("\nValidasi selesai.")

if __name__ == "__main__":
    main()
