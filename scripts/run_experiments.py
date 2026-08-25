import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.experiments import run_all_experiments

def main():
    print("Running experiments...")
    
    if not os.path.exists('data/processed/featured_full.csv'):
        print("Featured dataset not found. Please run build_features.py first.")
        return
        
    df = pd.read_csv('data/processed/featured_full.csv')
    
    results = run_all_experiments(df)
    
    print("Experiments completed. Summary of results:")
    
    # Pilih metrik yang komprehensif untuk ditampilkan
    metrics = [
        'scenario', 'model', 
        'cv_accuracy_mean', 'test_accuracy',
        'cv_precision_mean', 'test_precision',
        'cv_recall_mean', 'test_recall',
        'cv_f1_mean', 'test_f1',
        'cv_specificity_mean', 'test_specificity'
    ]
    
    # Filter hanya kolom yang benar-benar ada (berjaga-jaga jika ada versi lama)
    available_metrics = [m for m in metrics if m in results.columns]
    
    # Cetak tabel lengkap (bisa diatur agar semua kolom terlihat)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results[available_metrics].to_string(index=False))

if __name__ == "__main__":
    main()
