import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.experiments import run_all_experiments

def main():
    print("Running comprehensive experiments...")
    
    populations = ['P0', 'P1', 'P2']
    all_results = []
    
    # Bersihkan file report lama agar tidak terjadi duplikasi append
    old_files = [
        'results/reports/class_distribution.csv',
        'results/reports/error_analysis.csv',
        'results/reports/threshold_analysis.csv',
        'results/reports/baseline_comparison.csv',
        'results/reports/repeated_cv_results.csv'
    ]
    for f in old_files:
        if os.path.exists(f):
            os.remove(f)
            
    for pop in populations:
        file_path = f'data/processed/featured_{pop}_full.csv'
        if not os.path.exists(file_path):
            print(f"Featured dataset {pop} not found. Please run build_features.py first.")
            continue
            
        print(f"\n--- Evaluating Population {pop} ---")
        df = pd.read_csv(file_path)
        results = run_all_experiments(df, pop)
        all_results.append(results)
        
    if not all_results:
        print("No experiments were run.")
        return
        
    final_results = pd.concat(all_results, ignore_index=True)
    
    # 1. Simpan Repeated CV Results
    final_results.to_csv('results/reports/repeated_cv_results.csv', index=False)
    
    # 2. Pisahkan Baseline Comparison (hanya Dummy & LogReg)
    baseline_df = final_results[final_results['model'].isin(['Dummy', 'LogisticRegression'])]
    baseline_df.to_csv('results/reports/baseline_comparison.csv', index=False)
    
    # 3. Model Comparison Utama
    final_results.to_csv('results/metrics/model_comparison.csv', index=False)
    
    print("\nExperiments completed. Summary of results:")
    
    metrics = [
        'population', 'scenario', 'model', 'best_threshold',
        'cv_accuracy_mean', 'test_accuracy',
        'cv_balanced_accuracy_mean', 'test_balanced_accuracy',
        'cv_f1_mean', 'test_f1',
        'test_recall_belum_kompeten'
    ]
    
    available_metrics = [m for m in metrics if m in final_results.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(final_results[available_metrics].to_string(index=False))

if __name__ == "__main__":
    main()
