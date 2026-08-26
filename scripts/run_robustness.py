"""
Eksperimen Robustness: Menguji konfigurasi terbaik dari eksperimen utama
melintasi populasi P0, P1, dan P2 untuk membuktikan generalisasi.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.experiments import run_all_experiments

def main():
    print("=" * 60)
    print("EKSPERIMEN ROBUSTNESS — P0, P1, P2")
    print("=" * 60)
    
    populations = ['P0', 'P1', 'P2']
    all_results = []
    
    # Clean old robustness files
    robustness_file = 'results/reports/robustness_results.csv'
    if os.path.exists(robustness_file):
        os.remove(robustness_file)
    
    # Clean appended files
    for f in ['results/reports/class_distribution.csv',
              'results/reports/error_analysis.csv',
              'results/reports/threshold_analysis.csv']:
        if os.path.exists(f):
            os.remove(f)
    
    for pop in populations:
        file_path = f'data/processed/featured_{pop}_full.csv'
        if not os.path.exists(file_path):
            print(f"Featured dataset {pop} not found. Skipping.")
            continue
            
        print(f"\n--- Evaluating Population {pop} ---")
        df = pd.read_csv(file_path)
        results = run_all_experiments(df, pop)
        all_results.append(results)
        
    if not all_results:
        print("No experiments were run.")
        return
        
    final_results = pd.concat(all_results, ignore_index=True)
    
    os.makedirs('results/reports', exist_ok=True)
    final_results.to_csv(robustness_file, index=False)
    
    print("\n" + "=" * 60)
    print("SUMMARY — Robustness Results")
    print("=" * 60)
    
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
    
    # Show only DT and RF for readability (plus Dummy for baseline)
    tree_results = final_results[final_results['model'].isin(['Dummy', 'DecisionTree', 'RandomForest'])]
    print(tree_results[available_metrics].to_string(index=False))
    
    # Cross-population stability check
    print("\n--- Cross-Population Stability (Best DT/RF per scenario) ---")
    best_per_pop = final_results[
        final_results['model'].isin(['DecisionTree', 'RandomForest'])
    ].groupby(['population', 'scenario', 'model']).agg({
        'cv_accuracy_mean': 'first',
        'test_accuracy': 'first',
        'test_balanced_accuracy': 'first',
    }).reset_index()
    print(best_per_pop.to_string(index=False))

if __name__ == "__main__":
    main()
