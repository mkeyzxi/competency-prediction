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
    print(results[['scenario', 'model', 'cv_f1_mean', 'test_f1']])

if __name__ == "__main__":
    main()
