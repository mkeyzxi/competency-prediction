import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.labeling import create_labels
from src.feature_engineering import compute_features
from src.feature_registry import get_features

def main():
    print("Building features...")
    # Read interim data
    if not os.path.exists('data/interim/combined_data.csv'):
        print("Interim data not found. Please run validate_data.py first.")
        return
        
    df = pd.read_csv('data/interim/combined_data.csv')
    
    # 1. Labeling
    df_labeled = create_labels(df)
    
    # 2. Feature Engineering
    df_featured = compute_features(df_labeled)
    
    # Run class analysis
    from src.class_analysis import run_class_analysis
    run_class_analysis(df_featured)
    
    # 3. Save feature sets based on scenarios
    os.makedirs('data/processed', exist_ok=True)
    
    scenarios = ['S1', 'S2', 'S3']
    for scenario in scenarios:
        features = get_features(scenario)
        if features:
            # We must keep Competency_Label, Competency_Name, NIM, Kelas for modeling/tracking
            # Actually, split.py expects Competency_Label and metadata
            keep_cols = ['Competency_Label', 'Competency_Name', 'NIM', 'Kelas']
            # Only select required columns
            cols_to_save = features + [c for c in keep_cols if c in df_featured.columns]
            df_scenario = df_featured[cols_to_save]
            
            out_path = f'data/processed/featured_{scenario}.csv'
            df_scenario.to_csv(out_path, index=False)
            print(f"Saved {scenario} dataset to {out_path} with shape {df_scenario.shape}")
            
    # Also save the full dataset for easy loading in experiments
    df_featured.to_csv('data/processed/featured_full.csv', index=False)
    print("Features built successfully.")

if __name__ == "__main__":
    main()
