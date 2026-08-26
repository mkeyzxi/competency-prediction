import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.feature_engineering import compute_features
from src.feature_registry import get_features

def main():
    print("Building features...")
    # Read eligible data from preprocessing
    if not os.path.exists('data/processed/eligible.csv'):
        print("Eligible data not found. Please run src/preprocessing.py first.")
        return
        
    df = pd.read_csv('data/processed/eligible.csv')
    
    # Feature Engineering
    df_featured = compute_features(df)
    
    os.makedirs('data/processed', exist_ok=True)
    
    populations = ['P0', 'P1', 'P2']
    scenarios = ['S1', 'S2', 'S3', 'S4']
    
    for pop in populations:
        df_pop = pd.read_csv(f'data/processed/population_{pop}.csv')
        df_pop_featured = compute_features(df_pop)
        
        for scenario in scenarios:
            features = get_features(scenario)
            if features:
                # Keep metadata for evaluation and context analysis
                keep_cols = ['Competency_Label', 'Competency_Name', 'NIM', 'Class', 'Scoring_Scheme']
                cols_to_save = features + [c for c in keep_cols if c in df_pop_featured.columns]
                df_scenario = df_pop_featured[cols_to_save]
                
                out_path = f'data/processed/featured_{pop}_{scenario}.csv'
                df_scenario.to_csv(out_path, index=False)
                print(f"Saved {pop} {scenario} dataset to {out_path} with shape {df_scenario.shape}")
                
        # Save the full dataset for easy loading in experiments
        df_pop_featured.to_csv(f'data/processed/featured_{pop}_full.csv', index=False)
        
    print("Features built successfully.")

if __name__ == "__main__":
    main()
