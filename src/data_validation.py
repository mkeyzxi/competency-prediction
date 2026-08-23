import pandas as pd
import json
import os

def validate_data(df: pd.DataFrame, config_path: str = 'configs/data_config.yaml'):
    """
    Validate dataset structure and values.
    Returns a dictionary report.
    """
    report = {}
    
    # 1. Row/Col counts
    report['num_rows'] = int(df.shape[0])
    report['num_cols'] = int(df.shape[1])
    
    # 2. Check missing values
    missing = df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # 3. Check duplicates
    dupes = df.duplicated(subset=['NIM'])
    report['duplicate_nims'] = int(dupes.sum())
    
    # 4. Check label availability
    if 'Final_Individu' in df.columns:
        valid_final = df['Final_Individu'].dropna().apply(lambda x: isinstance(x, (int, float)))
        report['valid_final_individu_count'] = int(valid_final.sum())
    else:
        report['valid_final_individu_count'] = 0
        
    # Save report
    os.makedirs('results/data_quality', exist_ok=True)
    with open('results/data_quality/data_quality_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    # Save as CSV as well
    report_df = pd.DataFrame([{
        "Metric": k,
        "Value": str(v)
    } for k, v in report.items()])
    report_df.to_csv('results/data_quality/data_quality_report.csv', index=False)
    
    return report

if __name__ == "__main__":
    from src.data_loader import load_and_clean_data
    df = load_and_clean_data()
    rep = validate_data(df)
    print("Data Validation Report:")
    print(json.dumps(rep, indent=2))
