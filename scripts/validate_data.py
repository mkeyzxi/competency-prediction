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
    
    print("Validating data...")
    report = validate_data(df)
    print("Validation complete. Summary:")
    print(json.dumps(report, indent=2))
    
    
    print("Validation script complete.")
    
if __name__ == "__main__":
    main()
