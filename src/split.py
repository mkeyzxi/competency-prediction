import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_config

def get_train_test_split(df: pd.DataFrame, config_path: str = 'configs/experiment_config.yaml'):
    """
    Performs train-test split according to config.
    Returns X_train, X_test, y_train, y_test
    """
    config = load_config(config_path)
    test_size = config['split']['test_size']
    random_state = config['random_state']
    stratify_flag = config['split']['stratify']
    
    # We assume 'Competency_Label' is our target
    # and we drop non-feature columns
    metadata_cols = ['No', 'NIM', 'Nama', 'Kelas', 'Competency_Name']
    # Target col
    y_col = 'Competency_Label'
    
    # Drop metadata and target from X
    X = df.drop(columns=[y_col] + [c for c in metadata_cols if c in df.columns], errors='ignore')
    
    # Exclude raw activity columns since we only use S1/S2/S3 features
    # But wait, feature filtering will be done at the experiment level based on scenario.
    # So X here has all features. The experiment script will filter columns.
    
    y = df[y_col]
    
    stratify = y if stratify_flag else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test
