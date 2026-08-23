import yaml
import os
import joblib

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(model, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
