from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils import load_config

def get_model(model_name: str, config_path: str = 'configs/experiment_config.yaml'):
    """
    Returns an uninstantiated model class or an instance with default params.
    """
    config = load_config(config_path)
    random_state = config.get('random_state', 42)
    
    if model_name == 'DecisionTree':
        return DecisionTreeClassifier(random_state=random_state)
    elif model_name == 'RandomForest':
        return RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown model {model_name}")

def get_param_grid(model_name: str):
    """
    Returns reasonable hyperparameter grids for tuning.
    """
    if model_name == 'DecisionTree':
        return {
            'max_depth': [None, 3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
    elif model_name == 'RandomForest':
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': [None, 'balanced']
        }
    return {}
