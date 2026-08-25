from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import load_config

def get_model(model_name: str, config_path: str = 'configs/experiment_config.yaml'):
    """
    Returns a pipeline with an imputer and the model.
    """
    config = load_config(config_path)
    random_state = config.get('random_state', 42)
    
    # User requested: SimpleImputer with -1 and add_indicator=True
    imputer = SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)
    
    if model_name == 'DecisionTree':
        clf = DecisionTreeClassifier(random_state=random_state)
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown model {model_name}")
        
    return Pipeline([
        ('imputer', imputer),
        ('model', clf)
    ])

def get_param_grid(model_name: str):
    """
    Returns reasonable hyperparameter grids for tuning.
    Prefixed with 'model__' because of the Pipeline.
    """
    if model_name == 'DecisionTree':
        return {
            'model__max_depth': [None, 3, 5, 7, 10],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__class_weight': [None, 'balanced']
        }
    elif model_name == 'RandomForest':
        return {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 5, 10],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2],
            'model__class_weight': [None, 'balanced', 'balanced_subsample']
        }
    return {}
