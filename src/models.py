from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.utils import load_config

def get_model(model_name: str, config_path: str = 'configs/experiment_config.yaml'):
    """
    Returns a pipeline with an imputer, optional scaler/smote, and the model.
    """
    config = load_config(config_path)
    random_state = config.get('random_state', 42)
    
    # User requested: SimpleImputer with -1 and add_indicator=True
    imputer = SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)
    
    if model_name == 'DecisionTree':
        clf = DecisionTreeClassifier(random_state=random_state)
        scaler = 'passthrough'
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(random_state=random_state)
        scaler = 'passthrough'
    elif model_name == 'Dummy':
        clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
        scaler = 'passthrough'
    elif model_name == 'LogisticRegression':
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown model {model_name}")
        
    steps = [
        ('imputer', imputer),
        ('scaler', scaler)
    ]
    
    if model_name != 'Dummy':
        steps.append(('smote', SMOTE(random_state=random_state)))
        
    steps.append(('model', clf))
    
    return Pipeline(steps)

def get_param_grid(model_name: str):
    """
    Returns reasonable hyperparameter grids for tuning.
    Prefixed with 'model__' because of the Pipeline.
    """
    if model_name == 'DecisionTree':
        return {
            'model__criterion': ['gini', 'entropy', 'log_loss'],
            'model__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'model__min_samples_split': [2, 4, 6, 8, 10],
            'model__min_samples_leaf': [1, 2, 3, 4, 5],
            'model__max_features': ['sqrt', 'log2', None],
            'model__ccp_alpha': [0.0, 0.005, 0.01, 0.015, 0.02],
            'model__class_weight': [None],
        }
    elif model_name == 'RandomForest':
        return {
            'model__n_estimators': [200, 500, 800],
            'model__max_depth': [None, 3, 4, 5, 6, 8],
            'model__min_samples_split': [2, 4, 6, 8, 10],
            'model__min_samples_leaf': [1, 2, 3, 4, 5],
            'model__max_features': ['sqrt', 'log2', 0.5, 0.7, 1.0],
            'model__criterion': ['gini', 'entropy', 'log_loss'],
            'model__class_weight': [None],
        }
    elif model_name == 'LogisticRegression':
        return {
            'model__C': [0.01, 0.1, 1.0, 10.0],
            'model__class_weight': [None, 'balanced']
        }
    return {}
