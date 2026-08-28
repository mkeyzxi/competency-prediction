from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from src.utils import load_config


def get_model(model_name: str, balancing: str = 'SMOTE',
              config_path: str = 'configs/experiment_config.yaml'):
    """
    Returns a pipeline with imputer, optional SMOTE, and the model.

    Parameters
    ----------
    model_name : str
        One of 'Dummy', 'DecisionTree', 'RandomForest', 'GradientBoosting'.
    balancing : str
        'SMOTE'        – apply SMOTE in the pipeline (no class_weight).
        'ClassWeight'  – use class_weight='balanced' on the classifier (no SMOTE).
        'None'         – no balancing at all.
    """
    config = load_config(config_path)
    random_state = config.get('random_state', 42)

    imputer = SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)

    # Determine class_weight parameter
    use_class_weight = 'balanced' if balancing == 'ClassWeight' else None

    if model_name == 'DecisionTree':
        clf = DecisionTreeClassifier(
            random_state=random_state,
            class_weight=use_class_weight
        )
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(
            random_state=random_state,
            class_weight=use_class_weight
        )
    elif model_name == 'GradientBoosting':
        # GradientBoosting doesn't support class_weight natively.
        # We handle imbalance via SMOTE or sample_weight (not implemented here).
        clf = GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=100,
        )
    elif model_name == 'Dummy':
        clf = DummyClassifier(strategy="most_frequent", random_state=random_state)
    else:
        raise ValueError(f"Unknown model {model_name}")

    steps = [
        ('imputer', imputer),
    ]

    # Add SMOTE only when balancing=='SMOTE' and model is not Dummy
    if balancing == 'SMOTE' and model_name != 'Dummy':
        # k_neighbors=2 is used because minority class is very small (n=11)
        steps.append(('smote', SMOTE(random_state=random_state, k_neighbors=2)))

    steps.append(('model', clf))

    return Pipeline(steps)


def get_param_grid(model_name: str, balancing: str = 'SMOTE'):
    """
    Returns reasonable hyperparameter grids for tuning.
    Prefixed with 'model__' because of the Pipeline.

    When balancing == 'ClassWeight', class_weight is fixed to 'balanced'
    and NOT part of the search space. When 'None' or 'SMOTE',
    class_weight is fixed to None.
    """
    if model_name == 'DecisionTree':
        return {
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [2, 3, 4, 5],
            'model__min_samples_split': [4, 6, 10],
            'model__min_samples_leaf': [2, 3, 5],
        }
    elif model_name == 'RandomForest':
        return {
            'model__n_estimators': [200, 300, 500],
            'model__max_depth': [3, 4, 5, 6],
            'model__min_samples_split': [4, 6, 10],
            'model__min_samples_leaf': [1, 2, 3, 4],
            'model__max_features': ['sqrt', 'log2'],
        }
    elif model_name == 'GradientBoosting':
        return {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [2, 3, 4],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__min_samples_split': [2, 4, 6],
            'model__min_samples_leaf': [1, 2, 3, 5],
            'model__subsample': [0.7, 0.8, 1.0],
        }
    return {}
