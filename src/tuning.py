from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils import load_config
from src.models import get_param_grid

def tune_hyperparameters(model, model_name, X_train, y_train, config_path='configs/experiment_config.yaml'):
    config = load_config(config_path)
    cv_splits = config['cv']['n_splits']
    random_state = config['random_state']
    primary_scoring = config['scoring']['primary']
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    param_grid = get_param_grid(model_name)
    
    # We use GridSearchCV as per PRD "GridSearchCV atau RandomizedSearchCV"
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=primary_scoring,
        cv=cv,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_
