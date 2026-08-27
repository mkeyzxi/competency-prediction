from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold, ParameterGrid, cross_val_predict
from sklearn.metrics import (
    f1_score, make_scorer, accuracy_score, recall_score,
    precision_score, balanced_accuracy_score
)
import numpy as np
import warnings
from src.utils import load_config
from src.models import get_param_grid

def build_search_estimator(model, model_name, config_path='configs/experiment_config.yaml', custom_param_grid=None):
    config = load_config(config_path)
    cv_splits = config['cv']['n_splits']
    random_state = config['random_state']
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    param_grid = custom_param_grid if custom_param_grid is not None else get_param_grid(model_name)
    
    if not param_grid:
        return model
        
    bal_acc = make_scorer(balanced_accuracy_score)
    
    total_combinations = len(ParameterGrid(param_grid))
    n_iter = min(50, total_combinations) # reduced for nested CV performance
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=bal_acc,
        cv=cv,
        n_jobs=-1,
        random_state=random_state
    )
    return random_search

def tune_hyperparameters(model, model_name, X_train, y_train, config_path='configs/experiment_config.yaml', custom_param_grid=None):
    search_estimator = build_search_estimator(model, model_name, config_path, custom_param_grid)
    
    if search_estimator == model:
        model.fit(X_train, y_train)
        return model, {}
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search_estimator.fit(X_train, y_train)
        
    return search_estimator.best_estimator_, search_estimator.best_params_

def evaluate_thresholds(model, X, y, thresholds=np.arange(0.30, 0.75, 0.05)):
    """
    Evaluates different probability thresholds for models that support predict_proba.
    Returns (all_results, best_threshold, best_metric_value).
    
    Threshold selection uses RepeatedStratifiedKFold on the training set
    to avoid data leakage — we never peek at the test set.
    The best threshold is the one that maximizes balanced_accuracy on CV.
    """
    results = []
    best_threshold = 0.50
    best_metric = -1.0
    
    if not hasattr(model, "predict_proba"):
        return results, best_threshold, best_metric
        
    try:
        # Use cross_val_predict to get out-of-fold probabilities
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    except Exception:
        return results, best_threshold, best_metric
        
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y, y_pred_thresh)
        bal_acc = balanced_accuracy_score(y, y_pred_thresh)
        f1_macro = f1_score(y, y_pred_thresh, average='macro', zero_division=0)
        recall_bk = recall_score(y, y_pred_thresh, pos_label=0, zero_division=0)
        
        results.append({
            'threshold': round(float(thresh), 2),
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1_macro': f1_macro,
            'recall_belum_kompeten': recall_bk,
        })
        
        # Use balanced_accuracy as the optimization target
        if bal_acc > best_metric:
            best_metric = bal_acc
            best_threshold = round(float(thresh), 2)
    
    return results, best_threshold, best_metric
