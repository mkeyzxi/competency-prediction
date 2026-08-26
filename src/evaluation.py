import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, make_scorer, balanced_accuracy_score, roc_auc_score
)
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from src.utils import load_config
from src.tuning import build_search_estimator

def specificity_score(y_true, y_pred, pos_label=0):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # TN is when it's NOT the positive label. Since positive is 0, True Negative is correctly predicting 1 (Kompeten)
    # confusion_matrix(y_true, y_pred, labels=[0,1]) returns:
    # [ [True 0, False 1], [False 0, True 1] ]
    # Since we want pos_label=0:
    # TN = True 1, FP = False 1, FN = False 0, TP = True 0
    # Specificity = TN / (TN + FP) = True 1 / (True 1 + False 1)
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)

def evaluate_cv(model, model_name, X_train, y_train, config_path='configs/experiment_config.yaml'):
    """
    Evaluates the model using RepeatedStratifiedKFold on training set.
    For models with hyperparameters, this performs Nested Cross-Validation
    by wrapping the parameter search inside the outer evaluation loop.
    Returns a dictionary of mean, std metrics, and 95% CI for F1.
    """
    config = load_config(config_path)
    cv_splits = config['cv']['n_splits']
    # Default to 5 repeats if not in config
    n_repeats = config['cv'].get('n_repeats', 5)
    random_state = config['random_state']
    
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=n_repeats, random_state=random_state)
    
    # Build the inner CV search estimator (Nested CV setup)
    search_estimator = build_search_estimator(model, model_name, config_path)
    
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0),
        'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovr')
    }
    
    # Nested CV execution
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(search_estimator, X_train, y_train, cv=cv, scoring=scoring)
    
    results = {}
    for metric in scoring.keys():
        test_scores = scores[f'test_{metric}']
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)
        results[f'cv_{metric}_mean'] = mean_score
        results[f'cv_{metric}_std'] = std_score
        
        # Calculate 95% CI (Bootstrap-like or standard error)
        # Using t-distribution
        n = len(test_scores)
        sem = stats.sem(test_scores)
        ci = stats.t.interval(0.95, n-1, loc=mean_score, scale=sem)
        results[f'cv_{metric}_ci_lower'] = ci[0]
        results[f'cv_{metric}_ci_upper'] = ci[1]
        
    return results

def evaluate_test(model, X_test, y_test, scenario_name, model_name,
                  threshold=None, output_dir='results/confusion_matrix'):
    """
    Evaluates on test set and saves confusion matrix.
    
    If threshold is provided and the model supports predict_proba,
    predictions are made using that threshold instead of the default 0.50.
    """
    if threshold is not None and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
        except Exception:
            y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    
    results = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'test_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'test_f1': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    try:
        y_prob_test = model.predict_proba(X_test)[:, 1]
        results['test_roc_auc'] = roc_auc_score(y_test, y_prob_test)
    except:
        results['test_roc_auc'] = np.nan
        
    results['test_recall_belum_kompeten'] = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    
    thresh_label = f' (t={threshold})' if threshold is not None else ''
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Belum Kompeten (0)', 'Kompeten (1)'], yticklabels=['Belum Kompeten (0)', 'Kompeten (1)'])
    plt.title(f'Confusion Matrix - {scenario_name} - {model_name}{thresh_label}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cm_{scenario_name}_{model_name}.png')
    plt.close()
    
    return results, y_pred
