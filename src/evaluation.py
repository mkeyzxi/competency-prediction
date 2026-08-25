import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from src.utils import load_config

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

def evaluate_cv(model, X_train, y_train, config_path='configs/experiment_config.yaml'):
    """
    Evaluates the model using RepeatedStratifiedKFold on training set.
    Returns a dictionary of mean, std metrics, and 95% CI for F1.
    """
    config = load_config(config_path)
    cv_splits = config['cv']['n_splits']
    # Default to 5 repeats if not in config
    n_repeats = config['cv'].get('n_repeats', 5)
    random_state = config['random_state']
    
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=n_repeats, random_state=random_state)
    
    # We define positive class as 0 (Belum Kompeten)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0),
        'specificity': make_scorer(specificity_score, pos_label=0)
    }
    
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    
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

def evaluate_test(model, X_test, y_test, scenario_name, model_name, output_dir='results/confusion_matrix'):
    """
    Evaluates on test set and saves confusion matrix.
    """
    y_pred = model.predict(X_test)
    
    results = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'test_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'test_f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'test_specificity': specificity_score(y_test, y_pred, pos_label=0)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Belum Kompeten (0)', 'Kompeten (1)'], yticklabels=['Belum Kompeten (0)', 'Kompeten (1)'])
    plt.title(f'Confusion Matrix - {scenario_name} - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cm_{scenario_name}_{model_name}.png')
    plt.close()
    
    return results, y_pred
