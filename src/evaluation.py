from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils import load_config

def evaluate_cv(model, X_train, y_train, config_path='configs/experiment_config.yaml'):
    """
    Evaluates the model using Stratified 5-Fold CV on training set.
    Returns a dictionary of mean and std metrics.
    """
    config = load_config(config_path)
    cv_splits = config['cv']['n_splits']
    random_state = config['random_state']
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    
    results = {}
    for metric in scoring:
        results[f'cv_{metric}_mean'] = np.mean(scores[f'test_{metric}'])
        results[f'cv_{metric}_std'] = np.std(scores[f'test_{metric}'])
        
    return results

def evaluate_test(model, X_test, y_test, scenario_name, model_name):
    """
    Evaluates on test set and saves confusion matrix.
    """
    y_pred = model.predict(X_test)
    
    results = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {scenario_name} - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    os.makedirs('results/confusion_matrix', exist_ok=True)
    plt.savefig(f'results/confusion_matrix/cm_{scenario_name}_{model_name}.png')
    plt.close()
    
    return results, y_pred
