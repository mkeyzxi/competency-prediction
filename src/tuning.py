from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, ParameterGrid, cross_val_predict
)
from sklearn.metrics import (
    make_scorer, recall_score, balanced_accuracy_score, fbeta_score,
    precision_score, accuracy_score, f1_score
)
import numpy as np
import warnings
from src.utils import load_config
from src.models import get_param_grid


def _get_inner_scorer(scoring_target='recall_bk'):
    """
    Returns the scorer used by the inner CV for hyperparameter search.

    Options:
    - 'recall_bk': Recall for Belum Kompeten (pos_label=0)
    - 'f2_bk':     F2-score for BK (weights recall 2x more than precision)
    - 'balanced_accuracy': Standard balanced accuracy
    """
    if scoring_target == 'recall_bk':
        return make_scorer(recall_score, pos_label=0, zero_division=0)
    elif scoring_target == 'f2_bk':
        return make_scorer(fbeta_score, beta=2, pos_label=0, zero_division=0)
    elif scoring_target == 'balanced_accuracy':
        return make_scorer(balanced_accuracy_score)
    else:
        raise ValueError(f"Unknown scoring_target: {scoring_target}")


def build_search_estimator(model, model_name,
                           config_path='configs/experiment_config.yaml',
                           custom_param_grid=None,
                           balancing='SMOTE',
                           scoring_target='recall_bk'):
    """
    Wraps the model in a RandomizedSearchCV for use as the inner loop
    of nested cross-validation.

    This estimator is passed to cross_validate() in evaluate_cv(),
    ensuring that hyperparameter tuning happens INSIDE each outer fold
    and never touches the test set.
    """
    config = load_config(config_path)
    random_state = config['random_state']

    # Inner CV uses 3 folds for stability with small minority class (11 samples)
    cv = StratifiedKFold(n_splits=3, shuffle=True,
                         random_state=random_state)

    param_grid = (custom_param_grid if custom_param_grid is not None
                  else get_param_grid(model_name, balancing))

    if not param_grid:
        return model

    scorer = _get_inner_scorer(scoring_target)

    total_combinations = len(ParameterGrid(param_grid))
    n_iter = min(50, total_combinations)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        random_state=random_state
    )
    return random_search


def tune_hyperparameters(model, model_name, X_train, y_train,
                         config_path='configs/experiment_config.yaml',
                         custom_param_grid=None,
                         balancing='SMOTE',
                         scoring_target='recall_bk'):
    """
    Tunes hyperparameters on the training set using inner CV.
    Returns (best_estimator, best_params).
    """
    search_estimator = build_search_estimator(
        model, model_name, config_path, custom_param_grid,
        balancing, scoring_target
    )

    if search_estimator is model:
        model.fit(X_train, y_train)
        return model, {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search_estimator.fit(X_train, y_train)

    return search_estimator.best_estimator_, search_estimator.best_params_


def evaluate_thresholds(model, X, y,
                        thresholds=np.arange(0.20, 0.80, 0.02),
                        optimize_for='f2_bk'):
    """
    Evaluates different probability thresholds using inner CV on training data.

    Threshold selection uses StratifiedKFold cross_val_predict on the
    training set to avoid leakage — we never peek at the test set.

    Parameters
    ----------
    optimize_for : str
        'f2_bk'             – F2-score for BK (recommended for EWS)
        'recall_bk'         – maximize Recall BK
        'balanced_accuracy' – maximize balanced accuracy

    Returns (all_results, best_threshold, best_metric_value).
    """
    results = []
    best_threshold = 0.50
    best_metric = -1.0

    if not hasattr(model, "predict_proba"):
        return results, best_threshold, best_metric

    try:
        # Inner CV uses 3 folds for stability
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Get probability for class 0 (BK)
            p_bk = cross_val_predict(model, X, y, cv=cv,
                                     method='predict_proba')[:, 0]
    except Exception:
        return results, best_threshold, best_metric

    for thresh in thresholds:
        # Predict 0 (BK) if p_bk >= thresh, else 1 (K)
        y_pred_thresh = np.where(p_bk >= thresh, 0, 1)
        acc = accuracy_score(y, y_pred_thresh)
        bal_acc = balanced_accuracy_score(y, y_pred_thresh)
        rec_bk = recall_score(y, y_pred_thresh, pos_label=0, zero_division=0)
        prec_bk = precision_score(y, y_pred_thresh, pos_label=0, zero_division=0)
        f1_bk = f1_score(y, y_pred_thresh, pos_label=0, zero_division=0)
        f2_bk = fbeta_score(y, y_pred_thresh, beta=2, pos_label=0, zero_division=0)

        results.append({
            'threshold': round(float(thresh), 2),
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'recall_bk': rec_bk,
            'precision_bk': prec_bk,
            'f1_bk': f1_bk,
            'f2_bk': f2_bk,
        })

    # Constraint-based threshold selection:
    # 1. Must have recall_bk >= 0.70
    # 2. Among valid candidates, pick max balanced_accuracy
    valid_candidates = [r for r in results if r['recall_bk'] >= 0.70]
    
    if valid_candidates:
        # Sort by balanced_accuracy descending, then f2_bk descending (as tie-breaker)
        valid_candidates.sort(key=lambda x: (x['balanced_accuracy'], x['f2_bk']), reverse=True)
        best = valid_candidates[0]
        best_threshold = best['threshold']
        best_metric = best['balanced_accuracy'] # Returning bal_acc to show optimization metric
    else:
        # Fallback: if no threshold achieves 70% recall, pick the one with max recall
        results.sort(key=lambda x: (x['recall_bk'], x['balanced_accuracy']), reverse=True)
        best = results[0]
        best_threshold = best['threshold']
        best_metric = best['recall_bk']

    return results, best_threshold, best_metric
