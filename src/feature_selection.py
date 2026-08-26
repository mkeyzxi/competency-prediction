from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd

class DynamicTopKSelector(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer to select top K features based on
    either Gini importance or Permutation importance using a Random Forest.
    This allows 'k' and 'importance_type' to be tuned inside a GridSearchCV/RandomizedSearchCV.
    """
    def __init__(self, k=10, importance_type='gini', random_state=42):
        self.k = k
        self.importance_type = importance_type
        self.random_state = random_state
        self.top_indices_ = None
        self.importances_ = None
        
    def fit(self, X, y=None):
        n_features = X.shape[1] if isinstance(X, np.ndarray) else len(X.columns)
        
        if self.k == 'all':
            actual_k = n_features
        else:
            actual_k = min(int(self.k), n_features)
            
        # Base model for computing importances
        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        
        # If X is a dataframe, keep it, otherwise ensure it's handled properly
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
            
        # Impute NaNs just for the importance calculation because RF cannot handle NaNs natively
        # (Assuming imputer is run BEFORE this step in the pipeline, so X_arr shouldn't have NaNs, 
        # but just in case, we assume pipeline structure: Imputer -> Selector -> Classifier)
        
        clf.fit(X_arr, y)
        
        if self.importance_type == 'gini':
            importances = clf.feature_importances_
        elif self.importance_type == 'permutation':
            result = permutation_importance(clf, X_arr, y, n_repeats=5, random_state=self.random_state, n_jobs=-1)
            importances = result.importances_mean
        else:
            raise ValueError("importance_type must be 'gini' or 'permutation'")
            
        self.importances_ = importances
        
        # Sort indices by importance descending
        sorted_indices = np.argsort(importances)[::-1]
        self.top_indices_ = sorted_indices[:actual_k]
        
        return self
        
    def transform(self, X):
        if self.top_indices_ is None:
            raise RuntimeError("You must fit the transformer before transforming the data.")
        
        if isinstance(X, np.ndarray):
            return X[:, self.top_indices_]
        else:
            # pandas dataframe
            return X.iloc[:, self.top_indices_]
