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
        # Removed n_jobs=-1 to avoid joblib warnings during nested CV
        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # If X is a dataframe, keep it, otherwise ensure it's handled properly
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X
            
        if self.importance_type == 'gini':
            clf.fit(X_arr, y)
            importances = clf.feature_importances_
        elif self.importance_type == 'permutation':
            from sklearn.model_selection import train_test_split
            # Create an inner validation set to prevent leakage/overfitting in selection
            X_in_train, X_in_val, y_in_train, y_in_val = train_test_split(
                X_arr, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            clf.fit(X_in_train, y_in_train)
            # Use n_jobs=None to avoid nested joblib warnings
            result = permutation_importance(clf, X_in_val, y_in_val, n_repeats=5, random_state=self.random_state, n_jobs=None)
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
