from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class SimpleImputerWithMapping(BaseEstimator, TransformerMixin):
    def __init__(self, mapping, target_col, reference_col):
        self.mapping = mapping  
        self.target_col = target_col 
        self.reference_col = reference_col
    
    def fit(self, X, y=None):
        return self  # No training needed, just a transformation
    
    def transform(self, X):
        X = X.copy() 
        missing_idx = X[X[self.target_col].isnull()].index 
        X.loc[missing_idx, self.target_col] = X.loc[missing_idx, self.reference_col].map(self.mapping)
        return X



class CustomBinning(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X_new = pd.cut(X[self.target_col], bins=[0, 150, 300, 500], labels=['Short', 'Medium', 'Long'])
        #X = pd.concat([X.drop(self.target_col, axis=1), X_new], axis=1)
        X[self.target_col] = X_new
        return X


class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names)

