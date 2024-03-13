import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class StringClean(BaseEstimator):
    """
    features: ['f1', 'f2']
    function: strip + lower
    """
    def __init__(self, features):
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for col in self.features:
            X_[col] = X_[col].str.strip()
            X_[col] = X_[col].str.lower()
        return X_


class RemoveInvalidEmail(BaseEstimator):
    def __init__(self, features):
        self.features = features
        self.email_pattern = r'^[\w\.-]+@[\w\.-]+(\.[\w-]+)+$'

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for col in self.features:
            X_email_df = X_[X_[col].str.contains(self.email_pattern, na=False)]
        return X_email_df



class DateTypeConvert(BaseEstimator):
    """
    features: ['f1', 'f2']
    function: pd.to_datetime(X['f1'])
    """
    def __init__(self, features):
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for col in self.features:
            X_[col] = pd.to_datetime(X_[col])
        return X_



class AggMinMax(BaseEstimator):
    """
    groupby: 'f1', 
    aggcol: 'f2'
    function: X_.groupby('f1')['f2'].agg(min_='min', max_='max')
    return agg_df
    """
    def __init__(self, groupby= '', aggcol = ''):
        self.groupby = groupby
        self.aggcol = aggcol
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        agg_df = X_.groupby(self.groupby)[self.aggcol].agg(min_='min', max_='max')
        return agg_df

class FilterLE(BaseEstimator):
    def __init__(self, filter_col= None, filter_value = None, isDate=False):
        self.filter_col = filter_col
        self.filter_value = filter_value
        self.isDate = isDate
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.isDate is True:
            X_[self.filter_col] = pd.to_datetime(X_[self.filter_col])
        filter_df = X_[X_[self.filter_col]<self.filter_value]
        return filter_df


class FilterGE(BaseEstimator):
    def __init__(self, filter_col= None, filter_value = None, isDate=False):
        self.filter_col = filter_col
        self.filter_value = filter_value
        self.isDate = isDate
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.isDate is True:
            X_[self.filter_col] = pd.to_datetime(X_[self.filter_col])
        filter_df = X_[X_[self.filter_col]>self.filter_value]
        return filter_df


