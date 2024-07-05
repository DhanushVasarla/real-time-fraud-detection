from sklearn.base import BaseEstimator, TransformerMixin

class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_columns=None, new_col =None ):
        self.add_columns = add_columns
        self.new_col = new_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        col1,col2 = self.add_columns
        X_new[self.new_col] = X_new[col1] + X_new[col2]
        return X_new