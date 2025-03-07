# a preprocessor class which can be instantiated with a list of functions

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Preprocessor:
    """Class to preprocess data using a list of functions."""
    
    def __init__(self, funcs):
        """
        Initialize the preprocessor with a list of preprocessing functions (each defined with its own class).
        
        Args:
            funcs (list): List of class functions to apply to data
        """
        self.funcs = funcs
        
    def fit(self, X):
        """Fit the preprocessor on the data."""
        for func in self.funcs:
            func.fit(X)
        return self
    
    def transform(self, X, fit=False):
        """Transform the data using the list of functions"
        """
        if fit:
            self.fit(X)
        for func in self.funcs:
            X = func.transform(X)
        return X
    
    def inverse_transform(self, X):
        """Inverse transform the data using the list of functions."""
        for func in reversed(self.funcs):
            X = func.inverse_transform(X)
        return X
    
    
class BaseFunction:
    """Base class for preprocessing functions."""
    
    def fit(self, X):
        """Fit the function on the data."""
        return self
    
    def transform(self, X):
        """Transform the data using the function."""
        raise NotImplementedError
    
    def inverse_transform(self, X):
        """Inverse transform the data using the function."""
        raise NotImplementedError
    
class StandardScalerFunction(BaseFunction):
    """StandardScaler function for preprocessing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

class PCAFunction(BaseFunction):
    """PCA function for preprocessing."""
    
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
        
    def fit(self, X):
        self.pca.fit(X)
        return self
    
    def transform(self, X):
        return self.pca.transform(X)
    
    def inverse_transform(self, X):
        return self.pca.inverse_transform(X)
    
class TSNEFunction(BaseFunction):
    """t-SNE function for preprocessing."""
    
    def __init__(self, n_components):
        self.tsne = TSNE(n_components=n_components)
        
    def fit(self, X):
        self.tsne.fit(X)
        return self
    
    def transform(self, X, fit=True):
        if fit:
            return self.tsne.fit_transform(X)
        else:
            raise ValueError("t-SNE does not support transform without fitting.")
    
    def inverse_transform(self, X):
        return self.tsne.inverse_transform(X)