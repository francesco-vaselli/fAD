# a preprocessor class which can be instantiated with a list of functions

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


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
        """Transform the data using the list of functions" """
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


class PtOnlyPreprocessing(BaseFunction):
    """Preprocessing function that only processes p_T values."""

    def __init__(self, padding_value=0, new_padding_value=-11):
        self.padding_value = padding_value
        self.new_padding_value = new_padding_value
        self.scaler = StandardScaler()

    def fit(self, X):
        # Extract p_T values (every 3rd feature starting from index 0)
        pts = X[:, ::3]

        # Create mask for non-zero entries
        non_zero_mask = pts != self.padding_value

        # Fit scaler only on non-zero p_T values
        if np.any(non_zero_mask):
            non_zero_pts = pts[non_zero_mask].reshape(-1, 1)
            self.scaler.fit(non_zero_pts)

        return self

    def transform(self, X, fit=False):
        if fit:
            self.fit(X)

        # Create a copy to avoid modifying original data
        X_transformed = X.copy()

        # Extract p_T values
        pts = X_transformed[:, ::3]

        # Create mask for non-zero entries
        non_zero_mask = pts != self.padding_value

        # Transform non-zero p_T values
        if np.any(non_zero_mask):
            pts_transformed = pts.copy()
            non_zero_pts = pts[non_zero_mask].reshape(-1, 1)
            scaled_pts = self.scaler.transform(non_zero_pts).flatten()
            pts_transformed[non_zero_mask] = scaled_pts

            # Replace original padding with new padding value
            pts_transformed[~non_zero_mask] = self.new_padding_value

            # Update the p_T columns in the transformed array
            X_transformed[:, ::3] = pts_transformed

        return X_transformed

    def inverse_transform(self, X):
        raise NotImplementedError(
            "Inverse transform not implemented for PtOnlyPreprocessing."
        )
