from abc import ABC, abstractmethod
import numpy as np


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detection models.

    All model implementations should inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Fit the model to the training data.

        Args:
            X: Training data of shape (n_samples, n_features)
            **kwargs: Additional model-specific parameters
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict anomaly scores for the input data.

        Args:
            X: Data of shape (n_samples, n_features)
            **kwargs: Additional model-specific parameters

        Returns:
            np.ndarray: Anomaly scores where higher values indicate more anomalous samples
        """
        pass

    def predict_binary(
        self, X: np.ndarray, threshold: float = None, **kwargs
    ) -> np.ndarray:
        """
        Predict binary labels (0: normal, 1: anomaly) for the input data.

        Args:
            X: Data of shape (n_samples, n_features)
            threshold: Threshold for binary classification. If None, a default threshold is used.
            **kwargs: Additional model-specific parameters

        Returns:
            np.ndarray: Binary predictions (0: normal, 1: anomaly)
        """
        scores = self.predict(X, **kwargs)

        if threshold is None:
            # Default to a percentile-based threshold if none is specified
            threshold = np.percentile(scores, 95)

        return (scores >= threshold).astype(int)
