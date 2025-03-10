from sklearn.mixture import GaussianMixture as SKLearnGMM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from .base import BaseAnomalyDetector


class GaussianMixture(BaseAnomalyDetector):
    """
    Anomaly detection using Gaussian Mixture Models.
    Identifies anomalies as samples with low likelihood under the model.
    """

    def __init__(self, n_components=5, random_state=None):
        """
        Initialize the Gaussian Mixture Model.

        Args:
            n_components: Number of mixture components
            random_state: Random state for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = SKLearnGMM(
            n_components=n_components, covariance_type="full", random_state=random_state
        )

    def fit(self, X, **kwargs):
        """
        Fit the GMM to the training data.

        Args:
            X: Training data of shape (n_samples, n_features)
            **kwargs: Additional parameters passed to sklearn's GMM
        """
        self.model.fit(X, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """
        Predict anomaly scores as negative log-likelihood.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Anomaly scores where higher values = more anomalous
        """
        # Calculate negative log-likelihood as anomaly score
        log_likelihood = self.model.score_samples(X)
        # Return negative log-likelihood (higher = more anomalous)
        return -log_likelihood


class LocalOutlierFactorDetector(BaseAnomalyDetector):
    """Anomaly detection using Local Outlier Factor algorithm."""

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination, novelty=True
        )

    def fit(self, X, **kwargs):
        self.model.fit(X, **kwargs)
        return self

    def predict(self, X, **kwargs):
        # Return negative decision function as anomaly score
        return -self.model.decision_function(X)


class IsolationForestDetector(BaseAnomalyDetector):
    """Anomaly detection using Isolation Forest algorithm."""

    def __init__(self, n_estimators=100, contamination=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X, **kwargs):
        self.model.fit(X, **kwargs)
        return self

    def predict(self, X, **kwargs):
        # Return negative decision function as anomaly score
        return -self.model.decision_function(X)
