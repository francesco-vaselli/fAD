# various utilities for generating synthetic data
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def make_synthetic_blob_data(n_samples=1000, n_outliers=50, random_state=42):
    """Generate synthetic data with normal points and outliers."""
    # Generate normal data
    X, _ = make_blobs(
        n_samples=n_samples - n_outliers,
        centers=2,
        cluster_std=0.5,
        random_state=random_state,
    )

    # Generate outliers
    rng = np.random.RandomState(random_state)
    outliers_x = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

    # Combine normal and outlier data
    X = np.vstack([X, outliers_x])
    y = np.zeros(n_samples)
    y[-n_outliers:] = 1  # Mark outliers as anomalies (1)

    return X, y


def make_synthetic_phys_data(n_samples=1000, n_outliers=50, random_state=42):
    """Generate synthetic data with normal points and outliers.
    X has 4 features corresponding to p_T, eta, phi, and E
    outliers have points in the tails of p_T, E or both
    """

    # Generate p_T, eta, phi, E for normal data
    # p_T has a falling exponential distribution (log-normal)
    # eta is uniform between -2.5 and 2.5
    # phi is uniform between -pi and pi
    # E is a gaussian distribution with mean 100 and std 10
    p_T = np.random.lognormal(mean=1.5, sigma=1, size=n_samples - n_outliers)
    eta = np.random.uniform(low=-2.5, high=2.5, size=n_samples - n_outliers)
    phi = np.random.uniform(low=-np.pi, high=np.pi, size=n_samples - n_outliers)
    E = np.random.normal(loc=100, scale=10, size=n_samples - n_outliers)

    # Generate p_T, eta, phi, E for outliers
    # Outliers have high p_T and/or high E
    p_T_outliers = np.random.lognormal(mean=3, sigma=1, size=n_outliers)
    eta_outliers = np.random.uniform(low=-2.5, high=2.5, size=n_outliers)
    phi_outliers = np.random.uniform(low=-np.pi, high=np.pi, size=n_outliers)
    E_outliers = np.random.normal(loc=150, scale=20, size=n_outliers)

    # Combine normal and outlier data
    X = np.vstack(
        [
            np.column_stack([p_T, eta, phi, E]),
            np.column_stack([p_T_outliers, eta_outliers, phi_outliers, E_outliers]),
        ]
    )
    y = np.zeros(n_samples)
    y[-n_outliers:] = 1  # Mark outliers as anomalies (1)

    return X, y


if __name__ == "__main__":
    X, y = make_synthetic_blob_data()
    print(X.shape, y.shape)
    X, y = make_synthetic_phys_data(n_samples=10000, n_outliers=1000)
    print(X.shape, y.shape)
    # plot components of X
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    for i in range(4):
        axs[i].hist(X[:, i], bins=50)
        axs[i].set_title(f"Feature {i + 1}")
    plt.savefig("synthetic_phys_data.png")
