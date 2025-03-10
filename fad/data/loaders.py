import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
import h5py


@dataclass
class Dataset:
    """Class to hold train and test data along with metadata."""

    train: np.ndarray
    test: np.ndarray
    train_labels: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None
    feature_names: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None


def load_dataset(path: str, **kwargs) -> Dataset:
    """
    Load a dataset from a path.

    Args:
        path: Path to the dataset file or directory
        **kwargs: Additional parameters for loading

    Returns:
        Dataset: A Dataset object containing train and test data
    """
    # Determine file type and load accordingly
    if path.endswith(".csv"):
        return _load_csv_dataset(path, **kwargs)
    elif path.endswith(".npz"):
        return _load_npz_dataset(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format for {path}")


def load_challenge_dataset(path_bkg: str, path_anom: str, **kwargs) -> Dataset:
    """
    Load a dataset from a path.

    Args:
        path_bkg: Path to the background dataset file
        path_anom: Path to the anomaly dataset file
        **kwargs: Additional parameters for loading

    Returns:
        Dataset: A Dataset object containing train and test data
    """
    return _load_h5_challenge_dataset(path_bkg, path_anom, **kwargs)


def _load_csv_dataset(
    path: str, test_size: float = 0.2, label_column: Optional[str] = None, **kwargs
) -> Dataset:
    """Load dataset from a CSV file."""
    df = pd.read_csv(path)

    # Extract labels if specified
    if label_column:
        labels = df[label_column].values
        features = df.drop(columns=[label_column])
    else:
        labels = None
        features = df

    # Split into train and test sets
    from sklearn.model_selection import train_test_split

    if labels is not None:
        train_data, test_data, train_labels, test_labels = train_test_split(
            features.values, labels, test_size=test_size, random_state=42, **kwargs
        )
    else:
        train_data, test_data = train_test_split(
            features.values, test_size=test_size, random_state=42, **kwargs
        )
        train_labels, test_labels = None, None

    return Dataset(
        train=train_data,
        test=test_data,
        train_labels=train_labels,
        test_labels=test_labels,
        feature_names=features.columns.tolist(),
        metadata={"source": path},
    )


def _load_npz_dataset(path: str, **kwargs) -> Dataset:
    """Load dataset from a NumPy .npz file."""
    data = np.load(path)

    # Expected keys in the npz file
    train_key = kwargs.get("train_key", "X_train")
    test_key = kwargs.get("test_key", "X_test")
    train_labels_key = kwargs.get("train_labels_key", "y_train")
    test_labels_key = kwargs.get("test_labels_key", "y_test")

    train_data = data[train_key]
    test_data = data[test_key]

    # Try to load labels if they exist
    train_labels = data[train_labels_key] if train_labels_key in data else None
    test_labels = data[test_labels_key] if test_labels_key in data else None

    return Dataset(
        train=train_data,
        test=test_data,
        train_labels=train_labels,
        test_labels=test_labels,
        metadata={"source": path},
    )


def _load_h5_challenge_dataset(path_bkg: str, path_anom: str, **kwargs) -> Dataset:
    """Load dataset from a HDF5 file."""

    data_bkg = h5py.File(path_bkg, "r")
    data_anom = h5py.File(path_anom, "r")

    # limit to 100k events for now
    x_bkg = data_bkg["Particles"][:100000]
    x_anom = data_anom["Particles"][:30000]
    # shape is (n_events, n_part, n_features), we flatten the last two dimensions
    x_bkg = x_bkg.reshape(x_bkg.shape[0], -1)
    x_anom = x_anom.reshape(x_anom.shape[0], -1)

    # # assign the labels
    y_bkg = np.zeros(x_bkg.shape[0])
    y_anom = np.ones(x_anom.shape[0])

    # # split into train and test
    # 70k of bkg for training, the rest for testing with the anom
    n_train_bkg = 70000
    n_test_bkg = x_bkg.shape[0] - n_train_bkg
    n_test_anom = x_anom.shape[0]

    X_train_bkg = x_bkg[:n_train_bkg]
    X_test_bkg = x_bkg[n_train_bkg:]
    X_test_anom = x_anom

    y_train_bkg = y_bkg[:n_train_bkg]
    y_test_bkg = y_bkg[n_train_bkg:]
    y_test_anom = y_anom

    return Dataset(
        train=X_train_bkg,
        test=np.concatenate((X_test_bkg, X_test_anom)),
        train_labels=y_train_bkg,
        test_labels=np.concatenate((y_test_bkg, y_test_anom)),
        metadata={"source": path_bkg},
    )
