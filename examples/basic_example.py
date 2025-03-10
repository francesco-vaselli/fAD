"""
Basic example demonstrating how to use fAD for anomaly detection.
This example creates synthetic data with outliers and applies
multiple anomaly detection models for comparison.
"""

import matplotlib.pyplot as plt

# Import fAD components
from fad.models.statistical import GaussianMixture, IsolationForestDetector
from fad.data.generators import make_synthetic_blob_data
from fad.data.preprocessing import Preprocessor, StandardScalerFunction
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
# from fad.validation.metrics import calculate_auc, calculate_precision_recall
# from fad.visualization.plots import plot_anomaly_scores, plot_roc_curve


def main():
    # Generate synthetic data with outliers
    print("Generating synthetic data...")
    X, y = make_synthetic_blob_data(random_state=42)

    # Split into train (normal only) and test sets
    normal_idx = y == 0
    X_train = X[normal_idx]
    X_test = X
    y_test = y

    # preprocess data
    print("Preprocessing data...")
    preprocessor = Preprocessor([StandardScalerFunction()])
    X_train = preprocessor.transform(X_train, fit=True)
    X_test = preprocessor.transform(X_test)

    # Create and fit models
    print("Training models...")
    models = {
        "Gaussian Mixture": GaussianMixture(n_components=3),
        "Isolation Forest": IsolationForestDetector(contamination=0.05),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train)

        # Get anomaly scores
        scores = model.predict(X_test)
        results[name] = scores

        # # Calculate metrics
        auc = roc_auc_score(y_test, scores)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, scores >= 0, average="binary"
        )

        print(f"{name} results:")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

    # Visualize results
    print("Generating visualizations...")
    visualize_results(X_test, y_test, results)
    print("Done!")


def visualize_results(X, y_true, model_scores):
    """Visualize anomaly detection results."""
    # Plot data with true anomalies
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X[y_true == 0, 0], X[y_true == 0, 1], c="blue", alpha=0.5, label="Normal"
    )
    plt.scatter(
        X[y_true == 1, 0], X[y_true == 1, 1], c="red", alpha=0.5, label="Anomaly"
    )
    plt.title("True Data Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("true_data.png")

    # # Plot ROC curves for all models
    # RocCurveDisplay(y_true, model_scores).plot()
    # plt.title('ROC Curves')
    # plt.tight_layout()
    # plt.savefig('roc_curves.png')

    # Plot anomaly scores for each model
    for name, scores in model_scores.items():
        # a scatterplot with color corresponding to anomaly scores
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=scores, cmap="coolwarm", alpha=0.5)
        plt.colorbar(label="Anomaly Score")

        plt.title(f"{name} Anomaly Scores")
        plt.tight_layout()
        plt.savefig(f"{name}_scores.png")


if __name__ == "__main__":
    main()
