import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from fad.data.generators import make_synthetic_blob_data
from fad.models.flow_matching import FlowMatchingAnomalyDetector
from fad.data.preprocessing import Preprocessor, StandardScalerFunction


def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 1000
    n_outliers = 50
    n_features = 2

    # Generate synthetic data with outliers
    print("Generating synthetic data...")
    X, y = make_synthetic_blob_data(
        n_samples=n_samples, n_outliers=n_outliers, random_state=42
    )

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

    # Initialize and train the Flow Matching anomaly detector
    print("Training Flow Matching anomaly detector...")
    detector = FlowMatchingAnomalyDetector(
        input_dim=n_features,
        hidden_dim=128,
        batch_size=256,
        iterations=300,  # Reduced for demonstration
        print_every=10,
        lr=0.001,
    )
    detector.fit(X_train)

    # Predict anomaly scores
    print("Predicting anomaly scores...")
    scores, samples = detector.predict(X_test)

    # print mean of scores for normal and outlier data
    print(f"Mean score for normal data: {scores[y_test == 0].mean()}")
    print(f"Mean score for outlier data: {scores[y_test == 1].mean()}")

    anomaly_mask = scores > 10

    # plot the samples with 2 different colors
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_test[anomaly_mask == 0, 0],
        X_test[anomaly_mask == 0, 1],
        c="blue",
        label="Normal",
        alpha=0.5,
    )
    plt.scatter(
        X_test[anomaly_mask == 1, 0],
        X_test[anomaly_mask == 1, 1],
        c="red",
        label="Anomaly",
        alpha=0.5,
    )
    plt.title("Flow Matching Anomaly Detection")
    plt.legend()
    plt.savefig("flow_matching_ad_samples.png")
    # Calculate AUC-ROC
    auc = roc_auc_score(y_test, scores)
    print(f"AUC-ROC: {auc:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))

    # Plot normal points
    plt.scatter(
        X_test[y_test == 0, 0],
        X_test[y_test == 0, 1],
        c="blue",
        label="Normal",
        alpha=0.5,
    )

    # Plot anomalies
    plt.scatter(
        X_test[y_test == 1, 0],
        X_test[y_test == 1, 1],
        c="red",
        label="Anomaly",
        alpha=0.5,
    )

    plt.title(f"Flow Matching Anomaly Detection (AUC-ROC: {auc:.4f})")
    plt.legend()
    plt.savefig("flow_matching_ad_results.png")
    plt.close()

    print("Results saved to flow_matching_ad_results.png")


if __name__ == "__main__":
    main()
