import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    RocCurveDisplay,
)


def evaluate_performance(
    flow_matching,
    flow_name,
    path,
    epoch,
    X_test,
    dataset,
    X_test2,
    dataset2,
    X_test3,
    dataset3,
    X_test4,
    dataset4,
):
    results = {}
    target_fpr = 1e-5
    # %%
    # Predict anomaly scores
    flow_scores = flow_matching.predict(X_test, mode="vt")
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores[dataset.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores[dataset.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores[dataset.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores[dataset.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset.test_labels, flow_scores)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc = roc_auc_score(dataset.test_labels, flow_scores)
    flow_ap = average_precision_score(dataset.test_labels, flow_scores)
    flow_accuracy = accuracy_score(dataset.test_labels, flow_scores > 10)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy:.4f}")
    print(f"AUC-ROC: {flow_auc:.4f}")
    print(f"Average Precision: {flow_ap:.4f}")

    results["Flow Matching vt"] = {
        "scores": flow_scores,
        "auc": flow_auc,
        "ap": flow_ap,
        "time": 1,
        "TPR": tpr_interpolated,
    }

    # Predict anomaly scores
    flow_scores = flow_matching.predict(X_test, mode="vt_einsum")
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores[dataset.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores[dataset.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores[dataset.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores[dataset.test_labels == 1]):.4f}"
    )

    flow_auc = roc_auc_score(dataset.test_labels, flow_scores)
    flow_ap = average_precision_score(dataset.test_labels, flow_scores)
    flow_accuracy = accuracy_score(dataset.test_labels, flow_scores > 10)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy:.4f}")
    print(f"AUC-ROC: {flow_auc:.4f}")
    print(f"Average Precision: {flow_ap:.4f}")

    results["Flow Matching vt einsum"] = {
        "scores": flow_scores,
        "auc": flow_auc,
        "ap": flow_ap,
        "time": 1,
    }

    # now again in ODE mode
    flow_scores, transformed_data = flow_matching.predict(
        X_test, mode="ode", time_steps=2, step_size=None, return_transformed_data=True
    )
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores[dataset.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores[dataset.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores[dataset.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores[dataset.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset.test_labels, flow_scores)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc = roc_auc_score(dataset.test_labels, flow_scores)
    flow_ap = average_precision_score(dataset.test_labels, flow_scores)
    flow_accuracy = accuracy_score(dataset.test_labels, flow_scores > 1000)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy:.4f}")
    print(f"AUC-ROC: {flow_auc:.4f}")
    print(f"Average Precision: {flow_ap:.4f}")
    results["Flow Matching ode"] = {
        "scores": flow_scores,
        "auc": flow_auc,
        "ap": flow_ap,
        "time": 1,
        "TPR": tpr_interpolated,
    }

    # %%
    # Predict anomaly scores
    flow_scores2 = flow_matching.predict(X_test2, mode="vt")
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores2[dataset2.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores2[dataset2.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores2[dataset2.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores2[dataset2.test_labels == 1]):.4f}"
    )
    flow_auc2 = roc_auc_score(dataset2.test_labels, flow_scores2)
    flow_ap2 = average_precision_score(dataset2.test_labels, flow_scores2)
    flow_accuracy2 = accuracy_score(dataset2.test_labels, flow_scores2 > 10)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy2:.4f}")
    print(f"AUC-ROC: {flow_auc2:.4f}")
    print(f"Average Precision: {flow_ap2:.4f}")

    results["Flow Matching2 vt"] = {
        "scores": flow_scores2,
        "auc": flow_auc2,
        "ap": flow_ap2,
        "time": 1,
    }
    # Predict anomaly scores
    flow_scores2 = flow_matching.predict(X_test2, mode="vt_einsum")
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores2[dataset2.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores2[dataset2.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores2[dataset2.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores2[dataset2.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset2.test_labels, flow_scores2)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc2 = roc_auc_score(dataset2.test_labels, flow_scores2)
    flow_ap2 = average_precision_score(dataset2.test_labels, flow_scores2)
    flow_accuracy2 = accuracy_score(dataset2.test_labels, flow_scores2 > 10)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy2:.4f}")
    print(f"AUC-ROC: {flow_auc2:.4f}")
    print(f"Average Precision: {flow_ap2:.4f}")

    results["Flow Matching2 vt einsum"] = {
        "scores": flow_scores2,
        "auc": flow_auc2,
        "ap": flow_ap2,
        "time": 1,
        "TPR": tpr_interpolated,
    }
    # now again in ODE mode
    flow_scores2 = flow_matching.predict(
        X_test2, mode="ode", time_steps=2, step_size=None, log_density_calc="library"
    )
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores2[dataset2.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores2[dataset2.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores2[dataset2.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores2[dataset2.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset2.test_labels, flow_scores2)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc2 = roc_auc_score(dataset2.test_labels, flow_scores2)
    flow_ap2 = average_precision_score(dataset2.test_labels, flow_scores2)
    flow_accuracy2 = accuracy_score(dataset2.test_labels, flow_scores2 > 100)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy2:.4f}")
    print(f"AUC-ROC: {flow_auc2:.4f}")
    print(f"Average Precision: {flow_ap2:.4f}")
    results["Flow Matching2 ode"] = {
        "scores": flow_scores2,
        "auc": flow_auc2,
        "ap": flow_ap2,
        "time": 1,
        "TPR": tpr_interpolated,
    }

    # %%
    # Predict anomaly scores
    flow_scores3 = flow_matching.predict(X_test3, mode="vt")
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores3[dataset3.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores3[dataset3.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores3[dataset3.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores3[dataset3.test_labels == 1]):.4f}"
    )
    flow_auc3 = roc_auc_score(dataset3.test_labels, flow_scores3)
    flow_ap3 = average_precision_score(dataset3.test_labels, flow_scores3)
    flow_accuracy3 = accuracy_score(dataset3.test_labels, flow_scores3 > 10)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy3:.4f}")
    print(f"AUC-ROC: {flow_auc3:.4f}")
    print(f"Average Precision: {flow_ap3:.4f}")
    results["Flow Matching3 vt"] = {
        "scores": flow_scores3,
        "auc": flow_auc3,
        "ap": flow_ap3,
        "time": 1,
    }

    # Predict anomaly scores
    flow_scores3 = flow_matching.predict(X_test3, mode="vt_einsum")
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores3[dataset3.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores3[dataset3.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores3[dataset3.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores3[dataset3.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset3.test_labels, flow_scores3)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc3 = roc_auc_score(dataset3.test_labels, flow_scores3)
    flow_ap3 = average_precision_score(dataset3.test_labels, flow_scores3)
    flow_accuracy3 = accuracy_score(dataset3.test_labels, flow_scores3 > 10)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy3:.4f}")
    print(f"AUC-ROC: {flow_auc3:.4f}")
    print(f"Average Precision: {flow_ap3:.4f}")

    results["Flow Matching3 vt einsum"] = {
        "scores": flow_scores3,
        "auc": flow_auc3,
        "ap": flow_ap3,
        "time": 1,
        "TPR": tpr_interpolated,
    }
    # now again in ODE mode
    flow_scores3 = flow_matching.predict(
        X_test3, mode="ode", time_steps=2, step_size=None, log_density_calc="library"
    )
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores3[dataset3.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores3[dataset3.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores3[dataset3.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores3[dataset3.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset3.test_labels, flow_scores3)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc3 = roc_auc_score(dataset3.test_labels, flow_scores3)
    flow_ap3 = average_precision_score(dataset3.test_labels, flow_scores3)
    flow_accuracy3 = accuracy_score(dataset3.test_labels, flow_scores3 > 100)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy3:.4f}")
    print(f"AUC-ROC: {flow_auc3:.4f}")
    print(f"Average Precision: {flow_ap3:.4f}")
    results["Flow Matching3 ode"] = {
        "scores": flow_scores3,
        "auc": flow_auc3,
        "ap": flow_ap3,
        "time": 1,
        "TPR": tpr_interpolated,
    }

    # %%
    # Predict anomaly scores
    flow_scores4 = flow_matching.predict(X_test4, mode="vt")
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores4[dataset4.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores4[dataset4.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores4[dataset4.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores4[dataset4.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset4.test_labels, flow_scores4)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc4 = roc_auc_score(dataset4.test_labels, flow_scores4)
    flow_ap4 = average_precision_score(dataset4.test_labels, flow_scores4)
    flow_accuracy4 = accuracy_score(dataset4.test_labels, flow_scores4 > 10)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy4:.4f}")
    print(f"AUC-ROC: {flow_auc4:.4f}")
    print(f"Average Precision: {flow_ap4:.4f}")

    results["Flow Matching4 vt"] = {
        "scores": flow_scores4,
        "auc": flow_auc4,
        "ap": flow_ap4,
        "time": 1,
        "TPR": tpr_interpolated,
    }

    # Predict anomaly scores
    flow_scores4 = flow_matching.predict(X_test4, mode="vt_einsum")
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores4[dataset4.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores4[dataset4.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores4[dataset4.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores4[dataset4.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset4.test_labels, flow_scores4)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc4 = roc_auc_score(dataset4.test_labels, flow_scores4)
    flow_ap4 = average_precision_score(dataset4.test_labels, flow_scores4)
    flow_accuracy4 = accuracy_score(dataset4.test_labels, flow_scores4 > 10)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy4:.4f}")
    print(f"AUC-ROC: {flow_auc4:.4f}")
    print(f"Average Precision: {flow_ap4:.4f}")

    results["Flow Matching4 vt einsum"] = {
        "scores": flow_scores4,
        "auc": flow_auc4,
        "ap": flow_ap4,
        "time": 1,
        "TPR": tpr_interpolated,
    }
    # now again in ODE mode
    flow_scores4 = flow_matching.predict(
        X_test4, mode="ode", time_steps=2, step_size=None, log_density_calc="library"
    )
    # print mean and std of scores for normal and anomalous samples
    print(
        f"Mean score for normal samples: {np.mean(flow_scores4[dataset4.test_labels == 0]):.4f}"
    )
    print(
        f"Std score for normal samples: {np.std(flow_scores4[dataset4.test_labels == 0]):.4f}"
    )
    print(
        f"Mean score for anomalous samples: {np.mean(flow_scores4[dataset4.test_labels == 1]):.4f}"
    )
    print(
        f"Std score for anomalous samples: {np.std(flow_scores4[dataset4.test_labels == 1]):.4f}"
    )
    fpr, tpr, thresholds = roc_curve(dataset4.test_labels, flow_scores4)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    flow_auc4 = roc_auc_score(dataset4.test_labels, flow_scores4)
    flow_ap4 = average_precision_score(dataset4.test_labels, flow_scores4)
    flow_accuracy4 = accuracy_score(dataset4.test_labels, flow_scores4 > 100)
    # manually compute accuracy
    print(f"Accuracy: {flow_accuracy4:.4f}")
    print(f"AUC-ROC: {flow_auc4:.4f}")
    print(f"Average Precision: {flow_ap4:.4f}")
    results["Flow Matching4 ode"] = {
        "scores": flow_scores4,
        "auc": flow_auc4,
        "ap": flow_ap4,
        "time": 1,
        "TPR": tpr_interpolated,
    }

    # %% [markdown]
    # ## 5. Model Comparison and Visualization

    # %%
    # Compare ROC curves
    plt.figure(figsize=(10, 8))
    # remove "Flow Mathcing2" form results
    for name, result in results.items():
        print(name)
        if (
            (name == "Flow Matching2 vt")
            | (name == "Flow Matching2 vt einsum")
            | (name == "Flow Matching2 ode")
        ):
            # use the second dataset for this model
            RocCurveDisplay.from_predictions(
                dataset2.test_labels,
                result["scores"],
                name=f"{name} (AUC = {result['auc']:.4f})",
                ax=plt.gca(),
            )
        elif (
            (name == "Flow Matching3 vt")
            | (name == "Flow Matching3 vt einsum")
            | (name == "Flow Matching3 ode")
        ):
            # use the third dataset for this model
            RocCurveDisplay.from_predictions(
                dataset3.test_labels,
                result["scores"],
                name=f"{name} (AUC = {result['auc']:.4f})",
                ax=plt.gca(),
            )
        elif (
            (name == "Flow Matching4 vt")
            | (name == "Flow Matching4 vt einsum")
            | (name == "Flow Matching4 ode")
        ):
            # use the fourth dataset for this model
            RocCurveDisplay.from_predictions(
                dataset4.test_labels,
                result["scores"],
                name=f"{name} (AUC = {result['auc']:.4f})",
                ax=plt.gca(),
            )
        else:
            RocCurveDisplay.from_predictions(
                dataset.test_labels,
                result["scores"],
                name=f"{name} (AUC = {result['auc']:.4f})",
                ax=plt.gca(),
            )

    plt.axvline(x=1e-5, color="red", linestyle="--", label="FPR = 1e-5")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([1e-6, 1])
    plt.ylim([1e-6, 1])
    plt.title("ROC Curves Comparison")
    plt.grid(True)
    plt.legend()
    os.makedirs(f"{flow_name}", exist_ok=True)
    plt.savefig(f"{path}/{flow_name}/rocs_comparison_{name}_epoch{epoch}.png")

    # Compare Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        if (
            (name == "Flow Matching2 vt")
            | (name == "Flow Matching2 vt einsum")
            | (name == "Flow Matching2 ode")
        ):
            # use the second dataset for this model
            precision, recall, _ = precision_recall_curve(
                dataset2.test_labels, result["scores"]
            )
            plt.plot(recall, precision, lw=2, label=f"{name} (AP = {result['ap']:.4f})")
        elif (
            (name == "Flow Matching3 vt")
            | (name == "Flow Matching3 vt einsum")
            | (name == "Flow Matching3 ode")
        ):
            # use the third dataset for this model
            precision, recall, _ = precision_recall_curve(
                dataset3.test_labels, result["scores"]
            )
            plt.plot(recall, precision, lw=2, label=f"{name} (AP = {result['ap']:.4f})")
        elif (
            (name == "Flow Matching4 vt")
            | (name == "Flow Matching4 vt einsum")
            | (name == "Flow Matching4 ode")
        ):
            # use the fourth dataset for this model
            precision, recall, _ = precision_recall_curve(
                dataset4.test_labels, result["scores"]
            )
            plt.plot(recall, precision, lw=2, label=f"{name} (AP = {result['ap']:.4f})")
        else:
            precision, recall, _ = precision_recall_curve(
                dataset.test_labels, result["scores"]
            )
            plt.plot(recall, precision, lw=2, label=f"{name} (AP = {result['ap']:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{path}/{flow_name}/pr_comparison_{name}_epoch{epoch}.png")

    # %%
    # Define dataset names and corresponding results prefixes/labels
    # dataset_info = [
    #     {"name": "A -> 4l", "prefix": "Flow Matching ", "labels": dataset.test_labels, "suffix_vt": "vt", "suffix_ode": "ode"},
    #     {"name": "LQ -> b tau", "prefix": "Flow Matching2 ", "labels": dataset2.test_labels, "suffix_vt": "vt", "suffix_ode": "ode"},
    #     {"name": "h0 -> tau tau", "prefix": "Flow Matching3 ", "labels": dataset3.test_labels, "suffix_vt": "vt", "suffix_ode": "ode"},
    #     {"name": "h+- -> tau nu", "prefix": "Flow Matching4 ", "labels": dataset4.test_labels, "suffix_vt": "vt", "suffix_ode": "ode"},
    #     # Add entries for models trained with type index if they exist and need plotting
    #     # {"name": "A -> 4l (with type idx)", "prefix": "Flow Matching ", "labels": dataset.test_labels, "suffix_vt": "vt with type idx", "suffix_ode": "ode with type idx"},
    #     # {"name": "LQ -> b tau (with type idx)", "prefix": "Flow Matching2 ", "labels": dataset2.test_labels, "suffix_vt": "vt with type idx", "suffix_ode": "ode with type idx"},
    # ]

    # for info in dataset_info:
    #     plt.figure(figsize=(10, 8))
    #     ax = plt.gca()
    #     dataset_name = info["name"]
    #     prefix = info["prefix"]
    #     labels = info["labels"]

    #     # Plot vt mode
    #     vt_key = prefix + info["suffix_vt"]
    #     if vt_key in results:
    #         result_vt = results[vt_key]
    #         RocCurveDisplay.from_predictions(
    #             labels,
    #             result_vt["scores"],
    #             name=f"Flow Matching VT (AUC = {result_vt['auc']:.4f})",
    #             ax=ax,
    #         )

    #     # Plot ode mode
    #     ode_key = prefix + info["suffix_ode"]
    #     if ode_key in results:
    #         result_ode = results[ode_key]
    #         RocCurveDisplay.from_predictions(
    #             labels,
    #             result_ode["scores"],
    #             name=f"Flow Matching ODE (AUC = {result_ode['auc']:.4f})",
    #             ax=ax,
    #         )

    #     # Add Isolation Forest for the first dataset for comparison
    #     if dataset_name == "A -> 4l" and "Isolation Forest" in results:
    #          result_iso = results["Isolation Forest"]
    #          RocCurveDisplay.from_predictions(
    #              labels,
    #              result_iso["scores"],
    #              name=f"Isolation Forest (AUC = {result_iso['auc']:.4f})",
    #              ax=ax,
    #          )

    #     plt.axvline(x=1e-5, color="red", linestyle="--", label="FPR = 1e-5")
    #     plt.xscale("log")
    #     plt.yscale("log")
    #     plt.xlim([1e-6, 1])
    #     plt.ylim([1e-6, 1])
    #     plt.title(f"ROC Curves for {dataset_name}")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.show()

    # %%
    # Create a summary table
    print("==== Summary of Results ====")
    print(
        f"{'Model':<20} {'AUC-ROC':<10} {'Avg Precision':<15} {'Training Time (s)':<15} {'TPR@FPR 10^-5 %'}"
    )
    print("-" * 60)
    for m_name, result in results.items():
        if "TPR" not in result.keys():
            result["TPR"] = 0
        print(
            f"{m_name:<20} {result['auc']:<10.4f} {result['ap']:<15.4f} {result['time']:<15.2f} {result['TPR'] * 100}"
        )

    # save results dict to .csv
    my_csv_path = f"{path}/{flow_name}/results_{flow_name}_epoch{epoch}.csv"
    with open(my_csv_path, "w") as f:
        f.write("Model,AUC-ROC,Avg Precision,Training Time (s),TPR@FPR 10^-5%\n")
        for name, result in results.items():
            if "TPR" not in result.keys():
                result["TPR"] = 0
            f.write(
                f"{name},{result['auc']:.4f},{result['ap']:.4f},{result['time']:.2f},{result['TPR'] * 100}\n"
            )
