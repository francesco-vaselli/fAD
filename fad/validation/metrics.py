import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    RocCurveDisplay,
)


def _calculate_metrics(y_true, y_scores, target_fpr=1e-5):
    print(f"Mean score for normal samples: {np.mean(y_scores[y_true == 0]):.4f}")
    print(f"Std score for normal samples: {np.std(y_scores[y_true == 0]):.4f}")
    print(f"Mean score for anomalous samples: {np.mean(y_scores[y_true == 1]):.4f}")
    print(f"Std score for anomalous samples: {np.std(y_scores[y_true == 1]):.4f}")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    tpr_interpolated = np.interp(target_fpr, fpr, tpr)

    print("\n--- Accurate TPR Method: Interpolation ---")
    print(f"Target FPR: {target_fpr}")
    print(f"Interpolated TPR at target FPR: {tpr_interpolated:.6f}")
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    return auc, ap, tpr_interpolated


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
    return_results=False,
):
    target_fpr = 1e-5

    datasets_info = [
        {"name": "A", "data": dataset, "X_test": X_test},
        {"name": "LQ", "data": dataset2, "X_test": X_test2},
        {"name": "h0", "data": dataset3, "X_test": X_test3},
        {"name": "h+", "data": dataset4, "X_test": X_test4},
    ]

    modes_info = {
        "vt_einsum": {"mode": "vt_einsum"},
        "ode": {"mode": "ode", "time_steps": 2, "step_size": None},
    }

    all_results_for_plotting = {}
    csv_rows = []

    for mode_name, mode_params in modes_info.items():
        print(f"\n===== Evaluating mode: {mode_name} @ epoch {epoch} =====\n")

        model_spec = f"{flow_name}-{epoch}-{mode_name}"

        csv_row = {"Model specs": model_spec}

        for i, ds_info in enumerate(datasets_info):
            print(f"\n--- Evaluating on dataset: {ds_info['name']} ---")

            predict_params = mode_params.copy()
            if mode_name == "ode":
                if i == 0:
                    predict_params["return_transformed_data"] = True
                    scores, _ = flow_matching.predict(
                        ds_info["X_test"], **predict_params
                    )
                else:
                    predict_params["log_density_calc"] = "library"
                    scores = flow_matching.predict(ds_info["X_test"], **predict_params)
            else:
                scores = flow_matching.predict(ds_info["X_test"], **predict_params)

            auc, ap, tpr = _calculate_metrics(
                ds_info["data"].test_labels, scores, target_fpr
            )

            csv_row[f"AUC {ds_info['name']}"] = f"{auc:.4f}"
            csv_row[f"TPR {ds_info['name']} @ {target_fpr:.0e} FPR"] = (
                f"{tpr * 100:.4f}"
            )

            result_key_base = "Flow Matching"
            if i > 0:
                result_key_base += str(i + 1)

            result_key = f"{result_key_base} {mode_name}"

            all_results_for_plotting[result_key] = {
                "scores": scores,
                "auc": auc,
                "ap": ap,
                "labels": ds_info["data"].test_labels,
            }

        num_params = sum(
            p.numel() for p in flow_matching.vf.parameters() if p.requires_grad
        )
        csv_row["#params"] = num_params
        csv_rows.append(csv_row)

    # Plotting
    # Compare ROC curves
    if not return_results:
        plt.figure(figsize=(10, 8))
        for name, result in all_results_for_plotting.items():
            RocCurveDisplay.from_predictions(
                result["labels"],
                result["scores"],
                name=f"{name} (AUC = {result['auc']:.4f})",
                ax=plt.gca(),
            )
        plt.axvline(
            x=target_fpr, color="red", linestyle="--", label=f"FPR = {target_fpr:.0e}"
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim([1e-6, 1])
        plt.ylim([1e-6, 1])
        plt.title(f"ROC Curves Comparison at Epoch {epoch}")
        plt.grid(True)
        plt.legend()
        os.makedirs(f"{path}/{flow_name}", exist_ok=True)
        plt.savefig(f"{path}/{flow_name}/rocs_comparison_epoch{epoch}.png")
        plt.close()

        # Compare Precision-Recall curves
        plt.figure(figsize=(10, 8))
        for name, result in all_results_for_plotting.items():
            precision, recall, _ = precision_recall_curve(
                result["labels"], result["scores"]
            )
            plt.plot(recall, precision, lw=2, label=f"{name} (AP = {result['ap']:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curves at Epoch {epoch}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{path}/{flow_name}/pr_comparison_epoch{epoch}.png")
        plt.close()

    if return_results:
        return csv_rows

    # CSV writing
    my_csv_path = f"{path}/{flow_name}/results_{flow_name}.csv"

    header = ["Model specs"]
    for ds_info in datasets_info:
        header.append(f"AUC {ds_info['name']}")
        header.append(f"TPR {ds_info['name']} @ {target_fpr:.0e} FPR")
    header.append("#params")

    file_exists = os.path.isfile(my_csv_path)
    with open(my_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Results for epoch {epoch} saved to {my_csv_path}")
