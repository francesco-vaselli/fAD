import numpy as np
import os
import yaml
import time
import itertools
import pandas as pd
import torch

from fad.data.loaders import _load_h5_challenge_dataset
from fad.data.preprocessing import Preprocessor, StandardScalerFunction
from fad.models.flow_matching import FlowMatchingAnomalyDetector
from fad.validation.metrics import evaluate_performance

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Data Loading and Preprocessing
path_bkg = "/eos/user/f/fvaselli/fAD/fad/data/ad_challenge/background_for_training.h5"
path_anom = (
    "/eos/user/f/fvaselli/fAD/fad/data/ad_challenge/Ato4l_lepFilter_13TeV_filtered.h5"
)
dataset = _load_h5_challenge_dataset(path_bkg, path_anom, n_train=3500000, n_test=60000)

path_anom2 = "/eos/user/f/fvaselli/fAD/fad/data/ad_challenge/leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5"
dataset2 = _load_h5_challenge_dataset(
    path_bkg, path_anom2, n_train=100000, n_test=400000
)

path_anom3 = (
    "/eos/user/f/fvaselli/fAD/fad/data/ad_challenge/hToTauTau_13TeV_PU20_filtered.h5"
)
dataset3 = _load_h5_challenge_dataset(
    path_bkg, path_anom3, n_train=100000, n_test=400000
)

path_anom4 = (
    "/eos/user/f/fvaselli/fAD/fad/data/ad_challenge/hChToTauNu_13TeV_PU20_filtered.h5"
)
dataset4 = _load_h5_challenge_dataset(
    path_bkg, path_anom4, n_train=100000, n_test=400000
)

preprocessor = Preprocessor([StandardScalerFunction()])
X_train = preprocessor.transform(dataset.train, fit=True)
X_test = preprocessor.transform(dataset.test)
X_test2 = preprocessor.transform(dataset2.test)
X_test3 = preprocessor.transform(dataset3.test)
X_test4 = preprocessor.transform(dataset4.test)

# 2. Hyperparameter Search Space
# Experiments with standard MLP (hidden_dim, num_layers)
param_grid_mlp = {
    "hidden_dim": [16, 32, 64],
    "num_layers": [2, 3, 4, 5, 6],
    "lr": [0.01, 0.001],
    "batch_size": [256, 1024, 2048],
    "alpha": [0, 1, 10],
}
keys, values = zip(*param_grid_mlp.items())
experiments_mlp = [dict(zip(keys, v)) for v in itertools.product(*values)]
for exp in experiments_mlp:
    exp["list_dims"] = None

# Experiments with custom MLP (list_dims)
param_grid_custom = {
    "list_dims": [[32, 64, 32], [32, 128, 32], [128, 128]],
    "lr": [0.01, 0.001],
    "batch_size": [256, 1024, 2048],
    "alpha": [0, 1, 10],
}
keys, values = zip(*param_grid_custom.items())
experiments_custom = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Combine all experiments
experiments = experiments_mlp  # + experiments_custom

all_results = []
best_model_score = -1
best_model_params = None
best_model_results = None

# 3. Run Search
for i, params in enumerate(experiments):
    print(f"--- Experiment {i + 1}/{len(experiments)} ---")
    print(f"Parameters: {params}")

    config_path = "../fad/models/configs/flow_matching.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config.update(params)
    config["iterations"] = 100

    if config.get("list_dims") is not None:
        list_dims_str = "_".join(map(str, config["list_dims"]))
        model_name = f"search_{i + 1}_custom_{list_dims_str}_{config['lr']}_{config['batch_size']}_{config['alpha']}"
    else:
        model_name = f"search_{i + 1}_{config['hidden_dim']}_{config['num_layers']}_{config['lr']}_{config['batch_size']}_{config['alpha']}"

    # drop the "name" field from the config
    config.pop("name")
    config.pop("input_dim")
    config.pop("device")

    flow_matching = FlowMatchingAnomalyDetector(
        input_dim=X_train.shape[1], name=model_name, device=device, **config
    )

    # This is a real run. We call fit and it will evaluate the model at the specified epochs.
    results = flow_matching.fit(
        X_train,
        mode="OT",
        reflow=False,
        eval_epochs=[5, 20, 50, 100],
        eval_path="hyperparameter_search_results",
        return_results=True,
        X_test=X_test,
        dataset=dataset,
        X_test2=X_test2,
        dataset2=dataset2,
        X_test3=X_test3,
        dataset3=dataset3,
        X_test4=X_test4,
        dataset4=dataset4,
    )

    # Process the results from the fit
    best_score_for_exp = -1
    # Find the best score achieved during this experiment (across all epochs)
    for row in results:
        if row["Model specs"].endswith("vt_einsum"):
            auc_a_str = row.get("AUC A", "0.0")
            auc_LQ_str = row.get("AUC LQ", "0.0")
            auc_h0_str = row.get("AUC h0", "0.0")
            auc_hplus_str = row.get("AUC h+", "0.0")
            tpr_lq_str = row.get("TPR LQ @ 1e-05 FPR", "0.0")
            tpr_h0_str = row.get("TPR h0 @ 1e-05 FPR", "0.0")
            tpr_hplus_str = row.get("TPR h+ @ 1e-05 FPR", "0.0")

            auc_a = float(auc_a_str)
            auc_LQ = float(auc_LQ_str)
            auc_h0 = float(auc_h0_str)
            auc_hplus = float(auc_hplus_str)
            tpr_lq = float(tpr_lq_str.replace("%", ""))
            tpr_h0 = float(tpr_h0_str.replace("%", ""))
            tpr_hplus = float(tpr_hplus_str.replace("%", ""))

            # Combine TPR and AUC for scoring
            current_score = (
                100 * tpr_lq
                + 100 * tpr_h0
                + 100 * tpr_hplus
                + auc_a
                + auc_LQ
                + auc_h0
                + auc_hplus
            )
            if current_score > best_score_for_exp:
                best_score_for_exp = current_score

    # Add the best score for the experiment to all rows of the experiment
    for row in results:
        row["score"] = best_score_for_exp

    if best_score_for_exp > best_model_score:
        best_model_score = best_score_for_exp
        best_model_params = params
        best_model_results = results

    all_results.extend(results)

# 4. Save all results
if all_results:
    df = pd.DataFrame(all_results)
    df = df.sort_values(by="score", ascending=False)
    os.makedirs("hyperparameter_search_results", exist_ok=True)
    df.to_csv("hyperparameter_search_results/all_model_results.csv", index=False)
    print(
        "All model results saved to hyperparameter_search_results/all_model_results.csv"
    )

# 5. Save best model results
if best_model_results:
    df_best = pd.DataFrame(best_model_results)
    df_best.to_csv("hyperparameter_search_results/best_model_results.csv", index=False)
    print(f"Best model parameters: {best_model_params}")
    print(
        "Best model results saved to hyperparameter_search_results/best_model_results.csv"
    )
