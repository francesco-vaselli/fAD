# %% [markdown]
# # Anomaly Detection Challenge
#
# This notebook experiments with different anomaly detection algorithms on the HEP dataset.
# We'll compare Isolation Forest and Flow Matching approaches.

# %%
# Import necessary libraries
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
import time

# Import fAD components
from fad.data.loaders import _load_h5_challenge_dataset
from fad.data.preprocessing import Preprocessor, StandardScalerFunction, TSNEFunction
from fad.models.statistical import IsolationForestDetector
from fad.models.flow_matching import FlowMatchingAnomalyDetector
from fad.models.NNs import MLP_wrapper

# %%
import hls4ml

# %% [markdown]
# ## 1. Data Loading and Exploration

# %%
# Set paths to dataset files
path_bkg = (
    "/home/fvaselli/Documents/PHD/fAD/fad/data/ad_challenge/background_for_training.h5"
)
path_anom = "/home/fvaselli/Documents/PHD/fAD/fad/data/ad_challenge/Ato4l_lepFilter_13TeV_filtered.h5"

# Load challenge datasets
print("Loading datasets...")
dataset = _load_h5_challenge_dataset(path_bkg, path_anom, n_train=3500000, n_test=60000)

# Print dataset information
print(f"Training set shape: {dataset.train.shape}")
print(f"Test set shape: {dataset.test.shape}")
print(f"Number of anomalies in test set: {np.sum(dataset.test_labels)}")

# dataset 2 with different anomaly process
path_anom2 = "/home/fvaselli/Documents/PHD/fAD/fad/data/ad_challenge/leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5"

# Load challenge datasets
print("Loading datasets...")
dataset2 = _load_h5_challenge_dataset(
    path_bkg, path_anom2, n_train=100000, n_test=400000
)
# Print dataset information
print(f"Training set shape: {dataset2.train.shape}")
print(f"Test set shape: {dataset2.test.shape}")
print(f"Number of anomalies in test set: {np.sum(dataset2.test_labels)}")

path_anom3 = "/home/fvaselli/Documents/PHD/fAD/fad/data/ad_challenge/hToTauTau_13TeV_PU20_filtered.h5"
# Load challenge datasets
print("Loading datasets...")
dataset3 = _load_h5_challenge_dataset(
    path_bkg, path_anom3, n_train=100000, n_test=400000
)
# Print dataset information
print(f"Training set shape: {dataset3.train.shape}")
print(f"Test set shape: {dataset3.test.shape}")
print(f"Number of anomalies in test set: {np.sum(dataset3.test_labels)}")

path_anom4 = "/home/fvaselli/Documents/PHD/fAD/fad/data/ad_challenge/hChToTauNu_13TeV_PU20_filtered.h5"
# Load challenge datasets
print("Loading datasets...")
dataset4 = _load_h5_challenge_dataset(
    path_bkg, path_anom4, n_train=100000, n_test=400000
)
# Print dataset information
print(f"Training set shape: {dataset4.train.shape}")
print(f"Test set shape: {dataset4.test.shape}")
print(f"Number of anomalies in test set: {np.sum(dataset4.test_labels)}")


# %% [markdown]
# ## 2. Data Preprocessing

# %%
# Preprocess data with standardization
print("Preprocessing data...")
preprocessor = Preprocessor([StandardScalerFunction()])
X_train = preprocessor.transform(dataset.train, fit=True)
X_test = preprocessor.transform(dataset.test)
print("Done")
X_test2 = preprocessor.transform(dataset2.test)
X_test3 = preprocessor.transform(dataset3.test)
X_test4 = preprocessor.transform(dataset4.test)

# %% [markdown]
# ## 4. Train and Evaluate Models
#
# We'll train and evaluate two models:
# 1. Isolation Forest (traditional algorithm)
# 2. Flow Matching (advanced generative model)

# %%
# Dictionary to store results
results = {}


# %%
config_path = "../fad/models/configs/flow_matching.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# flow_matching = FlowMatchingAnomalyDetector(
#     input_dim=X_train.shape[1],
#     hidden_dim=config["hidden_dim"],
#     model_type=config["model_type"],
#     num_layers=config["num_layers"],
#     dropout_rate=config["dropout_rate"],
#     use_batch_norm=config["use_batch_norm"],
#     lr=config["lr"],
#     batch_size=config["batch_size"],
#     reflow_steps=config["reflow_steps"],
#     reflow_batches=config["reflow_batches"],
#     iterations=config["iterations"],
#     print_every=config["print_every"],
#     device=config["device"],
# )
# flow_matching.load("last_model.pt")

print("Training Flow Matching (this may take a while)...")
# load the model config from the config file
# use absolute path to avoid issues with relative paths: get my parent folder, then go to fad/models/configs/flow_matching.yaml
config_path = "../fad/models/configs/flow_matching.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
print(f"Config: {config['name']}")
start_time = time.time()
flow_matching = FlowMatchingAnomalyDetector(
    input_dim=X_train.shape[1],
    hidden_dim=config["hidden_dim"],
    model_type=config["model_type"],
    num_layers=config["num_layers"],
    # list_dims=[32, 32],
    dropout_rate=config["dropout_rate"],
    use_batch_norm=config["use_batch_norm"],
    lr=config["lr"],
    batch_size=config["batch_size"],
    reflow_steps=config["reflow_steps"],
    reflow_batches=config["reflow_batches"],
    iterations=config["iterations"],
    print_every=config["print_every"],
    device=config["device"],
    alpha=0,
    name=config["name"],
)
flow_matching.fit(
    X_train,
    mode="OT",
    reflow=False,
    eval_epochs=[-1, 5, 20, 100],
    **{
        "X_test": X_test,
        "dataset": dataset,
        "X_test2": X_test2,
        "dataset2": dataset2,
        "X_test3": X_test3,
        "dataset3": dataset3,
        "X_test4": X_test4,
        "dataset4": dataset4,
    },
)
flow_time = time.time() - start_time
print(f"Training time: {flow_time:.2f} seconds")
print(
    f"Number of parameters in the model:{sum(p.numel() for p in flow_matching.vf.parameters() if p.requires_grad)}"
)

# %%
