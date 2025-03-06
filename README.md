# fAD: Flow-based Anomaly Detection

> Contrary to what most people believe, fads are made, not born. -- Ken Hakuta

A Python package for anomaly detection and coverage checks using flow-based approaches.

## Features

- Data loading and preprocessing utilities
- Multiple anomaly detection models
- Validation and coverage check metrics
- Visualization tools

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd fAD

# Install the package in development mode
pip install -e .
```

## Usage

Basic example:

```python
from fad.data import loaders
from fad.models import statistical
from fad.validation import metrics

# Load data
data = loaders.load_dataset("path/to/data")

# Train model
model = statistical.GaussianMixture()
model.fit(data.train)

# Evaluate
anomalies = model.predict(data.test)
score = metrics.calculate_auc(data.test_labels, anomalies)
print(f"AUC score: {score:.4f}")
```

## Project Structure

- `fad/`: Main package code
- `config/`: Configuration files
- `notebooks/`: Jupyter notebooks for examples and exploration
- `tests/`: Unit tests
- `examples/`: Example scripts

## License

MIT
