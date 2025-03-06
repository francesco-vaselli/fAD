from setuptools import setup, find_packages

setup(
    name="fad",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "torch",
        "pyyaml",
        "jupyter",
        "flow_matching",
    ],
    author="Francesco Vaselli",
    author_email="your.email@example.com",
    description="Flow-based Anomaly Detection and Coverage Checks",
    keywords="anomaly-detection, coverage-checks, machine-learning",
    python_requires=">=3.8",
)
