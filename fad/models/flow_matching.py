import torch
import numpy as np
import time
from torch.distributions import Independent, Normal
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from .base import BaseAnomalyDetector
from .NNs import MLP

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

class FlowMatchingAnomalyDetector(BaseAnomalyDetector):
    """
    Anomaly detector based on Flow Matching.
    
    This detector transforms data to Gaussian space using Flow Matching and
    scores anomalies based on their likelihood in the Gaussian space.
    """
    
    def __init__(self, input_dim=2, hidden_dim=128, lr=0.001, batch_size=4096, 
                 iterations=20000, print_every=2000, device=None):
        """
        Initialize the Flow Matching anomaly detector.
        
        Args:
            input_dim: Dimension of input data
            hidden_dim: Hidden dimension of the MLP
            lr: Learning rate for optimizer
            batch_size: Batch size for training
            iterations: Number of training iterations
            print_every: Log frequency during training
            device: Device to use for computation
        """
        # Model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.iterations = iterations
        self.print_every = print_every
        
        # Initialize device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.vf = None
        self.wrapped_vf = None
        self.solver = None
        self.gaussian = None

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """
        Fit the Flow Matching model to the training data.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            **kwargs: Additional model-specific parameters
        """
        # Override parameters if provided in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Convert data to torch tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Initialize model components
        self.vf = MLP(input_dim=self.input_dim, time_dim=1, hidden_dim=self.hidden_dim).to(self.device)
        path = AffineProbPath(scheduler=CondOTScheduler())
        optim = torch.optim.Adam(self.vf.parameters(), lr=self.lr)
        
        # Training loop
        start_time = time.time()
        for i in range(self.iterations):
            optim.zero_grad()
            
            # Sample batch from data
            batch_indices = torch.randint(0, X_tensor.shape[0], (self.batch_size,))
            x_1 = X_tensor[batch_indices] if self.batch_size < X_tensor.shape[0] else X_tensor
            
            # Sample from Gaussian prior
            x_0 = torch.randn_like(x_1).to(self.device)
            
            # Sample time
            t = torch.rand(x_1.shape[0]).to(self.device)
            
            # Sample probability path
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            
            # Flow matching L2 loss
            loss = torch.pow(self.vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()
            
            # Optimizer step
            loss.backward()
            optim.step()
            
            # Log loss
            if (i+1) % self.print_every == 0:
                elapsed = time.time() - start_time
                print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} '.format(
                    i+1, elapsed*1000/self.print_every, loss.item()))
                start_time = time.time()
        
        # Wrap the model for the solver
        self.wrapped_vf = WrappedModel(self.vf)
        self.solver = ODESolver(velocity_model=self.wrapped_vf)
        
        # Initialize Gaussian distribution for scoring
        self.gaussian = Independent(Normal(torch.zeros(self.input_dim, device=self.device), 
                                        torch.ones(self.input_dim, device=self.device)), 1)

    def predict(self, X: np.ndarray, step_size=0.05, **kwargs) -> np.ndarray:
        """
        Predict anomaly scores for the input data based on the flow to Gaussian space.
        
        Args:
            X: Data of shape (n_samples, n_features)
            step_size: Step size for ODE solver
            **kwargs: Additional model-specific parameters
            
        Returns:
            np.ndarray: Anomaly scores where higher values indicate more anomalous samples
        """
        if self.solver is None or self.vf is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Convert data to torch tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set up reverse time grid (from 1 to 0)
        T = torch.tensor([1.0, 0.0], device=self.device)
        
        # Transform data to Gaussian space by solving the ODE in reverse
        with torch.no_grad():
            # Solve the ODE to get points in Gaussian space
            gaussian_samples = self.solver.sample(
                time_grid=T, 
                x_init=X_tensor, 
                method='midpoint', 
                step_size=step_size, 
                return_intermediates=False
            )
            
            # The samples are at t=0 (Gaussian space)
            x_gaussian = gaussian_samples[-1] if isinstance(gaussian_samples, list) else gaussian_samples
        
        # Calculate log density in Gaussian space
        log_density = self.gaussian.log_prob(x_gaussian)
        
        # Anomaly score is negative log density (lower density = higher anomaly score)
        anomaly_scores = -log_density.cpu().numpy()
        
        return anomaly_scores, x_gaussian.cpu().numpy()
