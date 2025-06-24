import torch
import numpy as np
import time
from torch.distributions import Independent, Normal
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from .base import BaseAnomalyDetector
from .NNs import (
    MLP,
    ResNet,
    fast_MLP,
    MLP_wrapper_with_einsum,
    MLP_wrapper,
    MLP_wrapper_with_sum,
)
from ..validation.metrics import evaluate_performance


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)


class FlowMatchingAnomalyDetector(BaseAnomalyDetector):
    """
    Anomaly detector based on Flow Matching.

    This detector transforms data to Gaussian space using Flow Matching and
    scores anomalies based on their likelihood in the Gaussian space.
    """

    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        model_type="mlp",
        num_layers=4,
        list_dims=None,
        dropout_rate=0.0,
        use_batch_norm=False,
        lr=0.001,
        batch_size=4096,
        reflow_steps=10,
        reflow_batches=100,
        iterations=20000,
        print_every=2000,
        device=None,
        alpha=1,
        name="null",
    ):
        """
        Initialize the Flow Matching anomaly detector.

        Args:
            input_dim: Dimension of input data
            hidden_dim: Hidden dimension of the MLP
            model_type: Type of model to use ('mlp' or 'resnet')
            num_layers: Number of layers/blocks in the model
            dropout_rate: Dropout rate (0.0 means no dropout)
            use_batch_norm: Whether to use batch normalization
            lr: Learning rate for optimizer
            batch_size: Batch size for training
            iterations: Number of training iterations
            print_every: Log frequency during training
            device: Device to use for computation
        """
        # Model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type.lower()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.reflow_steps = reflow_steps
        self.reflow_batches = reflow_batches
        self.iterations = iterations
        self.print_every = print_every

        # Initialize device
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.alpha = alpha

        # Model components
        self.vf = None
        self.wrapped_vf = None
        self.solver = None
        self.gaussian = None

        # Initialize model based on model_type
        if self.model_type == "mlp":
            self.vf = MLP(
                input_dim=self.input_dim,
                time_dim=1,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
                list_dims=list_dims,
            ).to(self.device)
        elif self.model_type == "resnet":
            self.vf = ResNet(
                input_dim=self.input_dim,
                time_dim=1,
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_layers,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
            ).to(self.device)
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}, expected 'mlp' or 'resnet'"
            )
        self.name = name

    def fit(
        self,
        X: np.ndarray,
        mode="OT",
        reflow: bool = False,
        eval_epochs=[5, 20, 100],
        **kwargs,
    ) -> None:
        """
        Fit the Flow Matching model to the training data.

        Args:
            X: Training data of shape (n_samples, n_features)
            mode: the type of loss to use; "OT" for optimal transport loss, "rectified" for rectified loss
            **kwargs: Additional model-specific parameters
        """
        # Override parameters if provided in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Convert data to torch tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        path = AffineProbPath(scheduler=CondOTScheduler())
        optim = torch.optim.Adam(self.vf.parameters(), lr=self.lr)

        # Training loop
        start_time = time.time()
        for i in range(self.iterations):
            self.vf.train()
            total_batches = X_tensor.shape[0] // self.batch_size
            for j in range(total_batches):
                optim.zero_grad()
                # Sample batch from data
                batch = X_tensor[j * self.batch_size : (j + 1) * self.batch_size]
                x_1 = batch
                # Sample from Gaussian prior
                x_0 = torch.randn_like(x_1).to(self.device)

                # Sample time
                t = torch.pow(torch.rand(x_0.shape[0]), 1 / (1 + self.alpha)).type_as(
                    x_0
                )

                # Sample probability path
                path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

                # Flow matching L2 loss
                # mode can be "OT" or "rectified"
                if mode == "OT":
                    loss = torch.pow(
                        self.vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2
                    ).mean()
                elif mode == "rectified":
                    loss = torch.pow(
                        (self.vf(path_sample.x_t, path_sample.t) + x_0 - x_1), 2
                    ).mean()

                # Optimizer step
                loss.backward()
                optim.step()

            # Log loss
            if (i + 1) % self.print_every == 0:
                elapsed = time.time() - start_time
                print(
                    "| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ".format(
                        i + 1, elapsed * 1000 / self.print_every, loss.item()
                    )
                )
                start_time = time.time()

            if (i + 1) in eval_epochs:
                print("SELF NAME", self.name)
                evaluate_performance(self, self.name, ".", i + 1, **kwargs)

        # if reflow and mode == "rectified":
        #     # Reflow the model
        #     for j in range(self.reflow_steps):
        #         # save the current model
        #         self.save(f"reflow_{j}.pt")
        #         # load the model
        #         self.load(f"reflow_{j}.pt")
        #         old_model = self.vf
        #         loss = 0
        #         for k in range(self.reflow_batches):
        #             optim.zero_grad()
        #             # sample batch from model
        #             batch = old_model.sample(self.batch_size)
        #             x_1 = batch
        #             # sample from Gaussian prior
        #             x_0 = torch.randn_like(x_1).to(self.device)
        #             # sample time
        #             t = torch.rand(x_1.shape[0]).to(self.device)
        #             # sample probability path
        #             path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        #             # rectified loss
        #             loss = torch.pow(
        #                 (self.vf(path_sample.x_t, path_sample.t) + x_0 - x_1), 2
        #             ).mean()
        #             # optimizer step
        #             loss.backward()
        #             optim.step()
        #             loss += loss.item()

        #         # log loss every reflow_step
        #         print(
        #             "| reflow iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ".format(
        #                 j + 1,
        #                 elapsed * 1000 / self.print_every,
        #                 loss.item() / (self.reflow_batches + 1),  # average loss
        #             )
        #         )

    def sample(
        self,
        batch_size: np.ndarray,
        time_steps=100,
        step_size=0.05,
        solver: str = "euler",
        **kwargs,
    ) -> tuple:
        """ """
        if self.vf is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Wrap the model for the solver
        self.wrapped_vf = WrappedModel(self.vf)
        self.solver = ODESolver(velocity_model=self.wrapped_vf)

        # Initialize Gaussian distribution for scoring
        self.gaussian = Independent(
            Normal(
                torch.zeros(self.input_dim, device=self.device),
                torch.ones(self.input_dim, device=self.device),
            ),
            1,
        )

        # Set up reverse time grid (from 1 to 0) of len(time_steps)
        T = torch.linspace(0.0, 1.0, time_steps, device=self.device)

        # Transform data to Gaussian space by solving the ODE in reverse
        with torch.no_grad():
            # Process in batches if data is large
            data_samples = self.solver.sample(
                time_grid=T,
                x_init=self.gaussian.sample((batch_size,)).to(self.device),
                method=solver,
                step_size=step_size,
                return_intermediates=False,
            )

            x_sampled = (
                data_samples[-1] if isinstance(data_samples, list) else data_samples
            )

        x_sampled = x_sampled.cpu().numpy()

        return x_sampled

    def predict(
        self,
        X: np.ndarray,
        time_steps=100,
        step_size=0.05,
        mode: str = "ODE",
        log_density_calc: str = "manual",
        return_transformed_data=False,
        **kwargs,
    ) -> tuple:
        """
        Predict anomaly scores for the input data.

        Mode options:
        - "ODE": Uses the ODE solver to transform data to Gaussian space
        - "vt": Directly evaluates the model at time T=1 (anomalous data get less displaced)

        Args:
            X: Data of shape (n_samples, n_features)
            step_size: Step size for ODE solver when mode="ODE"
            mode: Prediction mode - "ODE" or "vt"
            **kwargs: Additional model-specific parameters

        Returns:
            tuple: (anomaly_scores, transformed_data) where anomaly_scores is an np.ndarray
                  with higher values indicating more anomalous samples
        """
        if self.vf is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        # Convert data to torch tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        if mode.lower() == "ode":
            # Wrap the model for the solver
            self.wrapped_vf = WrappedModel(self.vf)
            self.solver = ODESolver(velocity_model=self.wrapped_vf)

            # Initialize Gaussian distribution for scoring
            self.gaussian = Independent(
                Normal(
                    torch.zeros(self.input_dim, device=self.device),
                    torch.ones(self.input_dim, device=self.device),
                ),
                1,
            )

            # Set up reverse time grid (from 1 to 0) of len(time_steps)
            T = torch.linspace(1.0, 0.0, time_steps, device=self.device)
            print(f"time grid: {T}")

            # Transform data to Gaussian space by solving the ODE in reverse
            with torch.no_grad():
                # Process in batches if data is large
                batch_size = 10000
                if log_density_calc == "library":
                    if X_tensor.shape[0] > batch_size:
                        log_density = torch.zeros_like(X_tensor[:, 0])  # .view(-1, 1)
                        for i in range(0, X_tensor.shape[0], batch_size):
                            end_idx = min(i + batch_size, X_tensor.shape[0])
                            batch = X_tensor[i:end_idx]
                            _, exact_log_p = self.solver.compute_likelihood(
                                x_1=batch,
                                method="midpoint",
                                step_size=step_size,
                                exact_divergence=False,
                                log_p0=self.gaussian.log_prob,
                            )
                            log_density[i:end_idx] = exact_log_p
                    else:
                        # For smaller datasets, process all at once
                        _, exact_log_p = self.solver.compute_likelihood(
                            x_1=X_tensor,
                            method="midpoint",
                            step_size=step_size,
                            exact_divergence=False,
                            log_p0=self.gaussian.log_prob,
                        )
                        log_density = exact_log_p
                    if return_transformed_data:
                        raise (
                            ValueError(
                                "return_transformed_data is not supported for log_density_calc = library"
                            )
                        )

                if log_density_calc == "manual":
                    if X_tensor.shape[0] > batch_size:
                        # Initialize storage for all Gaussian samples
                        x_gaussian = torch.zeros_like(X_tensor)

                        # Process in batches
                        for i in range(0, X_tensor.shape[0], batch_size):
                            end_idx = min(i + batch_size, X_tensor.shape[0])
                            batch = X_tensor[i:end_idx]

                            # Solve the ODE for this batch
                            gaussian_batch = self.solver.sample(
                                time_grid=T,
                                x_init=batch,
                                method="euler",
                                step_size=step_size,
                                return_intermediates=False,
                            )

                            # Store the results
                            x_gaussian[i:end_idx] = (
                                gaussian_batch[-1]
                                if isinstance(gaussian_batch, list)
                                else gaussian_batch
                            )
                    else:
                        # For smaller datasets, process all at once
                        gaussian_samples = self.solver.sample(
                            time_grid=T,
                            x_init=X_tensor,
                            method="euler",
                            step_size=step_size,
                            return_intermediates=False,
                        )

                        # The samples are at t=0 (Gaussian space)
                        x_gaussian = (
                            gaussian_samples[-1]
                            if isinstance(gaussian_samples, list)
                            else gaussian_samples
                        )

                    # Calculate log density in Gaussian space
                    log_density = self.gaussian.log_prob(x_gaussian)
                    transformed_data = x_gaussian.cpu().numpy()

            # Anomaly score is negative log density (lower density = higher anomaly score)
            anomaly_scores = -log_density.cpu().numpy()

        elif mode.lower() == "vt":
            # Direct vector field evaluation at t=1
            with torch.no_grad():
                t = torch.ones(X_tensor.shape[0], device=self.device)
                # Get vector field at t=1
                vector_field = self.vf(X_tensor, t)
                # Use the magnitude of the vector field as anomaly score
                # Higher magnitude indicates more displacement needed, suggesting anomaly
                anomaly_scores = torch.norm(vector_field, dim=1).cpu().numpy()
                if return_transformed_data:
                    raise ValueError(
                        "return_transformed_data is not supported for mode vt"
                    )
        elif mode.lower() == "vt_einsum":
            with torch.no_grad():
                t = torch.ones(X_tensor.shape[0], device=self.device)
                # Get vector field at t=1
                vector_field = self.vf(X_tensor, t)
                # Use the magnitude of the vector field as anomaly score
                # Higher magnitude indicates more displacement needed, suggesting anomaly
                anomaly_scores = (
                    torch.einsum("ij,ij->i", vector_field, vector_field).cpu().numpy()
                )
                if return_transformed_data:
                    raise ValueError(
                        "return_transformed_data is not supported for mode vt"
                    )
        else:
            raise ValueError(f"Unknown mode: {mode}, expected 'ODE' or 'vt'")

        if return_transformed_data:
            return anomaly_scores, transformed_data
        else:
            return anomaly_scores

    def return_model_for_hls(self, mode="no_reduction"):
        if mode == "einsum":
            return MLP_wrapper_with_einsum(self.vf)
        elif mode == "sum":
            return MLP_wrapper_with_sum(self.vf)
        else:
            return MLP_wrapper(self.vf)

    def return_trajectories(
        self,
        X: np.ndarray,
        time_steps=100,
        step_size=0.05,
        mode: str = "ODE",
        return_vts=False,
        **kwargs,
    ):
        """similar to predict, but return the trajectories of the data points from data to gaussian instead of the anomaly scores

        Args:
            X (np.ndarray): _description_
            step_size (float, optional): _description_. Defaults to 0.05.
            mode (str, optional): _description_. Defaults to "ODE".

        """
        if self.vf is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        # Convert data to torch tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if mode.lower() == "ode":
            # Wrap the model for the solver
            # Wrap the model for the solver
            self.wrapped_vf = WrappedModel(self.vf)
            self.solver = ODESolver(velocity_model=self.wrapped_vf)

            # Initialize Gaussian distribution for scoring
            self.gaussian = Independent(
                Normal(
                    torch.zeros(self.input_dim, device=self.device),
                    torch.ones(self.input_dim, device=self.device),
                ),
                1,
            )

            # Set up reverse time grid (from 1 to 0)
            T = torch.linspace(1.0, 0.0, time_steps, device=self.device)

            # Transform data to Gaussian space by solving the ODE in reverse
            with torch.no_grad():
                # Process in batches if data is large
                batch_size = 10000
                if X_tensor.shape[0] > batch_size:
                    # Initialize storage for all Gaussian samples
                    x_gaussian_traj = torch.zeros(
                        (T.shape[0], X_tensor.shape[0], X_tensor.shape[1]),
                        device=self.device,
                    )
                    if return_vts:
                        vts = torch.zeros(
                            (T.shape[0], X_tensor.shape[0], X_tensor.shape[1]),
                            device=self.device,
                        )

                    # Process in batches
                    for i in range(0, X_tensor.shape[0], batch_size):
                        end_idx = min(i + batch_size, X_tensor.shape[0])
                        batch = X_tensor[i:end_idx]

                        # Solve the ODE for this batch
                        gaussian_batch = self.solver.sample(
                            time_grid=T,
                            x_init=batch,
                            method="euler",
                            step_size=step_size,
                            return_intermediates=True,
                        )

                        # Store the results
                        x_gaussian_traj[i:end_idx] = (
                            gaussian_batch[-1]
                            if isinstance(gaussian_batch, list)
                            else gaussian_batch
                        )
                else:
                    # For smaller datasets, process all at once
                    gaussian_samples = self.solver.sample(
                        time_grid=T,
                        x_init=X_tensor,
                        method="euler",
                        step_size=step_size,
                        return_intermediates=True,
                    )
                    print(gaussian_samples.shape)

                    # The samples are at t=0 (Gaussian space)
                    x_gaussian_traj = (
                        gaussian_samples[-1]
                        if isinstance(gaussian_samples, list)
                        else gaussian_samples
                    )
        return x_gaussian_traj.cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save the model to a file.
        """

        # Save the model state
        torch.save(self.vf.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the model from a file.
        """
        # Load the model state
        self.vf.load_state_dict(torch.load(path))
        self.vf.to(self.device)
        print(f"Model loaded from {path}")
        # Reinitialize the wrapped model and solver
        self.wrapped_vf = WrappedModel(self.vf)
        self.solver = ODESolver(velocity_model=self.wrapped_vf)
        self.gaussian = Independent(
            Normal(
                torch.zeros(self.input_dim, device=self.device),
                torch.ones(self.input_dim, device=self.device),
            ),
            1,
        )
        # Ensure the model is in evaluation mode
        self.vf.eval()
        self.wrapped_vf.eval()
        self.solver.eval()


class FlowMatchingDistiller(BaseAnomalyDetector):
    """distillation model for Flow Matching.
    This model will be trained on the pair (inputs, flow matching anomaly scores)
    The base model is again MLP/ResNet.
    No time input, only the input data.
    The model is trained to predict the flow matching anomaly score.

    Args:
        BaseAnomalyDetector (_type_): _description_
    """

    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        hidden_dim=128,
        model_type="mlp",
        num_layers=4,
        dropout_rate=0.0,
        use_batch_norm=False,
        lr=0.001,
        batch_size=4096,
        iterations=20000,
        print_every=2000,
        device=None,
    ):
        """
        Initialize the Flow Matching distillation model.

        Args:
            input_dim: Dimension of input data
            hidden_dim: Hidden dimension of the MLP
            model_type: Type of model to use ('mlp' or 'resnet')
            num_layers: Number of layers/blocks in the model
            dropout_rate: Dropout rate (0.0 means no dropout)
            use_batch_norm: Whether to use batch normalization
            lr: Learning rate for optimizer
            batch_size: Batch size for training
            iterations: Number of training iterations
            print_every: Log frequency during training
            device: Device to use for computation
        """
        # Model parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model_type = model_type.lower()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.iterations = iterations
        self.print_every = print_every
        self.loss_function = torch.nn.MSELoss()

        # Initialize device
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize model based on model_type
        if self.model_type == "mlp":
            self.model = MLP(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                time_dim=0,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
            ).to(self.device)
        elif self.model_type == "resnet":
            self.model = ResNet(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                time_dim=0,
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_layers,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm,
            ).to(self.device)
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}, expected 'mlp' or 'resnet'"
            )

    def fit(self, X: np.ndarray, scores: np.ndarray, **kwargs) -> None:
        """
        Fit the Flow Matching distillation model to the training data.

        Args:
            X: Training data of shape (n_samples, n_features)
            scores: Anomaly scores from Flow Matching of shape (n_samples,)
            **kwargs: Additional model-specific parameters
        """
        # Override parameters if provided in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        print(f"Training distillation model with {self.model_type}...")
        # Convert data to torch tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        scores_tensor = torch.tensor(scores, dtype=torch.float32).to(self.device)

        # Optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        start_time = time.time()
        for i in range(self.iterations):
            optim.zero_grad()

            # Sample batch from data
            batch_indices = torch.randint(0, X_tensor.shape[0], (self.batch_size,))
            x_batch = (
                X_tensor[batch_indices]
                if self.batch_size < X_tensor.shape[0]
                else X_tensor
            )
            y_batch = (
                scores_tensor[batch_indices]
                if self.batch_size < scores_tensor.shape[0]
                else scores_tensor
            )

            # Forward pass
            y_pred = self.model(x_batch)
            # Loss calculation (L2 loss)
            loss = self.loss_function(y_pred, y_batch)

            # Backward pass and optimization
            loss.backward()
            optim.step()

            # Log loss
            if (i + 1) % self.print_every == 0:
                elapsed = time.time() - start_time
                print(
                    "| iter {:6d} | {:5.2f} ms/step | loss {:8.5f} ".format(
                        i + 1, elapsed * 1000 / self.print_every, loss.item()
                    )
                )
                start_time = time.time()

    def predict(self, X, **kwargs):
        self.model.eval()
        # Convert data to torch tensor and move to device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        batch_size = 10000
        if X_tensor.shape[0] > batch_size:
            # Initialize storage for all anomaly scores
            anomalies = torch.zeros_like(X_tensor[:, 0]).view(-1, 1)
            # Process in batches
            for i in range(0, X_tensor.shape[0], batch_size):
                end_idx = min(i + batch_size, X_tensor.shape[0])
                batch = X_tensor[i:end_idx]

                # Forward pass
                y_pred = self.model(batch)
                # Store the results
                anomalies[i:end_idx] = y_pred
        else:
            # For smaller datasets, process all at once
            with torch.no_grad():
                anomalies = self.model(X_tensor)
        # Convert to numpy
        return anomalies.detach().cpu().numpy()

    def save(self, path: str) -> None:
        """
        Save the model to a file.
        """

        # Save the model state
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the model from a file.
        """
        # Load the model state
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
        # Ensure the model is in evaluation mode
        self.model.eval()
