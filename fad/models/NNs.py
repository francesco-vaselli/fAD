import torch
from torch import nn, Tensor


# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Basic Residual Block
class ResidualBlock(nn.Module):
    def __init__(
        self, dim: int, dropout_rate: float = 0.0, use_batch_norm: bool = False
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(Swish())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))

        self.block = nn.Sequential(*layers)
        self.activation = Swish()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


# Model class
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim=None,
        time_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        list_dims=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()

        if list_dims is None:
            # Input layer
            layers = [nn.Linear(input_dim + time_dim, hidden_dim), self.relu]
            # Hidden layers
            for _ in range(
                num_layers - 2
            ):  # -2 because we have specific input and output layers
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.relu)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

            # Output layer
            layers.append(nn.Linear(hidden_dim, self.output_dim))

        else:
            # Input layer
            layers = [nn.Linear(input_dim + time_dim, list_dims[0]), self.relu]
            for i in range(0, len(list_dims[:-1])):
                layers.append(nn.Linear(list_dims[i], list_dims[i + 1]))
                layers.append(self.relu)

            # Output layer
            layers.append(nn.Linear(list_dims[-1], self.output_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        if t is not None:
            t = t.reshape(-1, self.time_dim).float()

            t = t.reshape(-1, 1).expand(x.shape[0], 1)
            h = torch.cat([x, t], dim=1)
        else:
            h = x
        output = self.main(h)
        # if self.output_dim == self.input_dim:
        #     output = output.reshape(*sz)
        # else:
        #     output = output.reshape(sz[0], self.output_dim)

        return output


# Model class
class fix_inputdim_MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim=None,
        time_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        list_dims=None,
    ):
        super().__init__()

        self.input_dim = input_dim + time_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()

        if list_dims is None:
            # Input layer
            layers = [nn.Linear(self.input_dim, hidden_dim), self.relu]
            # Hidden layers
            for _ in range(
                num_layers - 2
            ):  # -2 because we have specific input and output layers
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(self.relu)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

            # Output layer
            layers.append(nn.Linear(hidden_dim, self.output_dim))

        else:
            # Input layer
            layers = [nn.Linear(self.input_dim, list_dims[0]), self.relu]
            for i in range(0, len(list_dims[:-1])):
                layers.append(nn.Linear(list_dims[i], list_dims[i + 1]))
                layers.append(self.relu)

            # Output layer
            layers.append(nn.Linear(list_dims[-1], self.output_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        sz = x.size()
        # x = x.reshape(-1, self.input_dim-self.time_dim)
        if t is not None:
            t = t.reshape(-1, self.time_dim).float()

            t = t.reshape(-1, 1).expand(x.shape[0], 1)
            h = torch.cat([x, t], dim=1)
        else:
            h = x
        output = self.main(h)
        # if self.output_dim == self.input_dim:
        #     output = output.reshape(*sz)
        # else:
        #     output = output.reshape(sz[0], self.output_dim)

        return output


# Model class
class fast_MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim=None,
        time_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
    ):
        """move the checks in the constructor to avoid pointless if/else in the forward
        we are using this for flows in any case so the checks should always be true

        Args:
            input_dim (int, optional): _description_. Defaults to 2.
            output_dim (_type_, optional): _description_. Defaults to None.
            time_dim (int, optional): _description_. Defaults to 1.
            hidden_dim (int, optional): _description_. Defaults to 128.
            num_layers (int, optional): _description_. Defaults to 4.
            dropout_rate (float, optional): _description_. Defaults to 0.0.
            use_batch_norm (bool, optional): _description_. Defaults to False.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        if self.output_dim != self.input_dim:
            raise ValueError("expecting output dim equal to input dim for flows")

        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()

        # Input layer
        layers = [nn.Linear(input_dim + time_dim, hidden_dim), self.relu]

        # Hidden layers
        for _ in range(
            num_layers - 2
        ):  # -2 because we have specific input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.relu)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)

        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)

        output = self.main(h)
        output = output.reshape(*sz)
        return output


class MLP_wrapper(nn.Module):
    def __init__(self, mlp_model: MLP):
        super().__init__()
        # Copy the main sequential layers from the original model
        self.main = mlp_model.main
        self.input_dim = mlp_model.input_dim + 1
        self.output_dim = mlp_model.output_dim

    def load_from_mlp(self, mlp_model: MLP):
        """Load weights from an existing MLP model"""
        self.main.load_state_dict(mlp_model.main.state_dict())
        self.input_dim = mlp_model.input_dim
        self.output_dim = mlp_model.output_dim

    def forward(self, x: Tensor) -> Tensor:
        """Simple forward pass without time component and reshaping"""
        return self.main(x)


class MLP_wrapper_with_einsum(nn.Module):
    def __init__(self, mlp_model: MLP):
        super().__init__()
        # Copy the main sequential layers from the original model
        self.main = mlp_model.main
        self.input_dim = mlp_model.input_dim + 1
        self.output_dim = 1

    def load_from_mlp(self, mlp_model: MLP):
        """Load weights from an existing MLP model"""
        self.main.load_state_dict(mlp_model.main.state_dict())
        self.input_dim = mlp_model.input_dim
        self.output_dim = 1

    def forward(self, x: Tensor) -> Tensor:
        """using torch einsum to get the dot product"""
        out = self.main(x)
        return torch.einsum("ij,ij->i", out, out)


class MLP_wrapper_with_sum(nn.Module):
    def __init__(self, mlp_model: MLP):
        super().__init__()
        # Copy the main sequential layers from the original model
        self.main = mlp_model.main
        self.input_dim = mlp_model.input_dim + 1
        self.output_dim = 1

    def load_from_mlp(self, mlp_model: MLP):
        """Load weights from an existing MLP model"""
        self.main.load_state_dict(mlp_model.main.state_dict())
        self.input_dim = mlp_model.input_dim
        self.output_dim = 1

    def forward(self, x: Tensor) -> Tensor:
        """using torch einsum to get the dot product"""
        out = self.main(x).sum(dim=1)
        return out


# ResNet model class
class ResNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim=None,
        time_dim: int = 1,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim), Swish()
        )

        # Residual blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate, use_batch_norm)
                for _ in range(num_blocks)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        print(
            f"ResNet with input_dim={input_dim}, output_dim={output_dim}, time_dim={time_dim}, hidden_dim={hidden_dim}, num_blocks={num_blocks}"
        )

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        if t is not None:
            t = t.reshape(-1, self.time_dim).float()

            t = t.reshape(-1, 1).expand(x.shape[0], 1)
            h = torch.cat([x, t], dim=1)
        else:
            h = x
        # Input projection
        h = self.input_proj(h)
        # Apply residual blocks
        for block in self.blocks:
            h = block(h)

        # Output projection
        output = self.output_proj(h)

        if self.output_dim == self.input_dim:
            output = output.reshape(*sz)
        else:
            output = output.reshape(sz[0], self.output_dim)

        return output
