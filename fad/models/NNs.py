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
        time_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        # Input layer
        layers = [nn.Linear(input_dim + time_dim, hidden_dim), Swish()]

        # Hidden layers
        for _ in range(
            num_layers - 2
        ):  # -2 because we have specific input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(Swish())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.main = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)


# ResNet model class
class ResNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        time_dim: int = 1,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
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
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)

        # Input projection
        h = self.input_proj(h)

        # Apply residual blocks
        for block in self.blocks:
            h = block(h)

        # Output projection
        output = self.output_proj(h)

        return output.reshape(*sz)
