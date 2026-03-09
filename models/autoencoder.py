import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "relu": F.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
}


class AutoEncoder(nn.Module):
    """Parameterized autoencoder with configurable output activation.

    Replaces the former MyAutoEncoder (sigmoid), MyAutoEncoderRelu, and
    MyAutoEncoderTanh classes from constants.py.
    """

    def __init__(self, input_dim: int, output_activation: str = "sigmoid"):
        super().__init__()
        if output_activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown output_activation '{output_activation}', "
                f"choose from {list(_ACTIVATIONS)}"
            )

        self.input_dim = input_dim
        self.output_activation = output_activation
        self._act_fn = _ACTIVATIONS[output_activation]
        self._inter_dim_1 = int(0.67 * input_dim)
        self._inter_dim_2 = int(0.33 * input_dim)
        logger.info(
            f"Autoencoder dimensions - input: {input_dim}, "
            f"inter1: {self._inter_dim_1}, inter2: {self._inter_dim_2}."
        )

        self.fc1 = nn.Linear(input_dim, self._inter_dim_1)
        self.fc2 = nn.Linear(self._inter_dim_1, self._inter_dim_2)

        self.fc3 = nn.Linear(self._inter_dim_2, self._inter_dim_1)
        self.fc4 = nn.Linear(self._inter_dim_1, self.input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        encoded = F.relu(self.fc2(x))

        x = F.relu(self.fc3(encoded))
        decoded = self._act_fn(self.fc4(x))
        return decoded

    def decision_function(
        self, features: np.ndarray, is_tensor: bool = False
    ) -> np.ndarray:
        """Compute anomaly scores using MSE reconstruction error.

        Matches the sklearn decision_function convention: more negative = more
        anomalous.
        """
        if not is_tensor:
            test_data = torch.tensor(features, dtype=torch.float32)
        else:
            test_data = features

        self.eval()
        with torch.no_grad():
            recon = self(test_data)
            mse_per_sample = F.mse_loss(recon, test_data, reduction="none").mean(dim=1)
            recon_errors = mse_per_sample.cpu().numpy()
        return -recon_errors
