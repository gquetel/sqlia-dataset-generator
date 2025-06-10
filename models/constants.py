from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DotDict(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"Attribute {attr} not found")

    def __setattr__(self, attr, value):
        self[attr] = value


class ProjectPaths:
    def __init__(
        self,
        base_path: str,
    ):
        self.base_path = base_path

    @property
    def dataset_path(self) -> str:
        return f"{self.base_path}/../dataset.csv"

    @property
    def output_path(self) -> str:
        path = f"{self.base_path}/output/"
        Path(path).mkdir(exist_ok=True, parents=True)
        return path
    
    @property
    def logs_path(self) -> str:
        path = f"{self.base_path}../logs/"
        Path(path).mkdir(exist_ok=True, parents=True)
        return path

    @property
    def models_path(self) -> str:
        path = f"{self.base_path}/cache/"
        Path(path).mkdir(exist_ok=True, parents=True)
        return path


class MyAutoEncoder(nn.Module):
    # From: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/linear-autoencoder/Simple_Autoencoder_Solution.ipynb
    def __init__(self, input_dim):
        super(MyAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self._inter_dim_1 = int(0.67 * input_dim)
        self._inter_dim_2 = int(0.33 * input_dim)
        logger.info(
            f"Autoencoder dimensions - input: {input_dim}, "
            f"inter1: {self._inter_dim_1}, inter2: {self._inter_dim_2}."
        )

        # encoder
        self.fc1 = nn.Linear(input_dim, self._inter_dim_1)
        self.fc2 = nn.Linear(self._inter_dim_1, self._inter_dim_2)

        ## decoder ##
        self.fc3 = nn.Linear(self._inter_dim_2, self._inter_dim_1)
        self.fc4 = nn.Linear(self._inter_dim_1, self.input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        encoded = F.relu(self.fc2(x))

        x = F.relu(self.fc3(encoded))
        # decoded = F.relu(self.fc4(x))
        decoded = F.sigmoid(self.fc4(x))
        return decoded

    def decision_function(
        self, features: np.ndarray, is_tensor: bool = False
    ) -> np.ndarray:
        """Compute anomaly scores using MSE for reconstruction error scores.

        We manually define this function to possess the same behavior as
        sklearn-based model to keep the same training functions.

        Args:
            features (np.ndarray):

        Returns:
            np.ndarray: _description_
        """
        if not is_tensor:
            test_data = torch.tensor(features, dtype=torch.float32)
        else:
            test_data = features

        self.eval()
        with torch.no_grad():
            recon = self(test_data)
            mse_per_sample = F.mse_loss(recon, test_data, reduction="none").mean(dim=1)
            recon_errors = mse_per_sample.numpy()
        scores = -recon_errors
        return scores

class MyAutoEncoderRelu(nn.Module):
    # From: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/linear-autoencoder/Simple_Autoencoder_Solution.ipynb
    def __init__(self, input_dim):
        super(MyAutoEncoderRelu, self).__init__()

        self.input_dim = input_dim
        self._inter_dim_1 = int(0.67 * input_dim)
        self._inter_dim_2 = int(0.33 * input_dim)
        logger.info(
            f"Autoencoder dimensions - input: {input_dim}, "
            f"inter1: {self._inter_dim_1}, inter2: {self._inter_dim_2}."
        )

        # encoder
        self.fc1 = nn.Linear(input_dim, self._inter_dim_1)
        self.fc2 = nn.Linear(self._inter_dim_1, self._inter_dim_2)

        ## decoder ##
        self.fc3 = nn.Linear(self._inter_dim_2, self._inter_dim_1)
        self.fc4 = nn.Linear(self._inter_dim_1, self.input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        encoded = F.relu(self.fc2(x))
        
        x = F.relu(self.fc3(encoded))
        decoded = F.relu(self.fc4(x))
        # decoded = F.sigmoid(self.fc4(x))
        return decoded

    def decision_function(
        self, features: np.ndarray, is_tensor: bool = False
    ) -> np.ndarray:
        """Compute anomaly scores using MSE for reconstruction error scores.

        We manually define this function to possess the same behavior as
        sklearn-based model to keep the same training functions.

        Args:
            features (np.ndarray):

        Returns:
            np.ndarray: _description_
        """
        if not is_tensor:
            test_data = torch.tensor(features, dtype=torch.float32)
        else:
            test_data = features

        self.eval()
        with torch.no_grad():
            recon = self(test_data)
            mse_per_sample = F.mse_loss(recon, test_data, reduction="none").mean(dim=1)
            recon_errors = mse_per_sample.numpy()
        scores = -recon_errors
        return scores