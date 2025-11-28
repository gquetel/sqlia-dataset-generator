import hashlib
import logging
from pathlib import Path
from typing import List
import numpy as np
import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import transformers

from constants import MyAutoEncoderTanh, ProjectPaths
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from transformers import RobertaTokenizerFast

logger = logging.getLogger(__name__)


class BaseSecureBERT:
    """Share logic for all SecureBERT models.

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        device: torch.device,
        project_paths: ProjectPaths,
        bert_model: str,
        batch_size: int,
    ):
        self.device = device
        # We require a project_path object to find cached embeddings.
        self.project_paths = project_paths
        self.bert_model = bert_model
        self.batch_size = batch_size

        # This way, the same embeddings are generated for the same sentence
        # Indeed, some layers are randomly initialized as the warning states.
        torch.manual_seed(2)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.bert_model)
        self.rb_model = transformers.RobertaModel.from_pretrained(self.bert_model)
        self.rb_model.to(self.device)
        self.rb_model.eval()

        self.model_name = None
        self.clf = None

    # Shared proprocessing functions
    def _cache_path(self, df: pd.DataFrame) -> str:
        hash_val = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()
        return os.path.join(
            self.project_paths.embeddings_path,
            f"embeddings-{hash_val}.pkl",
        )

    def _load_or_compute_embeddings(
        self, df: pd.DataFrame, batch_size: int
    ) -> List[np.ndarray]:

        cache_path = self._cache_path(df)
        if os.path.isfile(cache_path):
            logger.info(f"Loaded cached SBERT embeddings from {cache_path}")
            return pd.read_pickle(cache_path)

        queries = df["full_query"].values
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size].tolist()
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.rb_model(**inputs, output_hidden_states=True)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                embeddings.extend(batch_embeddings)

        pd.to_pickle(embeddings, cache_path)
        return embeddings

    def preprocess(self, df: pd.DataFrame, batch_size: int = 64):
        embeddings = self._load_or_compute_embeddings(df, batch_size)
        labels = df["label"].to_numpy()
        return embeddings, labels


class OCSVM_SecureBERT(BaseSecureBERT):
    def __init__(
        self,
        device: torch.device,
        project_paths: ProjectPaths,
        bert_model: str = "ehsanaghaei/SecureBERT",
        batch_size: int = 16,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
    ):
        super().__init__(device, project_paths, bert_model, batch_size)
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        embeddings, _ = self.preprocess(df=df)
        self.clf = OneClassSVM(
            nu=self.nu, kernel=self.kernel, gamma=self.gamma, max_iter=self.max_iter
        )
        self.clf.fit(embeddings)


class LOF_SecureBERT(BaseSecureBERT):
    def __init__(
        self,
        device: torch.device,
        project_paths: ProjectPaths,
        bert_model: str = "ehsanaghaei/SecureBERT",
        batch_size: int = 16,
        n_jobs: int = -1,
    ):
        super().__init__(device, project_paths, bert_model, batch_size)
        self.n_jobs = n_jobs

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        embeddings, _ = self.preprocess(df=df)

        self.clf = LocalOutlierFactor(n_jobs=self.n_jobs, novelty=True)
        self.clf.fit(embeddings)


class AutoEncoder_SecureBERT(BaseSecureBERT):
    def __init__(
        self,
        device: torch.device,
        project_paths: ProjectPaths,
        bert_model: str = "ehsanaghaei/SecureBERT",
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 16,
    ):
        super().__init__(device, project_paths, bert_model, batch_size)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def load_model(self, model_name: str):
        self.model_name = model_name
        save_dir = self.project_paths.models_path
        model_path = f"{save_dir}{self.model_name}.pth"
        meta_path = f"{save_dir}{self.model_name}_meta.pth"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        if not Path(meta_path).exists():
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.learning_rate = metadata.get("learning_rate", self.learning_rate)
        self.epochs = metadata.get("epochs", self.epochs)
        self.batch_size = metadata.get("batch_size", self.batch_size)
        input_dim = metadata["input_dim"]

        self.clf = MyAutoEncoderTanh(input_dim=input_dim)
        self.clf.to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.clf.load_state_dict(state_dict)

        self.clf.eval()
        logger.info(f"Loaded AutoEncoder model from {model_path}")
    
    
    def save_model(self):
        save_dir = self.project_paths.models_path
        model_path = f"{save_dir}{self.model_name}.pth"
        meta_path = f"{save_dir}{self.model_name}_meta.pth"

        torch.save(self.clf.state_dict(), model_path)
        metadata = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "input_dim": next(self.clf.parameters()).shape[1],
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

    # TODO: Rename preprocess_for_preds into preprocess ?
    # Wrapper to fit to preprocessing_generic_ae function call.
    def preprocess_for_preds(self, df: pd.DataFrame):
        return self.preprocess(df=df)

    def X_to_tensor(self, X) -> torch.Tensor:
        """
        Used during testing.

        Args:
            df (_type_): _description_
        Returns:
            _type_: _description_
        """
        X_array = np.array(X)
        X_tensors = torch.FloatTensor(X_array).to(self.device)
        return X_tensors

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        embeddings, _ = self.preprocess(df=df)

        # Init variables for training + model
        input_dim = len(embeddings[0])
        # Because embeddings have values between -1 and 1, we use an autoencoder with tanh

        self.clf = MyAutoEncoderTanh(
            input_dim=input_dim,
        )
        self.clf.to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate)
        X_tensor = self.X_to_tensor(embeddings)

        self.clf.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size]
                batch = batch.to(self.device)

                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(X_tensor):.6f}"
            )
        self.save_model()


def preprocessing_sbert(
    model: OCSVM_SecureBERT | LOF_SecureBERT,
    df: pd.DataFrame,
    use_scaler: bool = False,
):
    """Preprocess queries from pandas DataFrame, returns embeddings and associated labels.

    Args:
        model (OCSVM_SecureBERT | LOF_SecureBERT | AutoEncoder_SecureBERT): _description_
        df (pd.DataFrame): _description_
        use_scaler (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # X is the embeddings, no further preprocessing required.
    X, labels = model.preprocess(df=df)
    return X, labels
