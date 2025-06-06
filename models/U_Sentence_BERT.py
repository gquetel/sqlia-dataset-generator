from pathlib import Path
import torch
import transformers
import pandas as pd
import numpy as np
from transformers import RobertaTokenizerFast
from sklearn.svm import OneClassSVM


class OCSVM_SecureBERT:
    def __init__(
        self,
        device: torch.device,
        bert_model: str = "ehsanaghaei/SecureBERT",
        batch_size: int = 16,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.bert_model = bert_model

        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter

        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.bert_model)
        self.rb_model = transformers.RobertaModel.from_pretrained(self.bert_model)
        self.rb_model.to(self.device)
        self.rb_model.eval()

        self.clf = None

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract embeddings from the RoBERTa model.

        Args:
            queries: List of SQL queries to process
            layer_idx: Which layer to extract embeddings from (-1 for pooler output)

        Returns:
            numpy array of embeddings
        """
        embeddings = []
        # Uses https://github.com/ehsanaghaei/SecureBERT?tab=readme-ov-file#how-to-use-securebert
        queries = df["full_query"].values
        with torch.no_grad():
            for query in queries:
                inputs = self.tokenizer(query, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.rb_model(**inputs, output_hidden_states=True)

                # Pooler_output represents the whole class.
                embedding = outputs.pooler_output
                embeddings.append(embedding.cpu().numpy().flatten())

        result_df = df.copy()
        result_df["embeddings"] = embeddings
        return result_df

    def train_model(self, df: pd.DataFrame):
        embeddings = self.preprocess(df)
        self.clf = OneClassSVM(
            nu=self.nu, kernel=self.kernel, gamma=self.gamma, max_iter=self.max_iter
        )
        self.clf.fit(embeddings)

    def get_scores(self, df: pd.DataFrame):
        """ Get scores from Dataset

        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        df_we = self.preprocess(df)
        embeddings = np.array(df_we["embeddings"].tolist())
        dists = self.clf.decision_function(embeddings)
        return (df["labels"].to_numpy(), dists)
