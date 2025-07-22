from sklearn.neighbors import LocalOutlierFactor
from constants import MyAutoEncoderTanh
import torch
import transformers
import pandas as pd
import numpy as np
from transformers import RobertaTokenizerFast
from sklearn.svm import OneClassSVM

import torch.nn as nn 
import torch
import logging 

logger = logging.getLogger(__name__)

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
        self.model_name = None

        
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
            embeddings = []
            queries = df["full_query"].values

            # Let's do smaller batch_size than self.batch_size: GPU is already saturated with
            # low batch size and this prevent the memory from being full (and potentially
            # crash if someone is using the GPU).
            _p_batch_size = 64
            with torch.no_grad():
                for i in range(0, len(queries), _p_batch_size):
                    batch_queries = queries[i:i + _p_batch_size]
                    
                    inputs = self.tokenizer(
                        batch_queries.tolist(), 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True,  
                        max_length=512  
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.rb_model(**inputs, output_hidden_states=True)
                    batch_embeddings = outputs.pooler_output.cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            result_df = df.copy()
            result_df["embeddings"] = embeddings
            return result_df


    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pp = self.preprocess(df=df)

        embeddings = np.array(df_pp["embeddings"].tolist())
        self.clf = OneClassSVM(
            nu=self.nu, kernel=self.kernel, gamma=self.gamma, max_iter=self.max_iter
        )
        self.clf.fit(embeddings)

    def get_scores(self, df: pd.DataFrame):
        """Get scores from Dataset

        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        df_we = self.preprocess(df)
        embeddings = np.array(df_we["embeddings"].tolist())
        dists = self.clf.decision_function(embeddings)
        return (df["label"].to_numpy(), dists)


class LOF_SecureBERT:
    def __init__(
        self,
        device: torch.device,
        bert_model: str = "ehsanaghaei/SecureBERT",
        batch_size: int = 16,
        n_jobs: int = -1,
    ):
        self.device = device
        self.batch_size = batch_size
        self.bert_model = bert_model

        self.n_jobs = n_jobs

        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.bert_model)
        self.rb_model = transformers.RobertaModel.from_pretrained(self.bert_model)
        self.rb_model.to(self.device)
        self.rb_model.eval()

        self.clf = None
        self.model_name = None

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
            embeddings = []
            queries = df["full_query"].values

            # Let's do smaller batch_size than self.batch_size: GPU is already saturated with
            # low batch size and this prevent the memory from being full (and potentially
            # crash if someone is using the GPU).
            _p_batch_size = 64
            with torch.no_grad():
                for i in range(0, len(queries), _p_batch_size):
                    batch_queries = queries[i:i + _p_batch_size]
                    
                    inputs = self.tokenizer(
                        batch_queries.tolist(), 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True,  
                        max_length=512  
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.rb_model(**inputs, output_hidden_states=True)
                    batch_embeddings = outputs.pooler_output.cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            result_df = df.copy()
            result_df["embeddings"] = embeddings
            return result_df


    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pp = self.preprocess(df=df)

        embeddings = np.array(df_pp["embeddings"].tolist())
        self.clf = LocalOutlierFactor(n_jobs=self.n_jobs, novelty=True)
        self.clf.fit(embeddings)

    def get_scores(self, df: pd.DataFrame):
        """Get scores from Dataset

        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        df_we = self.preprocess(df)
        embeddings = np.array(df_we["embeddings"].tolist())
        dists = self.clf.decision_function(embeddings)
        return (df["label"].to_numpy(), dists)

class AutoEncoder_SecureBERT:
    def __init__(
        self,
        device: torch.device,
        bert_model: str = "ehsanaghaei/SecureBERT",
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        self.device = device
        self.bert_model = bert_model


        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.bert_model)
        self.rb_model = transformers.RobertaModel.from_pretrained(self.bert_model)
        self.rb_model.to(self.device)
        self.rb_model.eval()

        self.clf = None
        self.model_name = None

   
    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        embeddings = []
        queries = df["full_query"].values

        # Let's do smaller batch_size than self.batch_size: GPU is already saturated with
        # low batch size and this prevent the memory from being full (and potentially
        # crash if someone is using the GPU).
        _p_batch_size = 64
        with torch.no_grad():
            for i in range(0, len(queries), _p_batch_size):
                batch_queries = queries[i:i + _p_batch_size]
                
                inputs = self.tokenizer(
                    batch_queries.tolist(), 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,  
                    max_length=512  
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.rb_model(**inputs, output_hidden_states=True)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        result_df = df.copy()
        result_df["embeddings"] = embeddings
        return result_df

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pp = self.preprocess(df=df)

        embeddings = np.array(df_pp["embeddings"].tolist())
        self.clf = LocalOutlierFactor(n_jobs=self.n_jobs, novelty=True)
        self.clf.fit(embeddings)

    def get_scores(self, df: pd.DataFrame):
        """Get scores from Dataset

        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        df_we = self.preprocess(df)
        embeddings = np.array(df_we["embeddings"].tolist())
        dists = self.clf.decision_function(embeddings)
        return (df["label"].to_numpy(), dists)
    
    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pp = self.preprocess(df=df)

        embeddings = np.array(df_pp["embeddings"].tolist())
    
        # Init variables for training + model
        input_dim = len(embeddings[0])
        # Because embeddings have values between -1 and 1, we use an autoencoder with tanh

        self.clf = MyAutoEncoderTanh(
            input_dim=input_dim,
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.clf.parameters(), lr=self.learning_rate
        )        
        train_data = torch.FloatTensor(embeddings)

        self.clf.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i : i + self.batch_size]

                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(train_data):.6f}"
            )
