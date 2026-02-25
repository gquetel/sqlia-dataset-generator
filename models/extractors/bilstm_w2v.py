"""Bi-LSTM + Attention over Word2Vec embeddings (whitespace tokenization).

Pre-trains a sequence autoencoder on normal SQL queries so the encoder
learns to compress variable-length token sequences into fixed 256-dim vectors.

Inspired by https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9638837
SQL Injection Detection Technology Based on BiLSTM-ATTENTION
"""

import hashlib
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from base import BaseExtractor
from cache_utils import hash_df, load_cache, save_cache

logger = logging.getLogger(__name__)


class BiLSTMSeqAutoEncoder(nn.Module):
    """Bi-LSTM encoder with attention + LSTM decoder for sequence reconstruction.

    The encoder produces a fixed-size context vector via attention-weighted
    pooling of Bi-LSTM hidden states.  The decoder reconstructs the original
    Word2Vec embedding sequence from that context vector.
    """

    def __init__(self, w2v_dim: int, hidden_size: int = 128):
        super().__init__()
        self.w2v_dim = w2v_dim
        self.hidden_size = hidden_size
        self.output_dim = hidden_size * 2  # bidirectional


        self.encoder = nn.LSTM(
            input_size=w2v_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.Linear(self.output_dim, 1)

        self.decoder = nn.LSTM(
            input_size=self.output_dim,
            hidden_size=w2v_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_proj = nn.Linear(w2v_dim, w2v_dim)

    def encode(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode variable-length embedding sequences into fixed-size vectors.

        Args:
            embeddings: (batch, max_seq_len, w2v_dim) padded embeddings
            lengths: (batch,) actual sequence lengths

        Returns:
            context: (batch, output_dim) attention-pooled encoder output
        """
        packed = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        enc_out, _ = self.encoder(packed)
        enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)
        # enc_out: (batch, max_seq_len, output_dim)

        # Attention weights with masking
        scores = self.attention(enc_out).squeeze(-1)  # (batch, max_seq_len)
        mask = torch.arange(scores.size(1), device=scores.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, max_seq_len, 1)

        context = (enc_out * weights).sum(dim=1)  # (batch, output_dim)
        return context

    def decode(self, context: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode context vector into reconstructed embedding sequence.

        Args:
            context: (batch, output_dim)
            seq_len: target sequence length

        Returns:
            reconstructed: (batch, seq_len, w2v_dim)
        """
        repeated = context.unsqueeze(1).expand(-1, seq_len, -1)
        dec_out, _ = self.decoder(repeated)
        return self.output_proj(dec_out)

    def forward(
        self, embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode.

        Returns:
            context: (batch, output_dim)
            reconstructed: (batch, max_seq_len, w2v_dim)
        """
        context = self.encode(embeddings, lengths)
        reconstructed = self.decode(context, embeddings.size(1))
        return context, reconstructed


def _compute_masked_mse(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """MSE loss only over non-padded positions."""
    mask = torch.arange(target.size(1), device=target.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(target)  # (batch, seq_len, w2v_dim)
    diff = (reconstructed - target) ** 2
    return (diff * mask).sum() / mask.sum()


class BiLSTMW2VExtractor(BaseExtractor):
    """Word2Vec + Bi-LSTM + Attention feature extractor.

    Tokenizes SQL queries by splitting on whitespace, trains Word2Vec on
    the resulting token sequences, then uses a pre-trained Bi-LSTM encoder
    with attention to produce fixed-size feature vectors.
    """

    def __init__(
        self,
        w2v_vector_size: int = 256,
        lstm_hidden_size: int = 128,
        lstm_epochs: int = 50,
        lstm_lr: float = 0.001,
        lstm_batch_size: int = 512,
        device: torch.device = None,
    ):
        self.w2v_vector_size = w2v_vector_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_epochs = lstm_epochs
        self.lstm_lr = lstm_lr
        self.lstm_batch_size = lstm_batch_size
        self.device = device or torch.device("cpu")

        self._w2v = None  # gensim Word2Vec model
        self._bilstm = None  # BiLSTMSeqAutoEncoder
        self._fitted = False
        self.cache_dir = None

    def _to_sequences(self, queries) -> list[list[str]]:
        """Tokenize queries by splitting on whitespace."""
        return [str(q).split() for q in queries]

    def _sequences_to_embeddings(
        self, sequences: list[list[str]]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Convert token sequences to Word2Vec embedding tensors.

        Returns:
            emb_list: list of (seq_len, w2v_dim) tensors
            lengths: (n_samples,) tensor of sequence lengths
        """
        wv = self._w2v.wv
        emb_list = []
        for tokens in sequences:
            vecs = [wv[t] for t in tokens if t in wv]
            if vecs:
                emb_list.append(torch.tensor(np.array(vecs), dtype=torch.float32))
            else:
                emb_list.append(torch.zeros(1, self.w2v_vector_size, dtype=torch.float32))
        lengths = torch.tensor([e.size(0) for e in emb_list], dtype=torch.long)
        return emb_list, lengths

    def prepare_for_training(self, df: pd.DataFrame):
        """Train Word2Vec and Bi-LSTM seq autoencoder on training data."""
        if self._fitted:
            return

        # Check cache
        if self.cache_dir:
            cache_key = f"bilstm_w2v-trained-{hash_df(df)}.pkl"
            cache_path = os.path.join(self.cache_dir, cache_key)
            cached = load_cache(cache_path)
            if cached is not None:
                self._w2v = cached["w2v"]
                self._bilstm = cached["bilstm"]
                self._bilstm.to(self.device)
                self._bilstm.eval()
                self._fitted = True
                logger.info("Loaded BiLSTM-W2V from cache.")
                return

        from gensim.models import Word2Vec

        queries = df["full_query"]
        logger.info("BiLSTM-W2V: tokenizing %d queries...", len(queries))
        sequences = self._to_sequences(queries)

        # Train Word2Vec
        logger.info("BiLSTM-W2V: training Word2Vec...")
        self._w2v = Word2Vec(
            sentences=sequences,
            vector_size=self.w2v_vector_size,
            window=5,
            min_count=2,
            workers=4,
        )

        # Convert to embeddings
        logger.info("BiLSTM-W2V: converting to embeddings...")
        emb_list, lengths = self._sequences_to_embeddings(sequences)

        # Train Bi-LSTM autoencoder
        self._bilstm = BiLSTMSeqAutoEncoder(
            w2v_dim=self.w2v_vector_size,
            hidden_size=self.lstm_hidden_size,
        ).to(self.device)

        self._train_bilstm(emb_list, lengths)
        self._bilstm.eval()
        self._fitted = True

        # Save to cache
        if self.cache_dir:
            bilstm_cpu = BiLSTMSeqAutoEncoder(
                w2v_dim=self.w2v_vector_size, hidden_size=self.lstm_hidden_size
            )
            bilstm_cpu.load_state_dict(self._bilstm.state_dict())
            save_cache(cache_path, {"w2v": self._w2v, "bilstm": bilstm_cpu})

    def _train_bilstm(self, emb_list: list[torch.Tensor], lengths: torch.Tensor):
        """Train the Bi-LSTM sequence autoencoder."""
        logger.info(
            "BiLSTM-W2V: training Bi-LSTM autoencoder (%d epochs, %d sequences)...",
            self.lstm_epochs,
            len(emb_list),
        )

        optimizer = torch.optim.Adam(self._bilstm.parameters(), lr=self.lstm_lr)
        n = len(emb_list)
        indices = np.arange(n)

        self._bilstm.train()
        for epoch in range(self.lstm_epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.lstm_batch_size):
                batch_idx = indices[start : start + self.lstm_batch_size]
                batch_embs = [emb_list[i] for i in batch_idx]
                batch_lengths = lengths[batch_idx]

                padded = pad_sequence(batch_embs, batch_first=True).to(self.device)
                batch_lengths = batch_lengths.to(self.device)

                optimizer.zero_grad()
                _, reconstructed = self._bilstm(padded, batch_lengths)
                loss = _compute_masked_mse(reconstructed, padded, batch_lengths)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            logger.debug("BiLSTM pre-training epoch %d/%d, loss: %.6f", epoch + 1, self.lstm_epochs, avg_loss)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract 256-dim features using frozen Bi-LSTM encoder + attention."""
        sequences = self._to_sequences(df["full_query"])
        emb_list, lengths = self._sequences_to_embeddings(sequences)

        self._bilstm.eval()
        all_contexts = []

        with torch.no_grad():
            n = len(emb_list)
            for start in range(0, n, self.lstm_batch_size):
                end = min(start + self.lstm_batch_size, n)
                batch_embs = emb_list[start:end]
                batch_lengths = lengths[start:end]

                padded = pad_sequence(batch_embs, batch_first=True).to(self.device)
                batch_lengths = batch_lengths.to(self.device)

                context = self._bilstm.encode(padded, batch_lengths)
                all_contexts.append(context.cpu().numpy())

        return np.concatenate(all_contexts, axis=0)

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        labels = df["label"].to_numpy()

        cache_path = None
        if self.cache_dir and self._w2v is not None:
            vocab = sorted(self._w2v.wv.key_to_index.keys())
            vocab_tag = hashlib.md5(str(vocab).encode()).hexdigest()[:8]
            cache_path = os.path.join(
                self.cache_dir,
                f"bilstm_w2v-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
            )
            cached = load_cache(cache_path)
            if cached is not None:
                return pd.DataFrame(cached), labels

        features = self.extract_features(df)
        if cache_path:
            save_cache(cache_path, features)
        return pd.DataFrame(features), labels

    def get_feature_names_out(self) -> np.ndarray:
        dim = self.lstm_hidden_size * 2
        return np.array([f"bilstm_w2v_{i}" for i in range(dim)])
