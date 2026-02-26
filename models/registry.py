"""Config-driven model registry.
We use a declarative MODEL_CONFIGS dict and a build_model() factory.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from base import BaseAutoEncoderModel, BaseExtractor, BaseLOF, BaseOCSVM

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    extractor_type: (
        str  # "li", "countvect", "sbert", "roberta", "kakisim", "w2v", "loginov"
    )
    model_type: str  # "ocsvm", "lof", "ae"
    use_scaler: bool = False
    display_name: str = ""
    hyperparams: dict[str, Any] = field(default_factory=dict)
    extractor_kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    # ---- Li ----
    "ocsvm_li": ModelConfig(
        extractor_type="li",
        model_type="ocsvm",
        use_scaler=True,
        display_name="Li and OCSVM-scaler",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=1000),
    ),
    "lof_li": ModelConfig(
        extractor_type="li",
        model_type="lof",
        use_scaler=True,
        display_name="Li and LOF-scaler",
    ),
    "ae_li": ModelConfig(
        extractor_type="li",
        model_type="ae",
        use_scaler=True,
        display_name="Li and AE-scaler",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
    ),
    # ---- CountVectorizer ----
    "ocsvm_cv": ModelConfig(
        extractor_type="countvect",
        model_type="ocsvm",
        use_scaler=False,
        display_name="CountVectorizer and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
    ),
    "lof_cv": ModelConfig(
        extractor_type="countvect",
        model_type="lof",
        use_scaler=False,
        display_name="CountVectorizer and LOF",
    ),
    "ae_cv": ModelConfig(
        extractor_type="countvect",
        model_type="ae",
        use_scaler=False,
        display_name="CountVectorizer and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=4096),
        extractor_kwargs=dict(max_features=20000),
    ),
    # ---- RoBERTa-base ----
    "ocsvm_roberta": ModelConfig(
        extractor_type="roberta",
        model_type="ocsvm",
        display_name="RoBERTa-base and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(batch_size=64),
    ),
    "lof_roberta": ModelConfig(
        extractor_type="roberta",
        model_type="lof",
        display_name="RoBERTa-base and LOF",
        extractor_kwargs=dict(batch_size=64),
    ),
    "ae_roberta": ModelConfig(
        extractor_type="roberta",
        model_type="ae",
        display_name="RoBERTa-base and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=512),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- SecureBERT ----
    "ocsvm_sbert": ModelConfig(
        extractor_type="sbert",
        model_type="ocsvm",
        use_scaler=False,
        display_name="SecureBERT and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(batch_size=64),
    ),
    "lof_sbert": ModelConfig(
        extractor_type="sbert",
        model_type="lof",
        use_scaler=False,
        display_name="SBERT and LOF",
        extractor_kwargs=dict(batch_size=64),
    ),
    "ae_sbert": ModelConfig(
        extractor_type="sbert",
        model_type="ae",
        use_scaler=False,
        display_name="SecureBERT and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=512),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- Kakisim (C-view) ----
    "ocsvm_kakisim_c": ModelConfig(
        extractor_type="kakisim",
        model_type="ocsvm",
        use_scaler=False,
        display_name="Kakisim-C and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(views=["C"]),
    ),
    "ae_kakisim_c": ModelConfig(
        extractor_type="kakisim",
        model_type="ae",
        use_scaler=False,
        display_name="Kakisim-C and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=64),
        extractor_kwargs=dict(views=["C"], min_df=1),
    ),
    # ---- W2V Mean Pool ----
    "ocsvm_w2v": ModelConfig(
        extractor_type="w2v",
        model_type="ocsvm",
        use_scaler=False,
        display_name="W2V-MeanPool and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(vector_size=256),
    ),
    "ae_w2v": ModelConfig(
        extractor_type="w2v",
        model_type="ae",
        use_scaler=False,
        display_name="W2V-MeanPool and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=64),
        extractor_kwargs=dict(vector_size=256),
    ),
    # ---- BiLSTM W2V ----
    "ae_bilstm_w2v": ModelConfig(
        extractor_type="bilstm_w2v",
        model_type="ae",
        use_scaler=False,
        display_name="BiLSTM-W2V and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=4096),
        extractor_kwargs=dict(w2v_vector_size=256, lstm_hidden_size=128),
    ),
    "ocsvm_bilstm_w2v": ModelConfig(
        extractor_type="bilstm_w2v",
        model_type="ocsvm",
        use_scaler=False,
        display_name="BiLSTM-W2V and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(w2v_vector_size=256, lstm_hidden_size=128),
    ),
    # ---- Loginov ----
    "ocsvm_loginov": ModelConfig(
        extractor_type="loginov",
        model_type="ocsvm",
        use_scaler=True,
        display_name="Loginov and OCSVM-scaler",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=1000),
    ),
    "ae_loginov": ModelConfig(
        extractor_type="loginov",
        model_type="ae",
        use_scaler=True,
        display_name="Loginov and AE-scaler",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
    ),
}


def _make_extractor(
    config: ModelConfig,
    device: torch.device = None,
    project_paths=None,
    cache_dir: str | None = None,
) -> BaseExtractor:
    """Instantiate the right extractor from config."""
    from extractors.li import LiExtractor
    from extractors.countvect import CountVectExtractor
    from extractors.sbert import SecureBERTExtractor
    from extractors.kakisim import KakisimExtractor
    from extractors.loginov import LoginovExtractor

    kwargs = dict(config.extractor_kwargs)

    if config.extractor_type == "li":
        return LiExtractor()

    if config.extractor_type == "countvect":
        return CountVectExtractor(**kwargs)

    if config.extractor_type == "sbert":
        ext = SecureBERTExtractor(
            device=device,
            embeddings_path=project_paths.embeddings_path,
            **kwargs,
        )
        return ext

    if config.extractor_type == "roberta":
        from extractors.roberta import RobertaExtractor

        return RobertaExtractor(
            device=device,
            embeddings_path=project_paths.embeddings_path,
            **kwargs,
        )

    if config.extractor_type == "kakisim":
        ext = KakisimExtractor(**kwargs)
        ext.cache_dir = cache_dir
        return ext

    if config.extractor_type == "w2v":
        from extractors.w2v import W2VMeanPoolExtractor

        ext = W2VMeanPoolExtractor(**kwargs)
        ext.cache_dir = cache_dir
        return ext

    if config.extractor_type == "bilstm_w2v":
        from extractors.bilstm_w2v import BiLSTMW2VExtractor

        ext = BiLSTMW2VExtractor(device=device, **kwargs)
        ext.cache_dir = cache_dir
        return ext

    if config.extractor_type == "loginov":
        return LoginovExtractor()

    raise ValueError(f"Unknown extractor type: {config.extractor_type}")


def _output_activation(config: ModelConfig) -> str:
    """Determine which AutoEncoder output activation to use from the config.

    Rule:
    - use_scaler=True  → sigmoid (features normalised to [0, 1])
    - use_scaler=False → relu (non-negative features)
    - sbert            → tanh (embeddings in [-1, 1])
    """
    if config.extractor_type in ("sbert", "roberta", "bilstm_w2v"):
        return "tanh"
    if config.use_scaler:
        return "sigmoid"
    return "relu"


def build_model(
    config_name: str,
    GENERIC,
    device: torch.device = None,
    project_paths=None,
    n_jobs: int = -1,
    cache_dir: str | None = None,
):
    """Factory: instantiate extractor + wrap in OCSVM/AE/LOF.

    Returns the wrapped model object (BaseOCSVM, BaseLOF, or BaseAutoEncoderModel).
    """
    config = MODEL_CONFIGS[config_name]
    extractor = _make_extractor(
        config, device=device, project_paths=project_paths, cache_dir=cache_dir
    )
    hp = config.hyperparams

    if config.model_type == "ocsvm":
        return BaseOCSVM(
            extractor=extractor,
            GENERIC=GENERIC,
            nu=hp.get("nu", 0.05),
            kernel=hp.get("kernel", "rbf"),
            gamma=hp.get("gamma", "scale"),
            max_iter=hp.get("max_iter", -1),
            use_scaler=config.use_scaler,
        )

    if config.model_type == "lof":
        return BaseLOF(
            extractor=extractor,
            GENERIC=GENERIC,
            n_jobs=n_jobs,
            use_scaler=config.use_scaler,
        )

    if config.model_type == "ae":
        model = BaseAutoEncoderModel(
            extractor=extractor,
            GENERIC=GENERIC,
            device=device,
            learning_rate=hp.get("lr", 0.001),
            epochs=hp.get("epochs", 100),
            batch_size=hp.get("batch_size", 64),
            use_scaler=config.use_scaler,
            output_activation=_output_activation(config),
        )
        return model

    raise ValueError(f"Unknown model type: {config.model_type}")


# ---- Scoring helpers ----


def decision_score_generic(model, X: np.ndarray):
    """Negate OCSVM/LOF decision_function so positive = anomalous."""
    return -model.clf.decision_function(X)


def decision_score_ae(model, X: np.ndarray):
    """Negate AE reconstruction-error scores so positive = anomalous."""
    return -model.clf.decision_function(X, is_tensor=True)


def preprocessing_generic_ae(model, df: pd.DataFrame, use_scaler: bool = False):
    """Preprocess for AE scoring: extract features → tensor."""
    X, labels = model.preprocess_for_preds(df=df)
    X_tensors = model.X_to_tensor(X)
    return X_tensors, labels


def preprocessing_sklearn(model, df: pd.DataFrame, use_scaler: bool = False):
    """Preprocess for OCSVM/LOF scoring: extract features → numpy."""
    X, labels = model.preprocess_for_preds(df=df)
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if use_scaler:
        X = model._scaler.transform(X)
    return X, labels


def get_preprocess_fn(config_name: str):
    """Return the appropriate preprocessing function for a model config."""
    config = MODEL_CONFIGS[config_name]
    if config.model_type == "ae":
        return preprocessing_generic_ae
    return preprocessing_sklearn


def get_score_fn(config_name: str):
    """Return the appropriate scoring function for a model config."""
    config = MODEL_CONFIGS[config_name]
    if config.model_type == "ae":
        return decision_score_ae
    return decision_score_generic
