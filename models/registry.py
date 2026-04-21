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
    extractor_type: str  # "li", "countvect", "securebert", "securebert2", "modernbert", "roberta", "kakisim", "w2v", "loginov", "gaur", "codebert", "codet5", "flan_t5", "sentbert", "llm2vec", "qwen3_emb"
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
    "ae_cv": ModelConfig(
        extractor_type="countvect",
        model_type="ae",
        use_scaler=False,
        display_name="CountVectorizer and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=4096),
        extractor_kwargs=dict(max_features=10000),
    ),
    # ---- RoBERTa-base ----
    "ocsvm_roberta": ModelConfig(
        extractor_type="roberta",
        model_type="ocsvm",
        display_name="RoBERTa-base and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
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
    "ocsvm_securebert": ModelConfig(
        extractor_type="securebert",
        model_type="ocsvm",
        use_scaler=False,
        display_name="SecureBERT and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(batch_size=64),
    ),
    "lof_securebert": ModelConfig(
        extractor_type="securebert",
        model_type="lof",
        use_scaler=False,
        display_name="SecureBERT and LOF",
        extractor_kwargs=dict(batch_size=64),
    ),
    "ae_securebert": ModelConfig(
        extractor_type="securebert",
        model_type="ae",
        use_scaler=False,
        display_name="SecureBERT and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=512),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- ModernBERT-base ----
    "ocsvm_modernbert": ModelConfig(
        extractor_type="modernbert",
        model_type="ocsvm",
        use_scaler=False,
        display_name="ModernBERT-base and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(batch_size=64),
    ),
    "ae_modernbert": ModelConfig(
        extractor_type="modernbert",
        model_type="ae",
        use_scaler=False,
        display_name="ModernBERT-base and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=512),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- SecureBERT 2.0 (ModernBERT-based) ----
    "ocsvm_securebert2": ModelConfig(
        extractor_type="securebert2",
        model_type="ocsvm",
        use_scaler=False,
        display_name="SecureBERT2 and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(batch_size=64),
    ),
    "ae_securebert2": ModelConfig(
        extractor_type="securebert2",
        model_type="ae",
        use_scaler=False,
        display_name="SecureBERT2 and AE",
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
    # ---- GAUR ----
    "ocsvm_gaur": ModelConfig(
        extractor_type="gaur",
        model_type="ocsvm",
        use_scaler=True,
        display_name="GAUR expert+hybrid and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=1000),
        extractor_kwargs=dict(use_hybrid=True, mode="expert"),
    ),
    "ae_gaur": ModelConfig(
        extractor_type="gaur",
        model_type="ae",
        use_scaler=True,
        display_name="GAUR expert+hybrid and AE",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
        extractor_kwargs=dict(use_hybrid=True, mode="expert"),
    ),
    "ocsvm_gaur_chatgpt": ModelConfig(
        extractor_type="gaur",
        model_type="ocsvm",
        use_scaler=True,
        display_name="GAUR chatgpt+hybrid and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=1000),
        extractor_kwargs=dict(use_hybrid=True, mode="chatgpt"),
    ),
    "ae_gaur_chatgpt": ModelConfig(
        extractor_type="gaur",
        model_type="ae",
        use_scaler=True,
        display_name="GAUR chatgpt+hybrid and AE",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
        extractor_kwargs=dict(use_hybrid=True, mode="chatgpt"),
    ),
    "ae_gaur_mistral": ModelConfig(
        extractor_type="gaur",
        model_type="ae",
        use_scaler=True,
        display_name="GAUR mistral+hybrid and AE",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
        extractor_kwargs=dict(use_hybrid=True, mode="mistral"),
    ),
    # ---- GAUR ablation (Li + GAUR feature subsets) ----
    "ae_li_gaur_chatgpt_sem": ModelConfig(
        extractor_type="gaur_ablation",
        model_type="ae",
        use_scaler=True,
        display_name="Li + GAUR chatgpt sem + AE",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
        extractor_kwargs=dict(gaur_features="SEMANTIC_TAGS", mode="chatgpt"),
    ),
    "ae_li_gaur_mistral_sem": ModelConfig(
        extractor_type="gaur_ablation",
        model_type="ae",
        use_scaler=True,
        display_name="Li + GAUR mistral sem + AE",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
        extractor_kwargs=dict(gaur_features="SEMANTIC_TAGS", mode="mistral"),
    ),
    "ae_li_gaur_lex": ModelConfig(
        extractor_type="gaur_ablation",
        model_type="ae",
        use_scaler=True,
        display_name="Li + GAUR Lex + AE",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
        extractor_kwargs=dict(
            gaur_features=["avg_c_sqlkywds", "max_c_sqlkywds", "min_c_sqlkywds"]
        ),
    ),
    "ae_li_gaur_synt": ModelConfig(
        extractor_type="gaur_ablation",
        model_type="ae",
        use_scaler=True,
        display_name="Li + GAUR Synt + AE",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
        extractor_kwargs=dict(
            gaur_features=[
                "n_terminal",
                "n_nonterminal",
                "is_syntax_error",
                "depth",
                "n_parser_invoc",
            ]
        ),
    ),
    "ae_li_gaur_sem": ModelConfig(
        extractor_type="gaur_ablation",
        model_type="ae",
        use_scaler=True,
        display_name="Li + GAUR Sem + AE",
        hyperparams=dict(lr=0.005, epochs=100, batch_size=8192),
        extractor_kwargs=dict(gaur_features="SEMANTIC_TAGS"),
    ),
    # ---- CodeBERT ----
    "ae_codebert": ModelConfig(
        extractor_type="codebert",
        model_type="ae",
        display_name="CodeBERT and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=64),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- CodeT5 ----
    "ae_codet5": ModelConfig(
        extractor_type="codet5",
        model_type="ae",
        display_name="CodeT5+ 110M Emb and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=64),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- Flan-T5 Small ----
    "ae_flan_t5": ModelConfig(
        extractor_type="flan_t5",
        model_type="ae",
        display_name="Flan-T5-Small and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=64),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- CodeBERT OCSVM ----
    "ocsvm_codebert": ModelConfig(
        extractor_type="codebert",
        model_type="ocsvm",
        display_name="CodeBERT and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- SentenceBERT (all-mpnet-base-v2) ----
    "ocsvm_sentbert": ModelConfig(
        extractor_type="sentbert",
        model_type="ocsvm",
        display_name="SentenceBERT-mpnet and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(batch_size=64),
    ),
    "ae_sentbert": ModelConfig(
        extractor_type="sentbert",
        model_type="ae",
        display_name="SentenceBERT-mpnet and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=64),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- LLM2Vec (Mistral-7B) ----
    "ae_llm2vec": ModelConfig(
        extractor_type="llm2vec",
        model_type="ae",
        display_name="LLM2Vec-Mistral and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=64),
        extractor_kwargs=dict(batch_size=64),
    ),
    # ---- Qwen3-Embedding-0.6B ----
    "ocsvm_qwen3_emb": ModelConfig(
        extractor_type="qwen3_emb",
        model_type="ocsvm",
        display_name="Qwen3-Emb-0.6B and OCSVM",
        hyperparams=dict(nu=0.05, kernel="rbf", gamma="scale", max_iter=10000),
        extractor_kwargs=dict(batch_size=16),
    ),
    "ae_qwen3_emb": ModelConfig(
        extractor_type="qwen3_emb",
        model_type="ae",
        display_name="Qwen3-Emb-0.6B and AE",
        hyperparams=dict(lr=0.001, epochs=100, batch_size=512),
        extractor_kwargs=dict(batch_size=16),
    ),
}


def _make_extractor(
    config: ModelConfig,
    device: torch.device = None,
    project_paths=None,
    cache_dir: str | None = None,
    no_cache: bool = False,
) -> BaseExtractor:
    """Instantiate the right extractor from config."""
    from extractors.li import LiExtractor
    from extractors.countvect import CountVectExtractor
    from extractors.securebert import SecureBERTExtractor
    from extractors.kakisim import KakisimExtractor
    from extractors.loginov import LoginovExtractor

    kwargs = dict(config.extractor_kwargs)
    embeddings_path = None if no_cache else project_paths.embeddings_path

    if config.extractor_type == "li":
        return LiExtractor()

    if config.extractor_type == "countvect":
        return CountVectExtractor(**kwargs)

    if config.extractor_type == "securebert":
        ext = SecureBERTExtractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )
        return ext

    if config.extractor_type == "securebert2":
        from extractors.securebert import SecureBERT2Extractor

        return SecureBERT2Extractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )

    if config.extractor_type == "modernbert":
        from extractors.modernbert import ModernBERTExtractor

        return ModernBERTExtractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )

    if config.extractor_type == "roberta":
        from extractors.roberta import RobertaExtractor

        return RobertaExtractor(
            device=device,
            embeddings_path=embeddings_path,
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

    if config.extractor_type == "loginov":
        return LoginovExtractor()

    if config.extractor_type == "gaur":
        from extractors.gaur import GaurExtractor

        ext = GaurExtractor(**kwargs)
        ext.cache_dir = cache_dir
        return ext

    if config.extractor_type == "gaur_ablation":
        from extractors.gaur_ablation import GaurAblationExtractor

        ext = GaurAblationExtractor(**kwargs)
        ext.cache_dir = cache_dir
        return ext

    if config.extractor_type == "codebert":
        from extractors.codebert import CodeBERTExtractor

        return CodeBERTExtractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )

    if config.extractor_type == "codet5":
        from extractors.codet5 import CodeT5Extractor

        return CodeT5Extractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )

    if config.extractor_type == "flan_t5":
        from extractors.flan_t5 import FlanT5Extractor

        return FlanT5Extractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )

    if config.extractor_type == "sentbert":
        from extractors.sentbert import SentBERTExtractor

        return SentBERTExtractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )

    if config.extractor_type == "llm2vec":
        from extractors.llm2vec_ext import LLM2VecExtractor

        return LLM2VecExtractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )

    if config.extractor_type == "qwen3_emb":
        from extractors.qwen3_emb import Qwen3EmbExtractor

        return Qwen3EmbExtractor(
            device=device,
            embeddings_path=embeddings_path,
            **kwargs,
        )

    raise ValueError(f"Unknown extractor type: {config.extractor_type}")


def _output_activation(config: ModelConfig) -> str:
    """Determine which AutoEncoder output activation to use from the config.

    Rule:
    - use_scaler=True  -> sigmoid (for features normalised to [0, 1])
    - use_scaler=False -> relu (for non-negative features)
    - securebert       -> tanh (pooler_output is tanh-activated, so [-1, 1])
    - securebert2      -> linear (raw CLS hidden state, unbounded, centre around 0)
    """
    if config.extractor_type in ("securebert2", "modernbert"):
        return "linear"
    if config.extractor_type in (
        "securebert",
        "roberta",
        "codebert",
        "codet5",
        "flan_t5",
        "sentbert",
        "llm2vec",
        "qwen3_emb",
    ):
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
    no_cache: bool = False,
):
    """Factory: instantiate extractor + wrap in OCSVM/AE/LOF.

    Returns the wrapped model object (BaseOCSVM, BaseLOF, or BaseAutoEncoderModel).
    """
    config = MODEL_CONFIGS[config_name]
    extractor = _make_extractor(
        config,
        device=device,
        project_paths=project_paths,
        cache_dir=cache_dir,
        no_cache=no_cache,
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


def decision_score_lodo(model, X: np.ndarray):
    """Negate OCSVM/LOF decision_function so positive = anomalous."""
    return -model.clf.decision_function(X)


def decision_score_ae(model, X: np.ndarray):
    """Negate AE reconstruction-error scores so positive = anomalous."""
    return -model.clf.decision_function(X, is_tensor=True)


def preprocessing_lodo_ae(model, df: pd.DataFrame, use_scaler: bool = False):
    """Preprocess for AE scoring: extract features → tensor.

    Returns a 3-tuple (X_tensors, labels, valid_index). valid_index is the pandas
    index of rows that survived preprocessing (may be a subset of df.index when
    the extractor drops rows, e.g. gaur dropping unparseable queries).
    """
    result = model.preprocess_for_preds(df=df)
    if len(result) == 3:
        X, labels, valid_index = result
    else:
        X, labels = result
        valid_index = df.index
    X_tensors = model.X_to_tensor(X)
    return X_tensors, labels, valid_index


def preprocessing_sklearn(model, df: pd.DataFrame, use_scaler: bool = False):
    """Preprocess for OCSVM/LOF scoring: extract features → numpy."""
    result = model.preprocess_for_preds(df=df)
    if len(result) == 3:
        X, labels, valid_index = result
    else:
        X, labels = result
        valid_index = df.index
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if use_scaler:
        X = model._scaler.transform(X)
    return X, labels, valid_index


def get_preprocess_fn(config_name: str):
    """Return the appropriate preprocessing function for a model config."""
    config = MODEL_CONFIGS[config_name]
    if config.model_type == "ae":
        return preprocessing_lodo_ae
    return preprocessing_sklearn


def get_score_fn(config_name: str):
    """Return the appropriate scoring function for a model config."""
    config = MODEL_CONFIGS[config_name]
    if config.model_type == "ae":
        return decision_score_ae
    return decision_score_lodo
