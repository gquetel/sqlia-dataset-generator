## Models

Feature extraction and unsupervised anomaly detection for SQL injection.

### Entry point

```bash
python3 training.py --dataset <path/to/dataset.csv> --models <names>
```

Key arguments:

| Argument | Description |
|---|---|
| `--dataset` | Path to the dataset CSV (required) |
| `--models` | Space-separated model names or groups (default: `all`) |
| `--testing` | Subsample to 5000 rows for quick smoke-tests |
| `--n-samples N` | Fix dataset size to N (deterministic, overrides `--testing`) |
| `--on-user-inputs` | Train on extracted user inputs instead of full queries |
| `--capture-insider` | Treat insider attacks as true positives (default: false negatives) |
| `--save-model-path` | Persist the trained model to disk (without extension) |
| `--skip-eval` | Skip test-set evaluation; only compute the validation threshold |
| `--no-feature-cache` | Disable the feature matrix disk cache |
| `--with-shap` | Run SHAP analysis after training (SHAP-compatible models only) |
| `--subfolder` | Write results under an output subfolder (multi-node runs) |
| `--debug` | Verbose logging |

### Available models

Models follow the naming convention `{detector}_{extractor}`. Detectors are `ocsvm`, `lof`, and `ae` (autoencoder). Pass a group name to `--models` to select all variants of an extractor.

| Group | Model keys |
|---|---|
| `li` | `ocsvm_li`, `lof_li`, `ae_li` |
| `cv` | `ocsvm_cv`, `ae_cv` |
| `loginov` | `ocsvm_loginov`, `ae_loginov` |
| `kakisim_c` | `ocsvm_kakisim_c`, `ae_kakisim_c` |
| `securebert` | `ocsvm_securebert`, `lof_securebert`, `ae_securebert` |
| `securebert2` | `ocsvm_securebert2`, `ae_securebert2` |
| `modernbert` | `ocsvm_modernbert`, `ae_modernbert` |
| `codebert` | `ocsvm_codebert`, `ae_codebert` |
| `sentbert` | `ocsvm_sentbert`, `ae_sentbert` |
| `qwen3_emb` | `ocsvm_qwen3_emb`, `ae_qwen3_emb` |
| `flan_t5` | `ae_flan_t5` |
| `llm2vec` | `ae_llm2vec` |
| `gaur` | `ocsvm_gaur`, `ae_gaur`, `ocsvm_gaur_chatgpt`, `ae_gaur_chatgpt`, `ae_gaur_mistral`, `ae_li_gaur_chatgpt_sem`, `ae_li_gaur_mistral_sem` |

GAUR ablation variants (`ae_li_gaur_lex`, `ae_li_gaur_synt`, `ae_li_gaur_sem`, …) combine Li features with individual GAUR feature subsets (lexical, syntactic, semantic).

### Architecture

Models are declared in `registry.py` as `ModelConfig` entries in `MODEL_CONFIGS`. `build_model()` instantiates the correct extractor + detector pair. `training.py` orchestrates data loading, training, threshold calibration on a held-out validation split (10% of train), and test-set evaluation.

The threshold is set at the 99.9th percentile of validation scores (0.1% FPR target).

### Caching

Two independent caches accelerate repeated runs:

**1. GAUR traces cache** (internal to `gaur_sqld`)\
Activated when `extractor.cache_dir` is set. Caches the GAUR traces produced by the instrumented MySQL server. Shared across all models using the same GAUR mode (e.g. all `expert` models reuse the same traces). A `cache/` folder is created in the working directory.

**2. Feature matrix cache** (outer cache in `training.py`)\
Activated by default; disable with `--no-feature-cache`. Caches the final preprocessed `(X, labels, valid_index)` arrays to disk, keyed by `{ModelClass}-{split}-{df_hash}-{state_tag}`. Notes:

- Kakisim bypasses this cache and manages its own internal embedding cache.
- Embedding-based extractors (SecureBERT, CodeBERT, etc.) maintain a separate embeddings cache keyed by query content.
