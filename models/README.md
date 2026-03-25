## Training models

`training.py` is a script that uses the generated train and test dataset.csv splits to train ML models found in the litterature.

TODO

### Caching mechanism

To facilitate the experiments and prevent computing many times the same things, we implemented two caching mechanisms. One is dedicated to GAUR traces. The other, universal, is used for all features. In detail:

**1. GAUR traces cache** (internal to `gaur_sqld`)
Activated when `extractor.cache_dir` is set. Caches the GAUR traces produced by the instrumented MySQL server (`get_traces_from_df`). Shared across all models using the same GAUR mode (e.g. all `expert` models reuse the same traces regardless of which feature subset they select). `gaur_sqld` does not allow currently to override the location of these cache traces. A folder `cache` containing them will be created from where the python file is launched.

**2. Feature matrix cache** (outer cache in `training.py`)
Activated when `use_feature_cache=True`. Caches the final preprocessed `(X, labels)` numpy arrays to disk, keyed by `{model_type}-{split}-{df_hash}-{state_tag}`. Avoids re-running feature extraction on repeated runs. Notes (this should probably be normalized once the codebase is stable):

- Kakisim bypasses this cache because it manages its own internal embedding cache.
- For `GaurAblationExtractor`, the `state_tag` includes a hash of the `gaur_features` parameter so that the `lex`, `synt`, and `sem` variants each get their own cache file.
