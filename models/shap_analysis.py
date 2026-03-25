import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import shap
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

logger = logging.getLogger(__name__)

SHAP_COMPATIBLE_MODELS = {"ae_li", "ae_gaur", "ae_loginov"}


# https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137/
def run_shap_analysis(model, config_name, background, X_explain, output_dir):
    if model.use_scaler and not hasattr(model._scaler, "clip"):
        # Some compatibility issues
        model._scaler.clip = False

    def predict_fn(X):
        scaled = model._scaler.transform(X) if model.use_scaler else X
        return -model.clf.decision_function(scaled)

    def _shap_chunk(chunk):
        # Each worker needs its own explainer — KernelExplainer is not thread-safe.
        local_explainer = shap.KernelExplainer(predict_fn, background)
        return local_explainer.shap_values(chunk, silent=True)

    n_jobs = max(4, int(0.8 * os.cpu_count()))
    chunks = [c for c in np.array_split(X_explain, n_jobs) if len(c) > 0]
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_shap_chunk)(chunk) for chunk in chunks
    )
    shap_values = np.vstack(results)

    feature_names = model.feature_columns or [
        f"f{i}" for i in range(X_explain.shape[1])
    ]
    explanation = shap.Explanation(
        values=shap_values, data=X_explain, feature_names=feature_names
    )

    plt.figure()
    shap.plots.bar(explanation, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"shap_bar_{config_name}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"SHAP bar plot saved to {output_dir}/shap_bar_{config_name}.png")

    plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"shap_beeswarm_{config_name}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    logger.info(
        f"SHAP beeswarm plot saved to {output_dir}/shap_beeswarm_{config_name}.png"
    )


def _load_model(model_type: str, model_path: str, device):
    from constants import DotDict, ProjectPaths
    from registry import build_model

    GENERIC = DotDict(
        {
            "RANDOM_SEED": 2,
            "BASE_PATH": str(SCRIPT_DIR),
            "METRICS_AVERAGE_METHOD": "binary",
        }
    )
    project_paths = ProjectPaths(GENERIC.BASE_PATH)
    model = build_model(
        config_name=model_type,
        GENERIC=GENERIC,
        device=device,
        project_paths=project_paths,
        cache_dir=project_paths.features_cache_path,
    )
    model.load_model(model_path)
    return model


def main():
    import torch
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # shap INFO level is simply too verbose.
    logging.getLogger("shap").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Standalone SHAP bar plot for a saved AE model"
    )
    parser.add_argument(
        "--model-type", required=True, choices=list(SHAP_COMPATIBLE_MODELS)
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to saved .pth file (without extension)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="CSV dataset used for background and explanation samples",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()

    n_background = 20 if args.testing else 200
    n_explain = 100 if args.testing else 1000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.model_type, args.model_path, device)

    df = pd.read_csv(args.dataset, low_memory=False)
    rng = np.random.default_rng(2)
    df_background = df[df["split"] == "train"].iloc[
        rng.choice((df["split"] == "train").sum(), n_background, replace=False)
    ]
    df_explain = df[df["split"] == "test"].iloc[
        rng.choice((df["split"] == "test").sum(), n_explain, replace=False)
    ]
    del df

    background, _ = model.preprocess_for_preds(df_background)
    X_explain, _ = model.preprocess_for_preds(df_explain)
    del df_background, df_explain
    background = np.asarray(background, dtype=np.float64)
    X_explain = np.asarray(X_explain, dtype=np.float64)

    identifier = Path(args.model_path).stem
    output_dir = os.path.join(args.output_dir, identifier)
    os.makedirs(output_dir, exist_ok=True)
    run_shap_analysis(
        model=model,
        config_name=args.model_type,
        background=background,
        X_explain=X_explain,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
