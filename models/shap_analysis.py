import logging
import os

import numpy as np
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

SHAP_COMPATIBLE_MODELS = {"ae_li", "ae_gaur", "ae_loginov"}


def run_shap_analysis(model, config_name, df_train, df_test, output_dir, testing=False):
    n_background = 30 if testing else 1000
    n_explain = 100 if testing else 5000

    X_train, _ = model.preprocess_for_preds()
    X_test, _ = model.preprocess_for_preds(df_test)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    rng = np.random.default_rng(2)
    background = X_train[rng.choice(len(X_train), n_background, replace=False)]
    X_explain = X_test[rng.choice(len(X_test), n_explain, replace=False)]

    def predict_fn(X):
        scaled = model._scaler.transform(X) if model.use_scaler else X
        return -model.clf.decision_function(scaled)

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain)

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
