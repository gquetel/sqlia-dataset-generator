"""Definition of ML models configuration."""

import os

# We force device on which training happens.
# device = torch.device("cuda:0" if USE_CUDA else "cpu") is not taken
# into account apparently...
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

import argparse
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import numpy as np
import random
import pandas as pd
import sys
import logging
import torch


from RF_Li import CustomRF_Li, CustomDT_Li
from RF_CountVect import CustomDT_CountVectorizer, CustomRF_CountVectorizer
from Sentence_BERT import CustomBERT

from explain import (
    get_recall_per_attack,
    plot_confusion_matrices_by_technique,
    plot_pca,
    plot_pr_curves_plt,
    plot_roc_curves_plt,
    plot_tree_clf,
    print_and_save_metrics,
)


# ------------ Custom Data Structures  ------------
class DotDict(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"Attribute {attr} not found")

    def __setattr__(self, attr, value):
        self[attr] = value


class ProjectPaths:
    def __init__(
        self,
        base_path: str,
    ):
        self.base_path = base_path

    @property
    def dataset_path(self) -> str:
        return f"{self.base_path}/../dataset.csv"

    @property
    def output_path(self) -> str:
        path = f"{self.base_path}/output/"
        Path(path).mkdir(exist_ok=True, parents=True)
        return path

    @property
    def logs_path(self) -> str:
        path = f"{self.base_path}../logs/"
        Path(path).mkdir(exist_ok=True, parents=True)
        return path

    @property
    def models_path(self) -> str:
        path = f"{self.base_path}/cache/"
        Path(path).mkdir(exist_ok=True, parents=True)
        return path


# ------------ Global variables  ------------

GENERIC = DotDict(
    {
        "RANDOM_SEED": 7,
        "BASE_PATH": os.path.join(os.path.dirname(__file__), ""),
        "METRICS_AVERAGE_METHOD": "binary",
    }
)

# Bootstrap a custom object path.
project_paths = ProjectPaths(GENERIC.BASE_PATH)
# project_paths = ProjectPaths("/home/gquetel/repos/sqlia-dataset/models")
logger = logging.getLogger(__name__)
training_results = []


def init_logging():
    lf = TimedRotatingFileHandler(
        project_paths.logs_path + "/training.log",
        when="midnight",
    )
    lf.setLevel(logging.INFO)
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(logging.INFO)

    lstdof = logging.Formatter(" %(message)s")
    lstdo.setFormatter(lstdof)
    logging.basicConfig(level=logging.INFO, handlers=[lf, lstdo])


def init_device() -> torch.device:
    """Initialize the device to use for experiments

    Returns:
        torch.device: device to use
    """
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    if USE_CUDA:
        logger.info("Using device: %s for experiments.", torch.cuda.get_device_name())
        torch.cuda.set_per_process_memory_fraction(0.99, 0)
    else:
        logger.critical("Using CPU for experiments.")
    return device


def init_args() -> argparse.Namespace:
    """Argsparse initializing function.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable the training and testing of GPU models.",
    )

    args = parser.parse_args()
    return args


def compute_metrics_li(model, df_test: pd.DataFrame, model_name: str):
    # 0 => pped + original columns
    df_pped, labels = model.preprocess_for_preds(df=df_test, drop_og_columns=False)
    # 1 => Probas (ppeds only)
    # Warning: having this variable and df_pped is VERY memory consuming
    # for CountVectorizer.
    df_pped_wout_og_cols = df_pped.drop(df_test.columns.to_list(), axis=1)
    probas = model.clf.predict_proba(df_pped_wout_og_cols.to_numpy())

    # 1.5 => PCA
    # plot_pca(
    #     df_pped_wout_og_cols.to_numpy(),
    #     labels,
    #     project_paths=project_paths,
    #     model_name=model_name,
    # )

    # 2 => Preds
    preds = np.argmax(probas, axis=1)
    df_pped["probas"] = probas.tolist()
    df_pped["preds"] = preds

    # 2.5 => Extract lines for which preds and labels are different and precise
    # Whether they are FP or FN.
    miss_mask = df_pped["preds"] != labels
    df_errors = df_pped.loc[miss_mask, ["full_query", "preds", "probas"]].copy()
    df_errors["labels"] = labels[miss_mask]
    df_errors["error_type"] = df_errors.apply(
        lambda row: (
            "FP"
            if row["preds"] == 1 and row["labels"] == 0
            else "FN" if row["preds"] == 0 and row["labels"] == 1 else "Other"
        ),
        axis=1,
    )
    folder_name = f"output/errors/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)
    df_errors.to_csv(f"{folder_name}{model_name}.csv", index=False)

    # 3 => print_and_save_metrics all
    training_results.append(
        print_and_save_metrics(
            df_pped["label"],
            df_pped["preds"],
            probas,
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_all",
        )
    )

    # 4 =>  print_and_save_metrics challenging only
    df_chall = df_pped[df_pped["template_split"] == "challenging"]

    logger.info("Metrics for challenging set only:")
    training_results.append(
        print_and_save_metrics(
            df_chall["label"],
            df_chall["preds"],
            np.array(df_chall["probas"].to_list()),
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_chall",
        )
    )

    # 5 =>  print_and_save_metrics original only
    df_og = df_pped[df_pped["template_split"] == "original"]
    logger.info("Metrics for original set only:")
    training_results.append(
        print_and_save_metrics(
            df_og["label"],
            df_og["preds"],
            np.array(df_og["probas"].to_list()),
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_origin",
        )
    )

    sdf = df_pped.sort_values(by="probas")[["probas", "label"]]
    sdf["probas"] = sdf["probas"].apply(lambda x: x[1])
    sdf.to_csv("output/random.csv")

    # 6 => Confusion matrix all
    # This function needs the dataframe with the preds and the original
    # columns information (attack_technique).
    # plot_confusion_matrices_by_technique(
    #     df_test=df_pped, model_name=model_name, project_paths=project_paths
    # )

    # 7 => Confusion matrix challenging
    # plot_confusion_matrices_by_technique(
    #     df_test=df_chall,
    #     model_name=model_name,
    #     project_paths=project_paths,
    #     suffix="_challenge",
    # )

    # 8 => Recall per technique
    # get_recall_per_attack(df=df_pped, model_name=model_name)
    # get_recall_per_attack(df=df_chall, model_name=model_name, suffix="_chall")

    # For AUC plot
    return (
        df_pped["label"],
        df_pped["probas"],
        df_pped["preds"],
        df_chall.index.tolist(),
    )


def compute_metrics_cv(model, df_test: pd.DataFrame, model_name: str):
    # The index of labels is also resetted when returned by preprocess_for_preds.
    # If we don't reset the one of df_test earlier we get indexes mismatch
    # The proper way to do this would probably be to avoid resetting the index, but
    # I don't want to waste any more time on this.
    df_test = df_test.copy().reset_index(drop=True)
    # 0 => pped
    f_matrix, labels = model.preprocess_for_preds(df=df_test, drop_og_columns=True)

    # 1 => Probas (ppeds only)
    probas = model.clf.predict_proba(f_matrix)

    # 2 => Preds
    preds = np.argmax(probas, axis=1)

    # 2.5 => Extract lines for which preds and labels are different and precise
    # Whether they are FP or FN.
    miss_mask = preds != labels
    df_errors = df_test.loc[miss_mask, ["full_query"]].copy()
    df_errors["preds"] = preds[miss_mask]
    df_errors["probas"] = probas[miss_mask].tolist()
    df_errors["labels"] = labels[miss_mask]
    df_errors["error_type"] = df_errors.apply(
        lambda row: (
            "FP"
            if row["preds"] == 1 and row["labels"] == 0
            else "FN" if row["preds"] == 0 and row["labels"] == 1 else "Other"
        ),
        axis=1,
    )
    folder_name = f"output/errors/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)
    df_errors.to_csv(f"{folder_name}{model_name}.csv", index=False)

    # 3 => print_and_save_metrics all
    training_results.append(
        print_and_save_metrics(
            labels,
            preds,
            probas,
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_all",
        )
    )

    # 4 =>  print_and_save_metrics challenging only
    # To get challenging only, we need to fetch their indices:
    # And then retrieve rows from labels and preds to be given
    # to print_and_save_metrics
    ids_chall = df_test[df_test["template_split"] == "challenging"].index.tolist()

    logger.info("Metrics for challenging set only:")

    training_results.append(
        print_and_save_metrics(
            labels[ids_chall],
            preds[ids_chall],
            probas[ids_chall],
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_chall",
        )
    )

    # 5 =>  print_and_save_metrics original only
    ids_og = df_test[df_test["template_split"] == "original"].index.tolist()
    logger.info("Metrics for original set only:")
    training_results.append(
        print_and_save_metrics(
            labels[ids_og],
            preds[ids_og],
            probas[ids_og],
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_origin",
        )
    )

    # 6 => Confusion matrix all
    # This function needs the dataframe with the preds and the original
    # columns information (attack_technique). I want to keep the same function
    # For both models, let's hack by creating the dataframe the function requires:
    _df = pd.DataFrame(
        {
            "attack_technique": df_test["attack_technique"],
            "label": labels,
            "preds": preds,
            "probas": list(probas),
            "template_split": df_test["template_split"],
        }
    )
    # plot_confusion_matrices_by_technique(
    #     df_test=_df, model_name=model_name, project_paths=project_paths
    # )

    _df_chall = _df[df_test["template_split"] == "challenging"]
    # 7 => Confusion matrix challenging
    # plot_confusion_matrices_by_technique(
    #     df_test=_df_chall,
    #     model_name=model_name,
    #     project_paths=project_paths,
    #     suffix="_challenge",
    # )

    # 8 => Recall per technique
    # get_recall_per_attack(df=_df, model_name=model_name)
    # get_recall_per_attack(df=_df_chall, model_name=model_name, suffix="_chall")

    # For AUC plot
    return _df["label"], _df["probas"], _df["preds"], ids_chall


def train_rf_li(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "Li-LSyn_RF"
    model = CustomRF_Li(GENERIC=GENERIC, max_depth=None)
    model.train_model(df=df_train, model_name=model_name, project_paths=project_paths)
    return compute_metrics_li(model=model, df_test=df_test, model_name=model_name)


def train_dt_li(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "Li-LSyn_DT"
    model = CustomDT_Li(GENERIC=GENERIC, max_depth=None)
    model.train_model(df=df_train, model_name=model_name)
    plot_tree_clf(model=model, project_paths=project_paths, max_depth=20)
    return compute_metrics_li(model=model, df_test=df_test, model_name=model_name)


def train_rf_cv(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "CountVectorizer_RF"
    model = CustomRF_CountVectorizer(GENERIC=GENERIC, max_depth=None, max_features=None)
    model.train_model(df=df_train, model_name=model_name)
    return compute_metrics_cv(model=model, df_test=df_test, model_name=model_name)


def train_dt_cv(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "CountVectorizer_DT"
    model = CustomDT_CountVectorizer(GENERIC=GENERIC, max_depth=None, max_features=None)
    model.train_model(df=df_train, model_name=model_name)
    plot_tree_clf(model=model, project_paths=project_paths, max_depth=20)
    return compute_metrics_cv(model=model, df_test=df_test, model_name=model_name)


def train_cpu_models(df_train: pd.DataFrame, df_test: pd.DataFrame):
    logger.info(
        f"Training - number of attacks {len(df_train[df_train['label'] == 1])}"
        f" and number of normals {len(df_train[df_train['label'] == 0])}"
    )
    logger.info(
        f"Testing - number of attacks {len(df_test[df_test['label'] == 1])}"
        f" and number of normals {len(df_test[df_test['label'] == 0])}"
    )

    # Train models and get their output.

    labels_cv, probas_cv, preds_cv, ids_chall_cv = train_rf_cv(df_train, df_test)
    labels_lidt, probas_lidt, preds_lidt, ids_chall_lidt = train_dt_li(
        df_train, df_test
    )
    labels_li, probas_li, preds_li, ids_chall_li = train_rf_li(df_train, df_test)
    _, _, _, _ = train_dt_cv(df_train=df_train, df_test=df_test)
    # We put all probas variable to same type / structure
    probas_li = np.array(probas_li.to_list())
    probas_cv = np.array(probas_cv.to_list())
    probas_lidt = np.array(probas_lidt.to_list())
    assert np.array_equal(labels_li, labels_cv)

    # Plot AUCPRC & AUROC for original dataset
    plot_pr_curves_plt(
        labels=labels_li,
        l_preds=[probas_li, probas_cv, probas_lidt],
        l_model_names=[
            "Manual Features and RF",
            "CountVectorizer and RF",
            "Manual Features and DT",
        ],
        project_paths=project_paths,
    )
    plot_roc_curves_plt(
        labels=labels_li,
        l_preds=[probas_li, probas_cv, probas_lidt],
        l_model_names=[
            "Manual Features and RF",
            "CountVectorizer and RF",
            "Manual Features and DT",
        ],
        project_paths=project_paths,
    )

    # All of my indices are fucked up:
    # - labels_li possess the same index as df_test, is untouched
    # - probas_li,probas_lidt &  probas_cv's index is reset because of the cast to a numpy array
    # So while the values are not the same for ids_chall_li & ids_chall_cv,
    # they still refer to the same samples. I don't have time to render the code cleaner.

    plot_pr_curves_plt(
        labels=labels_li[ids_chall_li],
        l_preds=[
            probas_li[ids_chall_cv],
            probas_cv[ids_chall_cv],
            probas_lidt[ids_chall_cv],
        ],
        l_model_names=[
            "Manual Features and RF",
            "CountVectorizer and RF",
            "Manual Features and DT",
        ],
        project_paths=project_paths,
        suffix="_chall",
    )
    plot_roc_curves_plt(
        labels=labels_cv[ids_chall_cv],
        l_preds=[
            probas_li[ids_chall_cv],
            probas_cv[ids_chall_cv],
            probas_lidt[ids_chall_cv],
        ],
        l_model_names=[
            "Manual Features and RF",
            "CountVectorizer and RF",
            "Manual Features and DT",
        ],
        project_paths=project_paths,
        suffix="_chall",
    )

    # Finally, save results to csv.
    dfres = pd.DataFrame(training_results)
    dfres.to_csv("output/results.csv", index=False)


def train_gpu_models(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "Sentence-BERT"
    save_model = False

    logger.info(
        f"Training - number of attacks {len(df_train[df_train['label'] == 1])}"
        f" and number of normals {len(df_train[df_train['label'] == 0])}"
    )
    logger.info(
        f"Testing - number of attacks {len(df_test[df_test['label'] == 1])}"
        f" and number of normals {len(df_test[df_test['label'] == 0])}"
    )
    device = init_device()

    myBERT = CustomBERT(
        device=device,
        model_name=model_name,
        bert_model="ehsanaghaei/SecureBERT",
        project_paths=project_paths,
        batch_size=32,
        lr=2e-5,
        epochs=3,
        weight_decay=0.01,
    )
    myBERT.set_dataloader_train(df_train)
    myBERT.train(save_models=save_model)

    # Now evaluate, we want to avoid having to infer multiple times the same
    # sample. Therefore we first collect all data for original split, then
    # challenging one. And we compute all stats from these.

    # 1 => probas for original test set
    _df_og = df_test[df_test["template_split"] == "original"]
    myBERT.set_dataloader_test(_df_og)
    labels_og, probas_og = myBERT.predict_probas()

    # 1 => probas for original test set
    _df_chall = df_test[df_test["template_split"] == "challenging"]
    myBERT.set_dataloader_test(_df_chall)
    labels_c, probas_c = myBERT.predict_probas()

    # 2 => preds
    preds_og = np.argmax(probas_og, axis=1)
    preds_c = np.argmax(probas_c, axis=1)

    # 3 => print_and_save_metrics all
    labels = np.concatenate([labels_og, labels_c])
    probas = np.concatenate([probas_og, probas_c])
    preds = np.concatenate([preds_og, preds_c])

    training_results.append(
        print_and_save_metrics(
            labels,
            preds,
            probas,
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_all",
        )
    )

    # 4 =>  print_and_save_metrics challenging only
    logger.info("Metrics for challenging set only:")
    training_results.append(
        print_and_save_metrics(
            labels_c,
            preds_c,
            probas_c,
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_chall",
        )
    )

    # 5 =>  print_and_save_metrics original only
    logger.info("Metrics for original set only:")
    training_results.append(
        print_and_save_metrics(
            labels_og,
            preds_og,
            probas_og,
            average=GENERIC.METRICS_AVERAGE_METHOD,
            model=f"{model_name}_origin",
        )
    )

    # We need to be sure that the order is kept.
    # print(labels[:10],df_test.iloc[:10]["label"])
    # assert(np.array_equal(labels,df_test["label"]))
    # ORDER IS NOT KEPT

    # 6 => Confusion matrix all
    # This function needs the dataframe with the preds and the original
    # columns information (attack_technique).
    # plot_confusion_matrices_by_technique(
    #     df_test=df_pped, model_name=model_name, project_paths=project_paths
    # )

    # 7 => Confusion matrix challenging
    # plot_confusion_matrices_by_technique(
    #     df_test=df_chall,
    #     model_name=model_name,
    #     project_paths=project_paths,
    #     suffix="_challenge",
    # )

    # 8 =>  Plot AUCPRC & AUROC for original dataset
    plot_pr_curves_plt(
        labels=labels,
        l_preds=[probas],
        l_model_names=[
            "Sentence-BERT",
        ],
        project_paths=project_paths,
    )
    plot_roc_curves_plt(
        labels=labels,
        l_preds=[probas],
        l_model_names=[
            "Sentence-BERT",
        ],
        project_paths=project_paths,
    )

    # 9 => Plot AUPRC and AUROC for challenging testing set
    plot_pr_curves_plt(
        labels=labels_c,
        l_preds=[probas_c],
        l_model_names=[
            "Sentence-BERT",
        ],
        project_paths=project_paths,
        suffix="_chall",
    )
    plot_roc_curves_plt(
        labels=labels_c,
        l_preds=[probas_c],
        l_model_names=[
            "Sentence-BERT",
        ],
        project_paths=project_paths,
        suffix="_chall",
    )


if __name__ == "__main__":
    init_logging()
    np.random.seed(GENERIC.RANDOM_SEED)
    random.seed(GENERIC.RANDOM_SEED)
    args = init_args()
    df = pd.read_csv(
        project_paths.dataset_path,
        # "/home/gquetel/repos/sqlia-dataset/dataset.csv",
        # dtype is specified to prevent a DtypeWarning
        dtype={
            "full_query": str,
            "label": int,
            "statement_type": str,
            "query_template_id": str,
            "user_inputs": str,
            "attack_id": str,
            "attack_technique": str,
            "split": str,
            "attack_status": str,
            "attack_stage": str,
            "tamper_method": str,
            "template_split": str,
        },
    )

    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    # df_train = df_train.sample(int(len(df_train) / 200))
    # df_test = df_test.sample(int(len(df_test) / 200))

    if args.gpu:
        train_gpu_models(df_train, df_test)
    else:
        train_cpu_models(df_train, df_test)
