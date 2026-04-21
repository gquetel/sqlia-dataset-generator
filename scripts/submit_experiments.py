#!/usr/bin/env python3
"""
Unified experiment launcher.

Generates SLURM scripts (or runs locally) for training and evaluating models
across lodo, in_domain, and wafamole experiment modes.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = REPO_ROOT / "scripts" / "generated"


SLURM_PROFILES = {
    "A100": {
        "partition": "A100",
        "gres": "gpu:1",
        "cpus": 16,
        "mem": "32G",
        "time": "4:00:00",
    },
    "A100_long": {
        "partition": "A100",
        "gres": "gpu:1",
        "cpus": 16,
        "mem": "64G",
        "time": "12:00:00",
    },
    "V100": {
        "partition": "V100",
        "gres": "gpu:1",
        "cpus": 16,
        "mem": "64G",
        "time": "8:00:00",
        "exclude": ["node43"],
    },
    "V100_long": {
        "partition": "V100",
        "gres": "gpu:1",
        "cpus": 16,
        "mem": "64G",
        "time": "24:00:00",
        "exclude": ["node43"],
    },
    "cpu": {
        "partition": "CPU",
        "gres": None,
        "cpus": 32,
        "mem": "64G",
        "time": "12:00:00",
    },
    "cpu_long": {
        "partition": "CPU",
        "gres": None,
        "cpus": 32,
        "mem": "64G",
        "time": "24:00:00",
    },
}

MODEL_PROFILES = {
    "ae_securebert": "V100",
    "ae_securebert2": "V100",
    "ae_modernbert": "A100",  # For diversity
    "ae_roberta": "A100",
    "ae_kakisim_c": "V100_long",
    "ae_li": "cpu",
    "ae_loginov": "cpu",
    "ae_gaur": "cpu",
    "ae_gaur_chatgpt": "cpu",
    "ae_gaur_mistral": "cpu",
    "ae_li_gaur_chatgpt_sem": "cpu",
    "ae_li_gaur_mistral_sem": "cpu",
    "ae_li_gaur_lex": "cpu",
    "ae_li_gaur_synt": "cpu",
    "ae_li_gaur_sem": "cpu",
    "ae_codebert": "A100",
    "ae_codet5": "A100_long",  # Can't use V100 using this requirements.txt
    "ae_flan_t5": "A100_long",
    "ae_sentbert": "V100",
    "ae_llm2vec": "A100_long",  # Requires 32Gb+ GPU, and 64+ memory
    "ae_qwen3_emb": "gpu_long",
    "ae_cv": "cpu_long",
    "ocsvm_li": "cpu",
    "ocsvm_loginov": "cpu",
    "ocsvm_gaur": "cpu",
    "ocsvm_securebert": "V100_long",
    "ocsvm_sentbert": "V100_long",
    "ocsvm_codebert": "A100_long",
}
DATASETS = {
    "A": "OurAirports",
    "B": "sakila",
    "C": "AdventureWorks",
    "D": "OHR",
    "E": "wafamole",
}

# LODO mode: leave-one-out (train on 3, test on all 4)
# Scenario N trains on the dataset that *excludes* dataset N from {A,B,C,D}
LODO_SCENARIOS = {
    1: {
        "train_label": "BCD",
        "train_dataset": "OurAirports",
        "test_labels": ["A", "B", "C", "D"],
    },
    2: {
        "train_label": "ACD",
        "train_dataset": "sakila",
        "test_labels": ["A", "B", "C", "D"],
    },
    3: {
        "train_label": "ABD",
        "train_dataset": "AdventureWorks",
        "test_labels": ["A", "B", "C", "D"],
    },
    4: {
        "train_label": "ABC",
        "train_dataset": "OHR",
        "test_labels": ["A", "B", "C", "D"],
    },
}

# In-domain mode: train on single dataset, test on all 4
IN_DOMAIN_SCENARIOS = {
    1: {
        "train_label": "A",
        "train_dataset": "OurAirports",
        "test_labels": ["A", "B", "C", "D"],
    },
    2: {
        "train_label": "B",
        "train_dataset": "sakila",
        "test_labels": ["A", "B", "C", "D"],
    },
    3: {
        "train_label": "C",
        "train_dataset": "AdventureWorks",
        "test_labels": ["A", "B", "C", "D"],
    },
    4: {
        "train_label": "D",
        "train_dataset": "OHR",
        "test_labels": ["A", "B", "C", "D"],
    },
}


# Concept-drift mode: train on origin templates, test on origin vs shifted
CONCEPT_DRIFT_SCENARIOS = {
    1: {"train_label": "A", "train_dataset": "OurAirports"},
    2: {"train_label": "B", "train_dataset": "sakila"},
    3: {"train_label": "C", "train_dataset": "AdventureWorks"},
    4: {"train_label": "D", "train_dataset": "OHR"},
}

CONCEPT_DRIFT_DATASETS_DIR = os.path.expanduser("~/datasets/concept-drift/")

GAUR_MODELS = {m for m in MODEL_PROFILES if "gaur" in m}


def get_profile(model: str, long: bool = False) -> dict:
    """Return the SLURM resource profile for a model.

    When long=True, upgrade to the _long variant of the profile if one exists.
    """
    profile_name = MODEL_PROFILES[model]
    if long:
        long_name = f"{profile_name}_long"
        if long_name in SLURM_PROFILES:
            profile_name = long_name
    return SLURM_PROFILES[profile_name]


CONDA_BASE = "~/miniconda3"
CONDA_ENV = "conda-env-3.12"
CONDA_ENV_LLM2VEC = "conda-env-3.12-llm2vec"
CONDA_ENV_CODET5 = "conda-env-3.12codet5"


def conda_env_for(model: str) -> str:
    """Return the conda environment name for a model."""
    if "llm2vec" in model:
        return CONDA_ENV_LLM2VEC
    if "codet5" in model:
        return CONDA_ENV_CODET5
    return CONDA_ENV


def lodo_filename(test_letter: str) -> str:
    """Return the CSV filename for a LODO scenario with `test_letter` as the held-out dataset."""
    train = "".join(sorted(l for l in "ABCD" if l != test_letter)).lower()
    return f"{train}-{test_letter.lower()}.csv"


def in_domain_filename(letter: str) -> str:
    """Return the CSV filename for an in-domain scenario."""
    l = letter.lower()
    return f"{l}-{l}.csv"


def log_dir_for(model: str, job_suffix: str) -> str:
    """Return the absolute log directory path."""
    return f"{REPO_ROOT}/logs/{model}/{job_suffix}"


def make_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def sbatch_header(
    model: str, job_suffix: str, log_path: str, long: bool = False
) -> str:
    """Generate SBATCH header lines for a model."""
    profile = get_profile(model, long=long)
    job_name = f"{model}_{job_suffix}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --error={log_path}",
        f"#SBATCH --partition={profile['partition']}",
    ]
    if profile["gres"]:
        lines.append(f"#SBATCH --gres={profile['gres']}")
    if profile.get("exclude"):
        lines.append(f"#SBATCH --exclude={','.join(profile['exclude'])}")
    lines.extend(
        [
            f"#SBATCH --cpus-per-task={profile['cpus']}",
            f"#SBATCH --mem={profile['mem']}",
            f"#SBATCH --time={profile['time']}",
        ]
    )
    return "\n".join(lines)


def env_setup(
    testing: bool,
    datasets_dir: str,
    log_dir: str,
    log_file: str,
    conda_env: str = CONDA_ENV,
    n_samples: int | None = None,
) -> str:
    """Generate environment setup lines with log directory creation and latest symlink."""
    n_samples_flag = f"--n-samples={n_samples}" if n_samples else ""
    return dedent(
        f"""\
        echo "Starting job on node: $(hostname)"
        echo "Job started at: $(date)"

        cd ~/repos/sqlia-dataset/
        source {CONDA_BASE}/etc/profile.d/conda.sh
        conda activate {conda_env}

        mkdir -p {log_dir}
        ln -sfn {log_file} {log_dir}/latest.log

        DATASETS_DIR={datasets_dir}
        TESTING_FLAG="{'--testing' if testing else ''}"
        MODEL_NAME_SUFFIX="{'_testing' if testing else ''}"
        N_SAMPLES_FLAG="{n_samples_flag}"
    """
    )


def train_cmd(
    model: str,
    mode: str,
    train_dataset: str,
    model_name: str,
    models_dir: str,
    training_test_label: str,
    no_cache: bool = False,
    skip_eval: bool = False,
) -> str:
    """Generate a training command."""
    subfolder = f"{model}_{mode}/{model_name}_on_{training_test_label}"
    cmd = (
        f"python3 models/training.py \\\n"
        f"    --dataset=$DATASETS_DIR/{train_dataset} \\\n"
        f"    --models {model} \\\n"
        f"    --subfolder={subfolder} \\\n"
        f"    --save-model-path={models_dir}/{model_name} \\\n"
        f"    $TESTING_FLAG $N_SAMPLES_FLAG"
        + (" \\\n    --no-feature-cache" if no_cache else "")
        + (" \\\n    --skip-eval" if skip_eval else "")
    )
    return cmd


def eval_cmd(
    model: str,
    model_name: str,
    models_dir: str,
    results_dir: str,
    test_datasets: list[tuple[str, str]],
    no_cache: bool = False,
) -> str:
    """Generate an evaluation command using --test-datasets."""
    td_args = " ".join(f"$DATASETS_DIR/{path}:{label}" for path, label in test_datasets)
    cmd = (
        f"python3 experiments/evaluate_model.py \\\n"
        f"    --model-path={models_dir}/{model_name}${{MODEL_NAME_SUFFIX}}.pth \\\n"
        f"    --model-type={model} \\\n"
        f"    --test-datasets {td_args} \\\n"
        f"    --output-dir={results_dir}/ \\\n"
        f"    --fixed-fpr=0.01 \\\n"
        f"    $TESTING_FLAG $N_SAMPLES_FLAG"
        + (" \\\n    --no-feature-cache" if no_cache else "")
    )
    return cmd


def generate_lodo_script(
    model: str,
    scenario_num: int,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
    no_matrix: bool = False,
    no_cache: bool = False,
    eval_only: bool = False,
    n_samples: int | None = None,
    run_id: int | None = None,
) -> str:
    """Generate a script for a LODO (leave-one-dataset-out) scenario."""
    scenario = LODO_SCENARIOS[scenario_num]
    model_name = f"{model}_{scenario['train_label']}"
    # The held-out (test) dataset for this LODO scenario
    training_test_label = next(l for l in "ABCD" if l not in scenario["train_label"])
    train_file = lodo_filename(training_test_label)
    run_suffix = f"_repro{run_id}" if run_id is not None else ""
    models_dir = f"./output/checkpoints/{model}_lodo{run_suffix}"
    results_dir = f"./output/results/{model}_lodo{run_suffix}"

    job_suffix = f"lodo_s{scenario_num}{run_suffix}"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    if no_matrix:
        test_labels = [training_test_label]
    else:
        test_labels = scenario["test_labels"]
    # Primary held-out dataset: use LODO file; other datasets: use in-domain files
    test_datasets = [
        (
            (
                lodo_filename(label)
                if label == training_test_label
                else in_domain_filename(label)
            ),
            label,
        )
        for label in test_labels
    ]

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path, long=not no_matrix))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(
        env_setup(
            testing, datasets_dir, log_dir, log_file, conda_env_for(model), n_samples
        )
    )
    parts.append(f'echo "Running LODO scenario {scenario_num}: {model_name}"')
    parts.append("")
    if not eval_only:
        parts.append(f"# Train {model_name}")
        parts.append(
            train_cmd(
                model,
                "lodo",
                train_file,
                model_name,
                models_dir,
                training_test_label=training_test_label,
                no_cache=no_cache,
                skip_eval=True,
            )
        )
        parts.append("")
    parts.append(f"# Evaluate {model_name} on all test datasets")
    parts.append(
        eval_cmd(
            model,
            model_name,
            models_dir,
            results_dir,
            test_datasets,
            no_cache=no_cache,
        )
    )
    parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def generate_in_domain_script(
    model: str,
    scenario_num: int,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
    no_matrix: bool = False,
    no_cache: bool = False,
    eval_only: bool = False,
    n_samples: int | None = None,
    run_id: int | None = None,
) -> str:
    """Generate a script for an in-domain (single-dataset) scenario."""
    scenario = IN_DOMAIN_SCENARIOS[scenario_num]
    model_name = f"{model}_{scenario['train_label']}"
    training_test_label = scenario["train_label"]
    train_file = in_domain_filename(training_test_label)
    run_suffix = f"_repro{run_id}" if run_id is not None else ""
    models_dir = f"./output/checkpoints/{model}_in_domain{run_suffix}"
    results_dir = f"./output/results/{model}_in_domain{run_suffix}"

    job_suffix = f"in_domain_s{scenario_num}{run_suffix}"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    if no_matrix:
        test_labels = [scenario["train_label"]]
    else:
        test_labels = scenario["test_labels"]
    test_datasets = [(in_domain_filename(label), label) for label in test_labels]

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path, long=not no_matrix))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(
        env_setup(
            testing, datasets_dir, log_dir, log_file, conda_env_for(model), n_samples
        )
    )
    parts.append(f'echo "Running in-domain scenario {scenario_num}: {model_name}"')
    parts.append("")
    if not eval_only:
        parts.append(f"# Train {model_name}")
        parts.append(
            train_cmd(
                model,
                "in_domain",
                train_file,
                model_name,
                models_dir,
                training_test_label=training_test_label,
                no_cache=no_cache,
                skip_eval=True,
            )
        )
        parts.append("")
    parts.append(f"# Evaluate {model_name} on all test datasets")
    parts.append(
        eval_cmd(
            model,
            model_name,
            models_dir,
            results_dir,
            test_datasets,
            no_cache=no_cache,
        )
    )
    parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def generate_wafamole_script(
    model: str,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
    no_matrix: bool = False,
    no_cache: bool = False,
    no_train: bool = False,
) -> str:
    """Generate a script for wafamole experiments (3 phases)."""
    spec_models_dir = f"./output/checkpoints/{model}_in_domain"
    gen_models_dir = f"./output/checkpoints/{model}_lodo"
    spec_results_dir = f"./output/results/{model}_in_domain"
    gen_results_dir = f"./output/results/{model}_lodo"
    wafamole_file = in_domain_filename("E")

    job_suffix = "wafamole"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path, long=not no_matrix))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(
        env_setup(testing, datasets_dir, log_dir, log_file, conda_env_for(model))
    )

    # Phase 1: Train E model
    model_name_e = f"{model}_E"
    if not no_train:
        parts.append("# ── Phase 1: Train E (in-domain on wafamole) ──")
        parts.append(
            train_cmd(
                model,
                "in_domain",
                wafamole_file,
                model_name_e,
                spec_models_dir,
                training_test_label="E",
                no_cache=no_cache,
                skip_eval=True,
            )
        )
        parts.append("")

    # Phase 2: Evaluate all existing models on E
    if not no_matrix:
        parts.append("# ── Phase 2: Evaluate all models on wafamole (E) ──")
        wafamole_test = [(wafamole_file, "E")]

        parts.append('echo "Evaluating LODO models on wafamole..."')
        for scenario in LODO_SCENARIOS.values():
            gname = f"{model}_{scenario['train_label']}"
            parts.append(
                eval_cmd(
                    model,
                    gname,
                    gen_models_dir,
                    gen_results_dir,
                    wafamole_test,
                    no_cache=no_cache,
                )
            )
            parts.append("")

        parts.append('echo "Evaluating in-domain models on wafamole..."')
        for scenario in IN_DOMAIN_SCENARIOS.values():
            sname = f"{model}_{scenario['train_label']}"
            parts.append(
                eval_cmd(
                    model,
                    sname,
                    spec_models_dir,
                    spec_results_dir,
                    wafamole_test,
                    no_cache=no_cache,
                )
            )
            parts.append("")
        # Also evaluate E on E
        parts.append(
            eval_cmd(
                model,
                model_name_e,
                spec_models_dir,
                spec_results_dir,
                wafamole_test,
                no_cache=no_cache,
            )
        )
        parts.append("")

    # Phase 3: Evaluate E model on all other datasets
    if not no_train:
        parts.append("# ── Phase 3: Evaluate E model on other datasets ──")
        other_test = [
            (in_domain_filename(label), label) for label in ["A", "B", "C", "D"]
        ]
        parts.append(
            eval_cmd(
                model,
                model_name_e,
                spec_models_dir,
                spec_results_dir,
                other_test,
                no_cache=no_cache,
            )
        )
        parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def malignancy_cmd(
    model: str,
    train_label: str,
    models_dir: str,
    results_dir: str,
    no_cache: bool = False,
) -> str:
    """Generate the malignancy experiment command for a single scenario."""
    cmd = (
        f"python3 experiments/malignancy.py \\\n"
        f"    --model {model} \\\n"
        f"    --model-dir={models_dir} \\\n"
        f"    --scenario {train_label} \\\n"
        f"    --output-dir={results_dir} \\\n"
        f"    $TESTING_FLAG" + (" \\\n    --no-cache" if no_cache else "")
    )
    return cmd


def generate_malignancy_script(
    model: str,
    scenario_num: int,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
    no_cache: bool = False,
) -> str:
    """Generate a script for one malignancy scenario: train model if absent, then run malignancy."""
    scenario = LODO_SCENARIOS[scenario_num]
    model_name = f"{model}_{scenario['train_label']}"
    training_test_label = next(l for l in "ABCD" if l not in scenario["train_label"])
    train_file = lodo_filename(training_test_label)
    models_dir = f"./output/checkpoints/{model}_lodo"
    results_dir = "./output/results/malignancy"

    job_suffix = f"malignancy_s{scenario_num}"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(
        env_setup(testing, datasets_dir, log_dir, log_file, conda_env_for(model))
    )
    parts.append(f'echo "Running malignancy scenario {scenario_num}: {model_name}"')
    parts.append("")

    # Train only if model not already saved
    parts.append(f"if [ ! -f {models_dir}/{model_name}.pth ]; then")
    parts.append(f'    echo "Training {model_name} ..."')
    parts.append(
        "    "
        + train_cmd(
            model,
            "lodo",
            train_file,
            model_name,
            models_dir,
            training_test_label=training_test_label,
            no_cache=no_cache,
        ).replace("\n", "\n    ")
    )
    parts.append("else")
    parts.append(f'    echo "Skipping training — {model_name} already exists"')
    parts.append("fi")
    parts.append("")

    parts.append(f"# Malignancy experiment for {model_name}")
    parts.append(
        malignancy_cmd(
            model,
            scenario["train_label"],
            models_dir,
            results_dir,
            no_cache=no_cache,
        )
    )
    parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def shap_cmd(
    model: str,
    model_path: str,
    dataset_file: str,
    output_dir: str,
) -> str:
    """Generate the SHAP analysis command for a single scenario."""
    return (
        f"python3 models/shap_analysis.py \\\n"
        f"    --model-type={model} \\\n"
        f"    --model-path={model_path}${{MODEL_NAME_SUFFIX}} \\\n"
        f"    --dataset=$DATASETS_DIR/{dataset_file} \\\n"
        f"    --output-dir={output_dir} \\\n"
        f"    $TESTING_FLAG"
    )


def generate_shap_script(
    model: str,
    mode: str,
    scenario_num: int,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
    no_cache: bool = False,
) -> str:
    """Generate a SHAP script: train model if absent, then run SHAP analysis."""
    if mode == "lodo":
        scenario = LODO_SCENARIOS[scenario_num]
        models_dir = f"./output/checkpoints/{model}_lodo"
        training_test_label = next(
            l for l in "ABCD" if l not in scenario["train_label"]
        )
        train_file = lodo_filename(training_test_label)
    else:
        scenario = IN_DOMAIN_SCENARIOS[scenario_num]
        models_dir = f"./output/checkpoints/{model}_in_domain"
        training_test_label = scenario["train_label"]
        train_file = in_domain_filename(training_test_label)

    model_name = f"{model}_{scenario['train_label']}"
    output_dir = "./output/results/shap"

    job_suffix = f"shap_{mode}_s{scenario_num}"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(
        env_setup(testing, datasets_dir, log_dir, log_file, conda_env_for(model))
    )
    parts.append(f'echo "Running SHAP {mode} scenario {scenario_num}: {model_name}"')
    parts.append("")

    parts.append(
        f"if [ ! -f {models_dir}/{model_name}${{MODEL_NAME_SUFFIX}}.pth ]; then"
    )
    parts.append(f'    echo "Training {model_name} ..."')
    parts.append(
        "    "
        + train_cmd(
            model,
            mode,
            train_file,
            model_name,
            models_dir,
            training_test_label=training_test_label,
            no_cache=no_cache,
            skip_eval=True,
        ).replace("\n", "\n    ")
    )
    parts.append("else")
    parts.append(f'    echo "Skipping training — {model_name} already exists"')
    parts.append("fi")
    parts.append("")

    parts.append(f"# SHAP analysis for {model_name}")
    parts.append(shap_cmd(model, f"{models_dir}/{model_name}", train_file, output_dir))
    parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def generate_concept_drift_script(
    model: str,
    scenario_num: int,
    testing: bool,
    slurm: bool,
    no_cache: bool = False,
) -> str:
    """Generate a script for concept-drift (origin vs shifted templates)."""
    scenario = CONCEPT_DRIFT_SCENARIOS[scenario_num]
    model_name = f"{model}_{scenario['train_label']}"
    train_file = dataset_filename("origin", scenario["train_dataset"])
    models_dir = f"./output/checkpoints/{model}_concept_drift"
    results_dir = f"./output/results/{model}_concept_drift"
    datasets_dir = CONCEPT_DRIFT_DATASETS_DIR

    job_suffix = f"concept_drift_s{scenario_num}"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    test_datasets = [
        (dataset_filename("origin", scenario["train_dataset"]), "origin"),
        (dataset_filename("shifted", scenario["train_dataset"]), "shifted"),
    ]

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(
        env_setup(testing, datasets_dir, log_dir, log_file, conda_env_for(model))
    )
    parts.append(f'echo "Running concept-drift scenario {scenario_num}: {model_name}"')
    parts.append("")
    parts.append(f"# Train {model_name} on origin templates")
    parts.append(
        train_cmd(
            model,
            "concept_drift",
            train_file,
            model_name,
            models_dir,
            training_test_label="origin",
            no_cache=no_cache,
            skip_eval=True,
        )
    )
    parts.append("")
    parts.append(
        f"# Evaluate {model_name} on origin (seen) and shifted (unseen) templates"
    )
    parts.append(
        eval_cmd(
            model,
            model_name,
            models_dir,
            results_dir,
            test_datasets,
            no_cache=no_cache,
        )
    )
    parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def fine_tuning_cmd(
    model: str,
    model_path: str,
    target_dataset_file: str,
    output_dir: str,
    no_cache: bool = False,
) -> str:
    """Generate the fine-tuning command for a single scenario."""
    cmd = (
        f"python3 experiments/fine_tuning.py \\\n"
        f"    --model-type={model} \\\n"
        f"    --model-path={model_path} \\\n"
        f"    --target-dataset=$DATASETS_DIR/{target_dataset_file} \\\n"
        f"    --output-dir={output_dir} \\\n"
        f"    $TESTING_FLAG"
    )
    return cmd


def generate_fine_tuning_script(
    model: str,
    scenario_num: int,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
    no_cache: bool = False,
) -> str:
    """Generate a fine-tuning script for one LODO scenario.

    Loads the pre-trained LODO model (trains it if absent) and sweeps k normal
    samples from the held-out target domain to fine-tune the AE.
    """
    scenario = LODO_SCENARIOS[scenario_num]
    model_name = f"{model}_{scenario['train_label']}"
    target_label = next(l for l in "ABCD" if l not in scenario["train_label"])
    target_file = in_domain_filename(target_label)
    models_dir = f"./output/checkpoints/{model}_lodo"
    results_dir = f"./output/results/{model}_fine_tuning"

    job_suffix = f"fine_tuning_s{scenario_num}"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    output_dir = f"{results_dir}/{model_name}_on_{target_label}"

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(
        env_setup(testing, datasets_dir, log_dir, log_file, conda_env_for(model))
    )
    parts.append(
        f'echo "Running fine-tuning scenario {scenario_num}: {model_name} → {target_label}"'
    )
    parts.append("")

    train_file = lodo_filename(target_label)
    training_test_label = target_label
    parts.append(
        f"if [ ! -f {models_dir}/{model_name}${{MODEL_NAME_SUFFIX}}.pth ]; then"
    )
    parts.append(f'    echo "Training {model_name} ..."')
    parts.append(
        "    "
        + train_cmd(
            model,
            "lodo",
            train_file,
            model_name,
            models_dir,
            training_test_label=training_test_label,
            no_cache=no_cache,
            skip_eval=True,
        ).replace("\n", "\n    ")
    )
    parts.append("else")
    parts.append(f'    echo "Skipping training — {model_name} already exists"')
    parts.append("fi")
    parts.append("")

    parts.append(
        fine_tuning_cmd(
            model,
            f"{models_dir}/{model_name}${{MODEL_NAME_SUFFIX}}.pth",
            target_file,
            output_dir,
            no_cache=no_cache,
        )
    )
    parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def generate_domain_shift_script(
    model: str,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
) -> str:
    """Generate a domain-shift detection script for a single extractor."""
    job_suffix = "domain_shift"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(
        env_setup(testing, datasets_dir, log_dir, log_file, conda_env_for(model))
    )
    parts.append(f'echo "Running domain-shift detection for {model}"')
    parts.append("")
    cmd = (
        f"python3 experiments/domain_shift.py \\\n"
        f"    --dataset A $DATASETS_DIR/bcd-a.csv \\\n"
        f"    --dataset B $DATASETS_DIR/acd-b.csv \\\n"
        f"    --dataset C $DATASETS_DIR/abd-c.csv \\\n"
        f"    --dataset D $DATASETS_DIR/abc-d.csv \\\n"
        f"    --extractor {model} \\\n"
        f"    --workers ${{SLURM_CPUS_PER_TASK:-16}} \\\n"
        f"    $TESTING_FLAG"
    )
    parts.append(cmd)
    parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def write_and_submit(
    script_content: str,
    script_name: str,
    dry_run: bool,
    local: bool,
) -> None:
    """Write a generated script and optionally submit/run it."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    script_path = GENERATED_DIR / script_name

    if dry_run:
        print(f"{'=' * 60}")
        print(f"# {script_name}")
        print(f"{'=' * 60}")
        print(script_content)
        print()
        return

    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"Written: {script_path}")

    if local:
        print(f"Running locally: {script_path}")
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            print(f"Script exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
    else:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"sbatch failed: {result.stderr}", file=sys.stderr)
            sys.exit(result.returncode)
        print(f"Submitted: {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM experiment scripts"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_PROFILES.keys()),
        help="Model type to run experiments for",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "lodo",
            "in_domain",
            "wafamole",
            "domain_shift",
            "malignancy",
            "shap",
            "concept_drift",
            "fine_tuning",
        ],
        help="Experiment mode",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario number (1-4) or 'all' (default: all). Ignored for wafamole mode.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=os.path.expanduser("~/datasets/100k-training/"),
        help="Path to datasets directory",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Enable testing mode (limit samples)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        dest="n_samples",
        help="Limit training to N samples (deterministic). Use with --repro-runs for reproducibility checks.",
    )
    parser.add_argument(
        "--repro-runs",
        type=int,
        default=1,
        dest="repro_runs",
        choices=range(1, 5),
        metavar="{1-4}",
        help="Number of independent runs to submit with identical settings (for reproducibility checks). Each run gets isolated output dirs (e.g. _repro1, _repro2, ...). Only applies to lodo and in_domain modes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated scripts without submitting or running",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of submitting to SLURM",
    )
    parser.add_argument(
        "--no-matrix",
        action="store_true",
        help="Only evaluate on the key dataset (left-out for lodo, trained for in_domain); skips wafamole phase 2",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training (phases 1 and 3 for wafamole mode); only run evaluations on existing models.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable all feature and embedding caches in training and evaluation",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation (requires a pre-trained model). Only applies to lodo and in_domain modes.",
    )
    args = parser.parse_args()

    if args.dry_run and args.local:
        parser.error("--dry-run and --local are mutually exclusive")

    if args.model in GAUR_MODELS:
        if not args.local and not args.dry_run:
            parser.error(
                f"Model '{args.model}' requires --local (gaur experiments cannot run on SLURM)"
            )
        if args.local and shutil.which("nix") is None:
            parser.error(
                "nix is not available in PATH — required to build instrumented servers for gaur experiments"
            )

    use_slurm = not args.local and not args.dry_run

    if args.mode == "malignancy":
        if args.scenario == "all":
            scenario_nums = [1, 2, 3, 4]
        else:
            try:
                n = int(args.scenario)
                if n < 1 or n > 4:
                    raise ValueError
                scenario_nums = [n]
            except ValueError:
                parser.error(f"--scenario must be 1-4 or 'all', got '{args.scenario}'")
        for n in scenario_nums:
            script = generate_malignancy_script(
                args.model,
                n,
                args.testing,
                args.datasets_dir,
                use_slurm,
                args.no_cache,
            )
            write_and_submit(
                script,
                f"{args.model}_malignancy_scenario{n}.sh",
                args.dry_run,
                args.local,
            )
    elif args.mode == "concept_drift":
        if args.scenario == "all":
            scenario_nums = [1, 2, 3, 4]
        else:
            try:
                n = int(args.scenario)
                if n < 1 or n > 4:
                    raise ValueError
                scenario_nums = [n]
            except ValueError:
                parser.error(f"--scenario must be 1-4 or 'all', got '{args.scenario}'")
        for n in scenario_nums:
            script = generate_concept_drift_script(
                args.model, n, args.testing, use_slurm, args.no_cache
            )
            write_and_submit(
                script,
                f"{args.model}_concept_drift_scenario{n}.sh",
                args.dry_run,
                args.local,
            )
    elif args.mode == "fine_tuning":
        if args.scenario == "all":
            scenario_nums = [1, 2, 3, 4]
        else:
            try:
                n = int(args.scenario)
                if n < 1 or n > 4:
                    raise ValueError
                scenario_nums = [n]
            except ValueError:
                parser.error(f"--scenario must be 1-4 or 'all', got '{args.scenario}'")
        for n in scenario_nums:
            script = generate_fine_tuning_script(
                args.model,
                n,
                args.testing,
                args.datasets_dir,
                use_slurm,
                args.no_cache,
            )
            write_and_submit(
                script,
                f"{args.model}_fine_tuning_scenario{n}.sh",
                args.dry_run,
                args.local,
            )
    elif args.mode == "domain_shift":
        script = generate_domain_shift_script(
            args.model,
            args.testing,
            args.datasets_dir,
            use_slurm,
        )
        write_and_submit(
            script, f"{args.model}_domain_shift.sh", args.dry_run, args.local
        )
    elif args.mode == "wafamole":
        script = generate_wafamole_script(
            args.model,
            args.testing,
            args.datasets_dir,
            use_slurm,
            args.no_matrix,
            args.no_cache,
            args.no_train,
        )
        write_and_submit(script, f"{args.model}_wafamole.sh", args.dry_run, args.local)
    elif args.mode == "shap":
        SHAP_COMPATIBLE_MODELS = {"ae_li", "ae_gaur", "ae_loginov"}
        if args.model not in SHAP_COMPATIBLE_MODELS:
            parser.error(
                f"--mode shap only supports: {', '.join(sorted(SHAP_COMPATIBLE_MODELS))}"
            )
        if args.scenario == "all":
            scenario_nums = [1, 2, 3, 4]
        else:
            try:
                n = int(args.scenario)
                if n < 1 or n > 4:
                    raise ValueError
                scenario_nums = [n]
            except ValueError:
                parser.error(f"--scenario must be 1-4 or 'all', got '{args.scenario}'")
        for n in scenario_nums:
            for shap_mode in ("lodo", "in_domain"):
                script = generate_shap_script(
                    args.model,
                    shap_mode,
                    n,
                    args.testing,
                    args.datasets_dir,
                    use_slurm,
                    args.no_cache,
                )
                write_and_submit(
                    script,
                    f"{args.model}_shap_{shap_mode}_scenario{n}.sh",
                    args.dry_run,
                    args.local,
                )
    else:
        # Determine scenarios to run
        if args.scenario == "all":
            scenario_nums = [1, 2, 3, 4]
        else:
            try:
                n = int(args.scenario)
                if n < 1 or n > 4:
                    raise ValueError
                scenario_nums = [n]
            except ValueError:
                parser.error(f"--scenario must be 1-4 or 'all', got '{args.scenario}'")

        generator = (
            generate_lodo_script if args.mode == "lodo" else generate_in_domain_script
        )
        run_ids = range(1, args.repro_runs + 1) if args.repro_runs > 1 else [None]
        for n in scenario_nums:
            for run_id in run_ids:
                script = generator(
                    args.model,
                    n,
                    args.testing,
                    args.datasets_dir,
                    use_slurm,
                    args.no_matrix,
                    args.no_cache,
                    args.eval_only,
                    args.n_samples,
                    run_id,
                )
                repro_suffix = f"_repro{run_id}" if run_id is not None else ""
                write_and_submit(
                    script,
                    f"{args.model}_{args.mode}_scenario{n}{repro_suffix}.sh",
                    args.dry_run,
                    args.local,
                )


if __name__ == "__main__":
    main()
