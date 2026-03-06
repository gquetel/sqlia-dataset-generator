#!/usr/bin/env python3
"""
Unified experiment launcher.

Generates SLURM scripts (or runs locally) for training and evaluating models
across generic, specialised, and wafamole experiment modes.
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
    "gpu_standard": {
        "partition": "A100",
        "gres": "gpu:1",
        "cpus": 16,
        "mem": "32G",
        "time": "4:00:00",
    },
    "gpu_v100": {
        "partition": "V100",
        "gres": "gpu:1",
        "cpus": 16,
        "mem": "32G",
        "time": "4:00:00",
        "exclude": ["node13", "node42", "node43"],  # nodes with != python version
    },
    "gpu_long": {
        "partition": "A100",
        "gres": "gpu:1",
        "cpus": 16,
        "mem": "64G",
        "time": "12:00:00",
    },
    "gpu_A40": {
        "partition": "A40",
        "gres": "gpu:1",
        "cpus": 12,  # Fewer CPUs to reach smaller machines
        "mem": "32G",
        "time": "12:00:00",
    },
    "cpu": {
        "partition": "CPU",
        "gres": None,
        "cpus": 32,
        "mem": "32G",
        "time": "12:00:00",
    },
}

MODEL_PROFILES = {
    "ae_securebert": "gpu_standard",
    "ae_roberta": "gpu_v100",
    "ae_kakisim_c": "gpu_long",
    "ae_bilstm_w2v": "gpu_standard",  # 4H is sufficient
    "ae_w2v": "gpu_v100",
    "ae_kakisim_w2v_max": "gpu_v100",
    "ocsvm_kakisim_w2v_max": "cpu",
    "ae_kakisim_w2v_mean": "gpu_standard",
    "ocsvm_kakisim_w2v_mean": "cpu",
    "ae_li": "cpu",
    "ae_loginov": "cpu",
    "ae_gaur": "cpu",
    "ae_gaur_chatgpt": "cpu",
    "ae_codebert": "gpu_v100",
    "ae_flan_t5": "gpu_v100",
    "ae_sentbert": "gpu_v100",
    "ae_llm2vec": "gpu_long",  # Requires 32Gb+ GPU, and 64+ memory
    "ocsvm_li": "cpu",
}
DATASETS = {
    "A": "OurAirports",
    "B": "sakila",
    "C": "AdventureWorks",
    "D": "OHR",
    "E": "wafamole",
}

# Generic mode: leave-one-out (train on 3, test on all 4)
# Scenario N trains on the dataset that *excludes* dataset N from {A,B,C,D}
GENERIC_SCENARIOS = {
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

# Specialised mode: train on single dataset, test on all 4
SPECIALISED_SCENARIOS = {
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


GAUR_MODELS = {m for m in MODEL_PROFILES if "gaur" in m}


def get_profile(model: str) -> dict:
    """Return the SLURM resource profile for a model."""
    profile_name = MODEL_PROFILES[model]
    return SLURM_PROFILES[profile_name]


def venv_for(model: str) -> str:
    """Return the virtualenv path for a model based on its target partition."""
    profile = get_profile(model)
    # They fixed it ! Same python version everywhere !
    # I keep the old code in case some drift reappears.
    # if profile["partition"] == "V100":
    #     return "venv-3.10.12"

    return "venv-3.12.12"


def dataset_filename(mode: str, db_name: str) -> str:
    """Return the CSV filename for a dataset.

    mode is 'generic' or 'specialised'.
    """
    return f"{mode}-{db_name}.csv"


def log_dir_for(model: str, job_suffix: str) -> str:
    """Return the absolute log directory path."""
    return f"{REPO_ROOT}/logs/{model}/{job_suffix}"


def make_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def sbatch_header(model: str, job_suffix: str, log_path: str) -> str:
    """Generate SBATCH header lines for a model."""
    profile = get_profile(model)
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
    venv: str = "venv-3.12.12",
) -> str:
    """Generate environment setup lines with log directory creation and latest symlink."""
    return dedent(
        f"""\
        echo "Starting job on node: $(hostname)"
        echo "Job started at: $(date)"

        cd ~/repos/sqlia-dataset/
        source {venv}/bin/activate

        mkdir -p {log_dir}
        ln -sfn {log_file} {log_dir}/latest.log

        DATASETS_DIR={datasets_dir}
        TESTING_FLAG="{'--testing' if testing else ''}"
    """
    )


def train_cmd(
    model: str,
    mode: str,
    train_dataset: str,
    model_name: str,
    models_dir: str,
    no_cache: bool = False,
) -> str:
    """Generate a training command."""
    subfolder = f"{model}_{mode}/{train_dataset.replace('.csv', '')}-{model}"
    cmd = (
        f"python3 models/training.py \\\n"
        f"    --dataset=$DATASETS_DIR/{train_dataset} \\\n"
        f"    --models {model} \\\n"
        f"    --subfolder={subfolder} \\\n"
        f"    --save-model-path={models_dir}/{model_name} \\\n"
        f"    $TESTING_FLAG" + (" \\\n    --no-feature-cache" if no_cache else "")
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
        f"    --model-path={models_dir}/{model_name}.pth \\\n"
        f"    --model-type={model} \\\n"
        f"    --test-datasets {td_args} \\\n"
        f"    --output-dir={results_dir}/ \\\n"
        f"    --fixed-fpr=0.01 \\\n"
        f"    $TESTING_FLAG" + (" \\\n    --no-feature-cache" if no_cache else "")
    )
    return cmd


def generate_generic_script(
    model: str,
    scenario_num: int,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
    no_matrix: bool = False,
    no_cache: bool = False,
) -> str:
    """Generate a script for a generic (leave-one-out) scenario."""
    scenario = GENERIC_SCENARIOS[scenario_num]
    model_name = f"{model}_{scenario['train_label']}"
    train_file = dataset_filename("generic", scenario["train_dataset"])
    models_dir = f"./models/output/models/{model}_generic"
    results_dir = f"./models/output/{model}_generic"

    job_suffix = f"generic_s{scenario_num}"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    if no_matrix:
        test_labels = [l for l in "ABCD" if l not in scenario["train_label"]]
    else:
        test_labels = scenario["test_labels"]
    test_datasets = [
        (dataset_filename("generic", DATASETS[label]), label) for label in test_labels
    ]

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(env_setup(testing, datasets_dir, log_dir, log_file, venv_for(model)))
    parts.append(f'echo "Running generic scenario {scenario_num}: {model_name}"')
    parts.append("")
    parts.append(f"# Train {model_name}")
    parts.append(
        train_cmd(
            model, "generic", train_file, model_name, models_dir, no_cache=no_cache
        )
    )
    parts.append("")
    parts.append(f"# Evaluate {model_name} on all test datasets")
    parts.append(
        eval_cmd(
            model, model_name, models_dir, results_dir, test_datasets, no_cache=no_cache
        )
    )
    parts.append("")
    parts.append('echo "Job finished at: $(date)"')
    return "\n".join(parts)


def generate_specialised_script(
    model: str,
    scenario_num: int,
    testing: bool,
    datasets_dir: str,
    slurm: bool,
    no_matrix: bool = False,
    no_cache: bool = False,
) -> str:
    """Generate a script for a specialised (single-dataset) scenario."""
    scenario = SPECIALISED_SCENARIOS[scenario_num]
    model_name = f"{model}_{scenario['train_label']}"
    train_file = dataset_filename("specialised", scenario["train_dataset"])
    models_dir = f"./models/output/models/{model}_specialised"
    results_dir = f"./models/output/{model}_specialised"

    job_suffix = f"specialised_s{scenario_num}"
    log_dir = log_dir_for(model, job_suffix)
    timestamp = make_timestamp()
    log_file = f"{timestamp}.log"
    log_path = f"{log_dir}/{log_file}"

    if no_matrix:
        test_labels = [scenario["train_label"]]
    else:
        test_labels = scenario["test_labels"]
    test_datasets = [
        (dataset_filename("specialised", DATASETS[label]), label)
        for label in test_labels
    ]

    parts = []
    if slurm:
        parts.append(sbatch_header(model, job_suffix, log_path))
    else:
        parts.append("#!/bin/bash")
    parts.append("")
    parts.append(env_setup(testing, datasets_dir, log_dir, log_file, venv_for(model)))
    parts.append(f'echo "Running specialised scenario {scenario_num}: {model_name}"')
    parts.append("")
    parts.append(f"# Train {model_name}")
    parts.append(
        train_cmd(
            model, "specialised", train_file, model_name, models_dir, no_cache=no_cache
        )
    )
    parts.append("")
    parts.append(f"# Evaluate {model_name} on all test datasets")
    parts.append(
        eval_cmd(
            model, model_name, models_dir, results_dir, test_datasets, no_cache=no_cache
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
) -> str:
    """Generate a script for wafamole experiments (3 phases)."""
    spec_models_dir = f"./models/output/models/{model}_specialised"
    gen_models_dir = f"./models/output/models/{model}_generic"
    spec_results_dir = f"./models/output/{model}_specialised"
    gen_results_dir = f"./models/output/{model}_generic"
    wafamole_file = dataset_filename("specialised", "wafamole")

    job_suffix = "wafamole"
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
    parts.append(env_setup(testing, datasets_dir, log_dir, log_file, venv_for(model)))

    # Phase 1: Train E model
    model_name_e = f"{model}_E"
    parts.append("# ── Phase 1: Train E (specialised on wafamole) ──")
    parts.append(
        train_cmd(
            model,
            "specialised",
            wafamole_file,
            model_name_e,
            spec_models_dir,
            no_cache=no_cache,
        )
    )
    parts.append("")

    # Phase 2: Evaluate all existing models on E
    if not no_matrix:
        parts.append("# ── Phase 2: Evaluate all models on wafamole (E) ──")
        wafamole_test = [(wafamole_file, "E")]

        parts.append('echo "Evaluating generic models on wafamole..."')
        for scenario in GENERIC_SCENARIOS.values():
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

        parts.append('echo "Evaluating specialised models on wafamole..."')
        for scenario in SPECIALISED_SCENARIOS.values():
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
    parts.append("# ── Phase 3: Evaluate E model on other datasets ──")
    other_test = [
        (dataset_filename("specialised", DATASETS[label]), label)
        for label in ["A", "B", "C", "D"]
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
        choices=["generic", "specialised", "wafamole"],
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
        help="Only evaluate on the key dataset (left-out for generic, trained for specialised); skips wafamole phase 2",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable all feature and embedding caches in training and evaluation",
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

    if args.mode == "wafamole":
        script = generate_wafamole_script(
            args.model,
            args.testing,
            args.datasets_dir,
            use_slurm,
            args.no_matrix,
            args.no_cache,
        )
        write_and_submit(script, f"{args.model}_wafamole.sh", args.dry_run, args.local)
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
            generate_generic_script
            if args.mode == "generic"
            else generate_specialised_script
        )
        for n in scenario_nums:
            script = generator(
                args.model,
                n,
                args.testing,
                args.datasets_dir,
                use_slurm,
                args.no_matrix,
                args.no_cache,
            )
            write_and_submit(
                script,
                f"{args.model}_{args.mode}_scenario{n}.sh",
                args.dry_run,
                args.local,
            )


if __name__ == "__main__":
    main()
