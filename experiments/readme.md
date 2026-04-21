# Experiments

| Script | Description |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [dataset_stats.py](dataset_stats.py) | Generates summary statistics (sample counts, statement type distributions, attack technique) for all datasets. |
| [generate_splits.py](generate_splits.py) | Creates LODO (leave-one-domain-out) and in-domain dataset configurations from input datasets. |
| [diversity_metric.py](diversity_metric.py) | Computes vocabulary size/TTR, unique parse trees, and semantic diversity via embeddings. Supports WAFAMOLE and Kaggle baselines. |
| [evaluate_model.py](evaluate_model.py) | Loads a trained detection model and evaluates it on test datasets. |
| [report_lodo_vs_in_domain.py](report_lodo_vs_in_domain.py) | Generates comparison visualizations across feature extractors: LODO vs in-domain bar charts, ROC curves, recall heatmaps per technique/statement type, and transfer-learning matrices. |
| [plot_baseline_curves.ipynb](plot_baseline_curves.ipynb) | Plots AUROC and AUPRC curves comparing feature extraction methods (Li, SecureBERT, CountVectorizer) with anomaly detectors (AE, LOF, OCSVM) on baseline datasets as done in the ANUBIS paper. |

The unified experiment launcher [`scripts/submit_experiments.py`](../scripts/submit_experiments.py) generates bash scripts that call `models/training.py` and `experiments/evaluate_model.py`, then either submits them to SLURM or runs them locally (`--local`).

## submit_experiments.py

### Common flags

| Flag | Description |
| --- | --- |
| `--model MODEL` | Model type (required). Must be a key from `MODEL_PROFILES` in the script (e.g. `ae_li`, `ocsvm_securebert`). |
| `--mode MODE` | Experiment mode (required). See modes below. |
| `--scenario {1-4\|all}` | Scenario number or `all` (default: `all`). Ignored for `wafamole` and `domain_shift`. |
| `--datasets-dir PATH` | Path to the datasets directory (default: `~/datasets/100k-training/`). |
| `--local` | Run the generated script locally instead of submitting to SLURM. |
| `--dry-run` | Print the generated script without writing or running it. |
| `--testing` | Limit samples for a quick sanity check. |
| `--no-matrix` | Evaluate only on the key dataset (left-out for `lodo`, trained for `in_domain`). |
| `--eval-only` | Skip training; only run evaluation on an existing model. |
| `--no-cache` | Disable all feature and embedding caches. |
| `--n-samples N` | Limit training to N samples (deterministic). |
| `--repro-runs {1-4}` | Submit N independent runs with isolated output dirs for reproducibility checks (`lodo`/`in_domain` only). |

### Modes

#### `lodo` — Leave-one-dataset-out

Train on three domains, evaluate on all four. Four scenarios (1=BCD→A, 2=ACD→B, 3=ABD→C, 4=ABC→D).

```bash
# Submit all 4 LODO scenarios to SLURM
python3 scripts/submit_experiments.py --model ae_li --mode lodo

# Run all 4 LODO scenarios locally
python3 scripts/submit_experiments.py --model ae_li --mode lodo --local

# Quick local test
python3 scripts/submit_experiments.py --model ae_li --mode lodo --local --testing

# Only evaluate (model already trained)
python3 scripts/submit_experiments.py --model ae_li --mode lodo --local --eval-only
```

#### `in_domain` — In-domain baseline

Train on one domain, evaluate on all four. Four scenarios (1=A, 2=B, 3=C, 4=D).

```bash
python3 scripts/submit_experiments.py --model ae_li --mode in_domain --local
python3 scripts/submit_experiments.py --model ae_li --mode in_domain --scenario 1 --local
```

#### `wafamole` — WAF-a-MoLE adversarial robustness

Three-phase experiment: (1) train on WAF-a-MoLE dataset E, (2) evaluate all existing models on E, (3) evaluate E model on domains A–D.

```bash
python3 scripts/submit_experiments.py --model ae_li --mode wafamole --local
# Skip re-training if models already exist
python3 scripts/submit_experiments.py --model ae_li --mode wafamole --local --no-train
```

#### `domain_shift` — Domain shift detection

Runs `experiments/domain_shift.py` to measure distributional shift between domains using a given feature extractor.

```bash
python3 scripts/submit_experiments.py --model ae_li --mode domain_shift --local
```

#### `malignancy` — Malignancy analysis

Trains the LODO model if absent, then runs `experiments/malignancy.py` to analyse attack malignancy scores per scenario.

```bash
python3 scripts/submit_experiments.py --model ae_li --mode malignancy --local
python3 scripts/submit_experiments.py --model ae_li --mode malignancy --scenario 1 --local
```

#### `shap` — SHAP feature importance

Trains the model if absent, then runs `models/shap_analysis.py` for both `lodo` and `in_domain` modes. Only supported for `ae_li`, `ae_gaur`, and `ae_loginov`.

```bash
python3 scripts/submit_experiments.py --model ae_li --mode shap --local
python3 scripts/submit_experiments.py --model ae_li --mode shap --scenario 2 --local
```

#### `concept_drift` — Concept drift

Trains on origin templates, evaluates on both origin and shifted templates.

```bash
python3 scripts/submit_experiments.py --model ae_li --mode concept_drift --local
```

#### `fine_tuning` — Fine-tuning LODO models

Loads a pre-trained LODO model (trains it if absent), then sweeps over k normal samples from the held-out target domain to fine-tune the autoencoder.

```bash
python3 scripts/submit_experiments.py --model ae_li --mode fine_tuning --local
python3 scripts/submit_experiments.py --model ae_li --mode fine_tuning --scenario 1 --local
```

## Commands

To compute the semantic diversity on the different dataset, run (change datasets paths accordingly):

```bash
python3 experiments/diversity_metric.py --dataset C ~/datasets/100k-training/c-c.csv --dataset D ~/datasets/100k-training/d-d.csv --dataset A ~/datasets/100k-training/a-a.csv --dataset B ~/datasets/100k-training/b-b.csv --dataset E ~/datasets/100k-training/e-e.csv --samples 10000 --div-sem
```

To compute the lexical / syntactic diversity, run:

```bash
python3 experiments/diversity_metric.py --dataset C ~/datasets/100k-training/c-c.csv --dataset D ~/datasets/100k-training/d-d.csv --dataset A ~/datasets/100k-training/a-a.csv --dataset B ~/datasets/100k-training/b-b.csv --dataset E ~/datasets/100k-training/e-e.csv --vocab --parse-trees
```

To compute the LOVO and in-domain splits:

```bash
python3 experiments/generate_splits.py --output-dir ~/datasets/100k-training/ --seed 2 
```

To compute the concept-drift splits (adapt `REPO_ROOT` in the script);

```bash
python3 experiments/generate_splits.py --output-dir ~/datasets/100k-training/ --seed 2 --concept-drift
```
