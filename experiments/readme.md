# Experiments

| Script | Description |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [dataset_stats.py](dataset_stats.py) | Generates summary statistics (sample counts, statement type distributions, attack technique) for all datasets. |
| [generate_splits.py](generate_splits.py) | Creates "generic" (leave-one-domain-out) datasets and "specialised" (in-domain) datasets configurations from input datasets. |
| [diversity_metric.py](diversity_metric.py) | Computes vocabulary size/TTR, unique parse trees, and semantic diversity via embeddings. Supports WAFAMOLE and Kaggle baselines. |
| [evaluate_model.py](evaluate_model.py) | Loads a trained detection model and evaluates it on test datasets. |
| [report_generic_vs_specialised.py](report_generic_vs_specialised.py) | Generates comparison visualizations across feature extractors: generic vs specialised bar charts, ROC curves, recall heatmaps per technique/statement type, and transfer-learning matrices. |
| [plot_baseline_curves.ipynb](plot_baseline_curves.ipynb) | Plots AUROC and AUPRC curves comparing feature extraction methods (Li, SecureBERT, CountVectorizer) with anomaly detectors (AE, LOF, OCSVM) on baseline datasets as done in the ANUBIS paper. |

## Commands

To compute the semantic diversity on the different dataset, run (change datasets paths accordingly):

```bash
python3 experiments/diversity_metric.py --dataset C ~/datasets/100k-training/specialised-AdventureWorks.csv --dataset D  ~/datasets/100k-training/specialised-OHR.csv --dataset A ~/datasets/100k-training/specialised-OurAirports.csv  --dataset B ~/datasets/100k-training/specialised-sakila.csv  --dataset E ~/datasets/100k-training/specialised-wafamole.csv --samples 10000 --div-sem
```

To compute the lexical / syntactic diversity, run:

```bash
python3 experiments/diversity_metric.py --dataset C ~/datasets/100k-training/specialised-AdventureWorks.csv --dataset D  ~/datasets/100k-training/specialised-OHR.csv --dataset A ~/datasets/100k-training/specialised-OurAirports.csv  --dataset B ~/datasets/100k-training/specialised-sakila.csv  --dataset E ~/datasets/100k-training/specialised-wafamole.csv --vocab --parse-trees
```
