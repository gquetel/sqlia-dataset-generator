# Experiments

| Script | Description |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [dataset_stats.py](dataset_stats.py) | Generates summary statistics (sample counts, statement type distributions, attack technique) for all datasets. |
| [generate_splits.py](generate_splits.py) | Creates "generic" (3 datasets → 1) and "specialised" datasets configurations from input datasets. |
| [diversity_metric.py](diversity_metric.py) | Computes vocabulary size/TTR, unique parse trees, and semantic diversity via embeddings. Supports WAFAMOLE and Kaggle baselines. |
| [evaluate_model.py](evaluate_model.py) | Loads a trained detection model and evaluates it on test datasets. |
| [report_generic_vs_specialised.py](report_generic_vs_specialised.py) | Generates comparison visualizations across feature extractors: generic vs specialised bar charts, ROC curves, recall heatmaps per technique/statement type, and transfer-learning matrices. |
| [domain_shift.py](domain_shift.py) | A script attempting to detect distribution shift between train and test feature spaces using MMD, KS test, and domain classifier. |
| [malignancy.py](malignancy.py) | Assesses whether distribution shifts are "malignant" by training a domain classifier and evaluating the AE detector on test sets filtered by target-domain likelihood. |
| [shift_auroc_correlation.py](shift_auroc_correlation.py) | Correlates domain shift detection accuracy with model generalization (AUROC), computing Pearson/Spearman correlations and generating scatter plots. |
| [malignancy_auroc_correlation.py](malignancy_auroc_correlation.py) | Correlates malignancy metrics (topk_roc_auc, topk_fpr) with AUROC generalization, assessing whether detected shifts actually hurt detector performance. |
| [plot_baseline_curves.ipynb](plot_baseline_curves.ipynb) | Plots AUROC and AUPRC curves comparing feature extraction methods (Li, SecureBERT, CountVectorizer) with anomaly detectors (AE, LOF, OCSVM) on baseline datasets. |
