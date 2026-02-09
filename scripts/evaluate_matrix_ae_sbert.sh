#!/bin/bash
#SBATCH --job-name=matrix-ae-sbert       # Name of your job
#SBATCH --output=%x_.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_.err             # Error file
#SBATCH --partition=A100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=16            # Request 8 CPU cores
#SBATCH --mem=64G                     # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/
RESULTS_DIR=./models/output/

# Evaluate ae_sbert_ABC on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABC.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_ABC_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABC.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_ABC_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABC.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_ABC_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABC.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_ABC_on_D/

# Evaluate ae_sbert_ABD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_ABD_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_ABD_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_ABD_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_ABD_on_D/

# Evaluate ae_sbert_ACD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ACD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_ACD_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ACD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_ACD_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ACD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_ACD_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ACD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_ACD_on_D/

# Evaluate ae_sbert_BCD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_BCD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_BCD_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_BCD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_BCD_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_BCD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_BCD_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_BCD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_BCD_on_D/

# Print job completion time
echo "Job finished at: $(date)"
