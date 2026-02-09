#!/bin/bash
#SBATCH --job-name=matrix-ae-sbert-spec  # Name of your job
#SBATCH --output=%x_%j.out               # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err                # Error file
#SBATCH --partition=A100                 # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=16               # Request 16 CPU cores
#SBATCH --mem=64G                        # Request 64 GB of memory
#SBATCH --time=24:00:00                  # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/
RESULTS_DIR=./models/output/

# Evaluate ae_sbert_A on all test datasets (trained on OurAirports)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_A.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_A_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_A.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_A_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_A.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_A_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_A.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_A_on_D/

# Evaluate ae_sbert_B on all test datasets (trained on sakila)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_B.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_B_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_B.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_B_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_B.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_B_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_B.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_B_on_D/

# Evaluate ae_sbert_C on all test datasets (trained on AdventureWorks)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_C.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_C_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_C.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_C_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_C.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_C_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_C.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_C_on_D/

# Evaluate ae_sbert_D on all test datasets (trained on OHR)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_D.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_D_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_D.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_D_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_D.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_D_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_D.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_D_on_D/

# Print job completion time
echo "Job finished at: $(date)"
