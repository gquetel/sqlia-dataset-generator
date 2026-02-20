#!/bin/bash
#SBATCH --job-name=ocsvm_sbert-specialized # Name of your job
#SBATCH --output=%x_%j.out              # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err               # Error file
#SBATCH --partition=A100                # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=16              # Request 8 CPU cores
#SBATCH --mem=64G                       # Request 32 GB of memory
#SBATCH --time=24:00:00                 # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate
cd models/

DATASET_PATH=$HOME/datasets/100k-training/

# Train ocsvm_sbert on generic datasets
srun python3 ./training.py --dataset=$DATASET_PATH/generic-OurAirports.csv --models ocsvm_sbert --subfolder=ocsvm_sbert_generic/generic-OurAirports-ocsvm_sbert
srun python3 ./training.py --dataset=$DATASET_PATH/generic-sakila.csv --models ocsvm_sbert --subfolder=ocsvm_sbert_generic/generic-sakila-ocsvm_sbert
srun python3 ./training.py --dataset=$DATASET_PATH/generic-AdventureWorks.csv --models ocsvm_sbert --subfolder=ocsvm_sbert_generic/generic-AdventureWorks-ocsvm_sbert
srun python3 ./training.py --dataset=$DATASET_PATH/generic-OHR.csv --models ocsvm_sbert --subfolder=ocsvm_sbert_generic/generic-OHR-ocsvm_sbert

# Print job completion time
echo "Job finished at: $(date)"
