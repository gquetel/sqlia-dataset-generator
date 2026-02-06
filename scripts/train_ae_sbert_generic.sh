#!/bin/bash
#SBATCH --job-name=ae_sbert_generic   # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
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
cd models/

DATASET_PATH=$HOME/datasets/100k-training/
MODELS_DIR=./output/models/

# Train ae_sbert on generic datasets - save model for matrix evaluation
# generic-OurAirports = trained on BCD, generic-sakila = trained on ACD, etc.

srun python3 ./training.py --dataset=$DATASET_PATH/generic-OurAirports.csv --models ae_sbert --subfolder=generic-OurAirports-ae_sbert --save-model-path=$MODELS_DIR/ae_sbert_BCD
srun python3 ./training.py --dataset=$DATASET_PATH/generic-sakila.csv --models ae_sbert --subfolder=generic-sakila-ae_sbert --save-model-path=$MODELS_DIR/ae_sbert_ACD
srun python3 ./training.py --dataset=$DATASET_PATH/generic-AdventureWorks.csv --models ae_sbert --subfolder=generic-AdventureWorks-ae_sbert --save-model-path=$MODELS_DIR/ae_sbert_ABD
srun python3 ./training.py --dataset=$DATASET_PATH/generic-OHR.csv --models ae_sbert --subfolder=generic-OHR-ae_sbert --save-model-path=$MODELS_DIR/ae_sbert_ABC

# Print job completion time
echo "Job finished at: $(date)"
