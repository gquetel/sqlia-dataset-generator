#!/bin/bash
#SBATCH --job-name=ae_sbert-specialized # Name of your job
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
MODELS_DIR=./output/models/ae_sbert_specialised/

# Train ae_sbert on specialised datasets
srun python3 ./training.py --dataset=$DATASET_PATH/specialised-OurAirports.csv --models ae_sbert --subfolder=ae_sbert_specialised/specialised-OurAirports-ae_sbert --save-model-path=$MODELS_DIR/ae_sbert_A
srun python3 ./training.py --dataset=$DATASET_PATH/specialised-sakila.csv --models ae_sbert --subfolder=ae_sbert_specialised/specialised-sakila-ae_sbert --save-model-path=$MODELS_DIR/ae_sbert_B
srun python3 ./training.py --dataset=$DATASET_PATH/specialised-AdventureWorks.csv --models ae_sbert --subfolder=ae_sbert_specialised/specialised-AdventureWorks-ae_sbert --save-model-path=$MODELS_DIR/ae_sbert_C
srun python3 ./training.py --dataset=$DATASET_PATH/specialised-OHR.csv --models ae_sbert --subfolder=ae_sbert_specialised/specialised-OHR-ae_sbert --save-model-path=$MODELS_DIR/ae_sbert_D

# Print job completion time
echo "Job finished at: $(date)"
