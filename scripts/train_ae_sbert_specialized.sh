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

# Train ae_sbert on specialised datasets
srun python3 ./training.py --dataset=../datasets/specialised-OurAirports.csv --models ae_sbert --subfolder=ae_sbert-specialised-OurAirports
srun python3 ./training.py --dataset=../datasets/specialised-sakila.csv --models ae_sbert --subfolder=ae_sbert-specialised-sakila
srun python3 ./training.py --dataset=../datasets/specialised-AdventureWorks.csv --models ae_sbert --subfolder=ae_sbert-specialised-AdventureWorks
srun python3 ./training.py --dataset=../datasets/specialised-OHR.csv --models ae_sbert --subfolder=ae_sbert-specialised-OHR

# Print job completion time
echo "Job finished at: $(date)"
