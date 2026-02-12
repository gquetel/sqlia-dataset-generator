#!/bin/bash
#SBATCH --job-name=diversity-metric   # Name of your job
#SBATCH --output=../output/%x_%j.out           # Output file (%x for job name, %j for job ID)
#SBATCH --error=../output/%x_%j.err            # Error file
#SBATCH --partition=A30              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=16           # Request 16 CPU cores
#SBATCH --mem=32G                    # Request 64 GB of memory
#SBATCH --time=12:00:00              # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate
cd experiments/

DATASET_DIR=~/datasets/100k-training

DATASETS="\
  --dataset generic-AdventureWorks $DATASET_DIR/generic-AdventureWorks.csv \
  --dataset generic-OHR $DATASET_DIR/generic-OHR.csv \
  --dataset generic-OurAirports $DATASET_DIR/generic-OurAirports.csv \
  --dataset generic-sakila $DATASET_DIR/generic-sakila.csv \
  --dataset specialised-AdventureWorks $DATASET_DIR/specialised-AdventureWorks.csv \
  --dataset specialised-OHR $DATASET_DIR/specialised-OHR.csv \
  --dataset specialised-OurAirports $DATASET_DIR/specialised-OurAirports.csv \
  --dataset specialised-sakila $DATASET_DIR/specialised-sakila.csv"

# We run lexical and syntactic metrics on entire datasets
srun python3 ./diversity_metric.py $DATASETS --vocab --parse-trees --output-dir ../output/diversity_metrics_lex_syn

# We run semantic diversity on 5k samples
srun python3 ./diversity_metric.py $DATASETS --samples 5000 --div-sem --output-dir ../output/diversity_metrics_sem

# Print job completion time
echo "Job finished at: $(date)"
