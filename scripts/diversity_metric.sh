#!/bin/bash
#SBATCH --job-name=diversity-metric   # Name of your job
#SBATCH --output=../output/results/%x_%j.out           # Output file (%x for job name, %j for job ID)
#SBATCH --error=../output/results/%x_%j.err            # Error file
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate conda-env-3.12
cd experiments/

DATASET_DIR=~/datasets/100k-training

DATASETS="\
  --dataset abd-c $DATASET_DIR/abd-c.csv \
  --dataset abc-d $DATASET_DIR/abc-d.csv \
  --dataset bcd-a $DATASET_DIR/bcd-a.csv \
  --dataset acd-b $DATASET_DIR/acd-b.csv \
  --dataset c-c $DATASET_DIR/c-c.csv \
  --dataset d-d $DATASET_DIR/d-d.csv \
  --dataset a-a $DATASET_DIR/a-a.csv \
  --dataset b-b $DATASET_DIR/b-b.csv"

# We run lexical and syntactic metrics on entire datasets
srun python3 ./diversity_metric.py $DATASETS --vocab --parse-trees --output-dir ../output/results/diversity_metric_lex_syn

# We run semantic diversity on 5k samples
srun python3 ./diversity_metric.py $DATASETS --samples 5000 --div-sem --output-dir ../output/results/diversity_metric_sem

# Print job completion time
echo "Job finished at: $(date)"
