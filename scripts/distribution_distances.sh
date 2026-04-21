#!/bin/bash
#SBATCH --job-name=dist-distances            # Name of your job
#SBATCH --output=logs/dist-distances/%x_%j.out      # Output file (%x for job name, %j for job ID)
#SBATCH --error=logs/dist-distances/%x_%j.err       # Error file
#SBATCH --partition=A40                   # Partition to submit to
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --cpus-per-task=16               # Request 16 CPU cores
#SBATCH --mem=64G                         # Request 64 GB of memory
#SBATCH --time=12:00:00                   # Time limit (hh:mm:ss)

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd ~/repos/sqlia-dataset/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate conda-env-3.12

DATASET_DIR=~/datasets/100k-training

srun python3 experiments/distribution_distances.py \
  --dataset A $DATASET_DIR/bcd-a.csv \
  --dataset B $DATASET_DIR/acd-b.csv \
  --dataset C $DATASET_DIR/abd-c.csv \
  --dataset D $DATASET_DIR/abc-d.csv \
  --extractor ae_li ae_loginov ae_securebert ae_kakisim_c ae_cv ae_roberta ae_codebert \
  --samples 1000 \

echo "Job finished at: $(date)"
