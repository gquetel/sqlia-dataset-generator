#!/bin/bash
#SBATCH --job-name=domain-leakage            # Name of your job
#SBATCH --output=logs/domain-leakage/%x_%j.out      # Output file (%x for job name, %j for job ID)
#SBATCH --error=logs/domain-leakage/%x_%j.err       # Error file
#SBATCH --partition=A40                   # Partition to submit to
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --cpus-per-task=16               # Request 16 CPU cores
#SBATCH --mem=64G                         # Request 32 GB of memory
#SBATCH --time=12:00:00                   # Time limit (hh:mm:ss)

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd ~/repos/sqlia-dataset/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate conda-env-3.12

DATASET_DIR=~/datasets/100k-training

srun python3 experiments/domain_leakage_rf.py \
  --dataset A $DATASET_DIR/specialised-OurAirports.csv \
            B $DATASET_DIR/specialised-sakila.csv \
            C $DATASET_DIR/specialised-AdventureWorks.csv \
            D $DATASET_DIR/specialised-OHR.csv \
  --extractor ae_li ae_loginov ae_securebert ae_kakisim_c ae_w2v ae_cv ae_roberta ae_bilstm_w2v ae_gaur ae_codebert \
  --train-samples 10000 \
  --test-samples 1000 \
  --normal-only \

echo "Job finished at: $(date)"
