echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd ~/repos/sqlia-dataset/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate conda-env-3.12

DATASET_DIR=~/datasets/100k-training

python3 experiments/h3_tokenizer_analysis.py \
  --dataset A $DATASET_DIR/bcd-a.csv \
  --dataset B $DATASET_DIR/acd-b.csv \
  --dataset C $DATASET_DIR/abd-c.csv \
  --dataset D $DATASET_DIR/abc-d.csv \

echo "Job finished at: $(date)"
