echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd ~/repos/sqlia-dataset/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate conda-env-3.12

DATASET_DIR=~/datasets/100k-training

python3 experiments/h3_tokenizer_analysis.py \
  --dataset A $DATASET_DIR/generic-OurAirports.csv \
  --dataset B $DATASET_DIR/generic-sakila.csv \
  --dataset C $DATASET_DIR/generic-AdventureWorks.csv \
  --dataset D $DATASET_DIR/generic-OHR.csv \

echo "Job finished at: $(date)"
