#!/bin/bash
#SBATCH --job-name=ae_kakisim_c_generic
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd ~/repos/sqlia-dataset/
source venv-3.12.12/bin/activate

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/ae_kakisim_c_generic/
RESULTS_DIR=./models/output/ae_kakisim_c_generic/

# Get scenario from command line argument (1-4)
SCENARIO=${1:-1}

case $SCENARIO in
    1)
        MODEL_NAME="ae_kakisim_c_BCD"
        TRAIN_DATASET="generic-OurAirports.csv"
        ;;
    2)
        MODEL_NAME="ae_kakisim_c_ACD"
        TRAIN_DATASET="generic-sakila.csv"
        ;;
    3)
        MODEL_NAME="ae_kakisim_c_ABD"
        TRAIN_DATASET="generic-AdventureWorks.csv"
        ;;
    4)
        MODEL_NAME="ae_kakisim_c_ABC"
        TRAIN_DATASET="generic-OHR.csv"
        ;;
    *)
        echo "Invalid scenario: $SCENARIO (must be 1-4)"
        exit 1
        ;;
esac

echo "Running scenario $SCENARIO: $MODEL_NAME"

# Train
python3 models/training.py \
    --dataset=$DATASETS_DIR/$TRAIN_DATASET \
    --models ae_kakisim_c \
    --subfolder=${TRAIN_DATASET%.csv}-ae_kakisim_c \
    --save-model-path=$MODELS_DIR/$MODEL_NAME

# Evaluate on all test datasets
for TEST in "generic-OurAirports.csv:A" "generic-sakila.csv:B" "generic-AdventureWorks.csv:C" "generic-OHR.csv:D"; do
    TEST_FILE="${TEST%:*}"
    TEST_LABEL="${TEST#*:}"

    python3 experiments/evaluate_model.py \
        --model-path=$MODELS_DIR/${MODEL_NAME}.pth \
        --model-type=ae_kakisim_c \
        --test-dataset=$DATASETS_DIR/$TEST_FILE \
        --output-dir=$RESULTS_DIR/${MODEL_NAME}_on_${TEST_LABEL}/ \
        --fixed-fpr=0.01
done

echo "Job finished at: $(date)"
