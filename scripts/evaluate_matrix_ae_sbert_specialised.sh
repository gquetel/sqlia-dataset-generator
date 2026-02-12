#!/bin/bash
#SBATCH --job-name=ae_sbert
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/sbert_specialised/
RESULTS_DIR=./models/output/sbert_specialised/

# Get scenario from command line argument (1-4)
SCENARIO=${1:-1}

case $SCENARIO in
    1)
        # Train on A (OurAirports), evaluate on all
        MODEL_NAME="ae_sbert_A"
        TRAIN_DATASET="specialised-OurAirports.csv"
        ;;
    2)
        # Train on B (sakila), evaluate on all
        MODEL_NAME="ae_sbert_B"
        TRAIN_DATASET="specialised-sakila.csv"
        ;;
    3)
        # Train on C (AdventureWorks), evaluate on all
        MODEL_NAME="ae_sbert_C"
        TRAIN_DATASET="specialised-AdventureWorks.csv"
        ;;
    4)
        # Train on D (OHR), evaluate on all
        MODEL_NAME="ae_sbert_D"
        TRAIN_DATASET="specialised-OHR.csv"
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
    --models ae_sbert \
    --subfolder=${TRAIN_DATASET%.csv}-ae_sbert \
    --save-model-path=$MODELS_DIR/$MODEL_NAME

# Evaluate on all test datasets
for TEST in "specialised-OurAirports.csv:A" "specialised-sakila.csv:B" "specialised-AdventureWorks.csv:C" "specialised-OHR.csv:D"; do
    TEST_FILE="${TEST%:*}"
    TEST_LABEL="${TEST#*:}"
    echo $TEST
    python3 experiments/evaluate_model.py \
        --model-path=$MODELS_DIR/${MODEL_NAME}.pth \
        --model-type=ae_sbert \
        --test-dataset=$DATASETS_DIR/$TEST_FILE \
        --output-dir=$RESULTS_DIR/${MODEL_NAME}_on_${TEST_LABEL}/
done

echo "Job finished at: $(date)"