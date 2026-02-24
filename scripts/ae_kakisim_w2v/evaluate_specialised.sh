#!/bin/bash
#SBATCH --job-name=ae_kakisim_w2v
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

TESTING=false

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/ae_kakisim_w2v_specialised/
RESULTS_DIR=./models/output/ae_kakisim_w2v_specialised/

TESTING_FLAG=""
if [ "$TESTING" = "true" ]; then
    TESTING_FLAG="--testing"
fi

# Get scenario from command line argument (1-4)
SCENARIO=${1:-1}

case $SCENARIO in
    1)
        MODEL_NAME="ae_kakisim_w2v_A"
        TRAIN_DATASET="specialised-OurAirports.csv"
        ;;
    2)
        MODEL_NAME="ae_kakisim_w2v_B"
        TRAIN_DATASET="specialised-sakila.csv"
        ;;
    3)
        MODEL_NAME="ae_kakisim_w2v_C"
        TRAIN_DATASET="specialised-AdventureWorks.csv"
        ;;
    4)
        MODEL_NAME="ae_kakisim_w2v_D"
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
    --models ae_kakisim_w2v \
    --subfolder=${TRAIN_DATASET%.csv}-ae_kakisim_w2v \
    --save-model-path=$MODELS_DIR/$MODEL_NAME \
    $TESTING_FLAG

# Evaluate on all test datasets
for TEST in "specialised-OurAirports.csv:A" "specialised-sakila.csv:B" "specialised-AdventureWorks.csv:C" "specialised-OHR.csv:D"; do
    TEST_FILE="${TEST%:*}"
    TEST_LABEL="${TEST#*:}"

    python3 experiments/evaluate_model.py \
        --model-path=$MODELS_DIR/${MODEL_NAME}.pth \
        --model-type=ae_kakisim_w2v \
        --test-dataset=$DATASETS_DIR/$TEST_FILE \
        --output-dir=$RESULTS_DIR/${MODEL_NAME}_on_${TEST_LABEL}/ \
        --fixed-fpr=0.01 \
        $TESTING_FLAG
done

echo "Job finished at: $(date)"
