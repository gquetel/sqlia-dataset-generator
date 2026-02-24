#!/bin/bash
#SBATCH --job-name=ae_sbert_wafamole
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
source venv-3.12.12/bin/activate

TESTING=false

DATASETS_DIR=$HOME/datasets/100k-training/
WAFAMOLE_DATASET=$DATASETS_DIR/specialised-wafamole.csv
GENERIC_MODELS_DIR=./models/output/models/ae_sbert_generic/
SPECIALISED_MODELS_DIR=./models/output/models/ae_sbert_specialised/
GENERIC_RESULTS_DIR=./models/output/ae_sbert_generic/
SPECIALISED_RESULTS_DIR=./models/output/ae_sbert_specialised/

TESTING_FLAG=""
if [ "$TESTING" = "true" ]; then
    TESTING_FLAG="--testing"
fi

# ── Phase 1: Train E→E specialised model ──────────────────────────────────────
echo "Training ae_sbert_E (specialised on WAFAMOLE)..."
python3 models/training.py \
    --dataset=$WAFAMOLE_DATASET \
    --models ae_sbert \
    --subfolder=ae_sbert_specialised/specialised-wafamole-ae_sbert \
    --save-model-path=$SPECIALISED_MODELS_DIR/ae_sbert_E \
    $TESTING_FLAG

# ── Phase 2: Evaluate all 9 models on WAFAMOLE (E) ────────────────────────────

# Generic models (trained on 3 datasets, tested on E)
echo "Evaluating generic models on WAFAMOLE..."
for TRAIN in "ABC" "ABD" "ACD" "BCD"; do
    python3 experiments/evaluate_model.py \
        --model-path=$GENERIC_MODELS_DIR/ae_sbert_${TRAIN}.pth \
        --model-type=ae_sbert \
        --test-dataset=$WAFAMOLE_DATASET \
        --output-dir=$GENERIC_RESULTS_DIR/ae_sbert_${TRAIN}_on_E/ \
        --fixed-fpr=0.01
done

# Specialised models (trained on single dataset, tested on E)
echo "Evaluating specialised models on WAFAMOLE..."
for TRAIN in "A" "B" "C" "D" "E"; do
    python3 experiments/evaluate_model.py \
        --model-path=$SPECIALISED_MODELS_DIR/ae_sbert_${TRAIN}.pth \
        --model-type=ae_sbert \
        --test-dataset=$WAFAMOLE_DATASET \
        --output-dir=$SPECIALISED_RESULTS_DIR/ae_sbert_${TRAIN}_on_E/ \
        --fixed-fpr=0.01
done

echo "Job finished at: $(date)"
