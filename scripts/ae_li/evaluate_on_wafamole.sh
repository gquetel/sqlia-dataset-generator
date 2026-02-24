# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
# source venv-3.12.12/bin/activate

TESTING=false

DATASETS_DIR=$HOME/datasets/100k-training/
WAFAMOLE_DATASET=$DATASETS_DIR/specialised-wafamole.csv
GENERIC_MODELS_DIR=./models/output/models/ae_li_generic/
SPECIALISED_MODELS_DIR=./models/output/models/ae_li_specialised/
GENERIC_RESULTS_DIR=./models/output/ae_li_generic/
SPECIALISED_RESULTS_DIR=./models/output/ae_li_specialised/

TESTING_FLAG=""
if [ "$TESTING" = "true" ]; then
    TESTING_FLAG="--testing"
fi

echo "Training ae_li_E (specialised on WAFAMOLE)..."
python3 models/training.py \
    --dataset=$WAFAMOLE_DATASET \
    --models ae_li \
    --subfolder=ae_li_specialised/specialised-wafamole-ae_li \
    --save-model-path=$SPECIALISED_MODELS_DIR/ae_li_E \
    $TESTING_FLAG


# Generic models (trained on 3 datasets, tested on E)
echo "Evaluating generic models on WAFAMOLE..."
python3 experiments/evaluate_model.py --model-path=$GENERIC_MODELS_DIR/ae_li_ABC.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$GENERIC_RESULTS_DIR/ae_li_ABC_on_E/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$GENERIC_MODELS_DIR/ae_li_ABD.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$GENERIC_RESULTS_DIR/ae_li_ABD_on_E/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$GENERIC_MODELS_DIR/ae_li_ACD.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$GENERIC_RESULTS_DIR/ae_li_ACD_on_E/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$GENERIC_MODELS_DIR/ae_li_BCD.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$GENERIC_RESULTS_DIR/ae_li_BCD_on_E/ --fixed-fpr=0.01

# Specialised models (trained on single dataset, tested on E)
echo "Evaluating specialised models on WAFAMOLE..."
python3 experiments/evaluate_model.py --model-path=$SPECIALISED_MODELS_DIR/ae_li_A.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$SPECIALISED_RESULTS_DIR/ae_li_A_on_E/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$SPECIALISED_MODELS_DIR/ae_li_B.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$SPECIALISED_RESULTS_DIR/ae_li_B_on_E/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$SPECIALISED_MODELS_DIR/ae_li_C.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$SPECIALISED_RESULTS_DIR/ae_li_C_on_E/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$SPECIALISED_MODELS_DIR/ae_li_D.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$SPECIALISED_RESULTS_DIR/ae_li_D_on_E/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$SPECIALISED_MODELS_DIR/ae_li_E.pth --model-type=ae_li --test-dataset=$WAFAMOLE_DATASET --output-dir=$SPECIALISED_RESULTS_DIR/ae_li_E_on_E/ --fixed-fpr=0.01

# Print job completion time
echo "Job finished at: $(date)"
