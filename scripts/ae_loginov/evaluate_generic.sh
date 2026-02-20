# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
# source venv-3.12.3/bin/activate

TESTING=false

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/ae_loginov_generic/
RESULTS_DIR=./models/output/ae_loginov_generic/

TESTING_FLAG=""
if [ "$TESTING" = "true" ]; then
    TESTING_FLAG="--testing"
fi

python3 models/training.py --dataset=$DATASETS_DIR/generic-OurAirports.csv --models ae_loginov --subfolder=ae_loginov_generic/generic-OurAirports-ae_loginov --save-model-path=$MODELS_DIR/ae_loginov_BCD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-sakila.csv --models ae_loginov --subfolder=ae_loginov_generic/generic-sakila-ae_loginov --save-model-path=$MODELS_DIR/ae_loginov_ACD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-AdventureWorks.csv --models ae_loginov --subfolder=ae_loginov_generic/generic-AdventureWorks-ae_loginov --save-model-path=$MODELS_DIR/ae_loginov_ABD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-OHR.csv --models ae_loginov --subfolder=ae_loginov_generic/generic-OHR-ae_loginov --save-model-path=$MODELS_DIR/ae_loginov_ABC $TESTING_FLAG

# Evaluate ae_loginov_ABC on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ABC.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_loginov_ABC_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ABC.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_loginov_ABC_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ABC.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_loginov_ABC_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ABC.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_loginov_ABC_on_D/ --fixed-fpr=0.01

# Evaluate ae_loginov_ABD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ABD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_loginov_ABD_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ABD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_loginov_ABD_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ABD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_loginov_ABD_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ABD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_loginov_ABD_on_D/ --fixed-fpr=0.01

# Evaluate ae_loginov_ACD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ACD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_loginov_ACD_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ACD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_loginov_ACD_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ACD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_loginov_ACD_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_ACD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_loginov_ACD_on_D/ --fixed-fpr=0.01

# Evaluate ae_loginov_BCD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_BCD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_loginov_BCD_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_BCD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_loginov_BCD_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_BCD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_loginov_BCD_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_loginov_BCD.pth --model-type=ae_loginov --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_loginov_BCD_on_D/ --fixed-fpr=0.01

# Print job completion time
echo "Job finished at: $(date)"
