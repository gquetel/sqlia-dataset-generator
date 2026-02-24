# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
# source venv-3.12.12/bin/activate

TESTING=false  

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/ae_li_generic/
RESULTS_DIR=./models/output/ae_li_generic/

TESTING_FLAG=""
if [ "$TESTING" = "true" ]; then
    TESTING_FLAG="--testing"
fi

python3 models/training.py --dataset=$DATASETS_DIR/generic-OurAirports.csv --models ae_li --subfolder=ae_li_generic/generic-OurAirports-ae_li --save-model-path=$MODELS_DIR/ae_li_BCD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-sakila.csv --models ae_li --subfolder=ae_li_generic/generic-sakila-ae_li --save-model-path=$MODELS_DIR/ae_li_ACD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-AdventureWorks.csv --models ae_li --subfolder=ae_li_generic/generic-AdventureWorks-ae_li --save-model-path=$MODELS_DIR/ae_li_ABD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-OHR.csv --models ae_li --subfolder=ae_li_generic/generic-OHR-ae_li --save-model-path=$MODELS_DIR/ae_li_ABC $TESTING_FLAG

# Evaluate ae_li_ABC on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABC.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_ABC_on_A/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABC.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_li_ABC_on_B/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABC.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_ABC_on_C/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABC.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_li_ABC_on_D/ --fixed-fpr=0.01 $TESTING_FLAG

# Evaluate ae_li_ABD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_ABD_on_A/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_li_ABD_on_B/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_ABD_on_C/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_li_ABD_on_D/ --fixed-fpr=0.01 $TESTING_FLAG

# Evaluate ae_li_ACD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ACD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_ACD_on_A/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ACD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_li_ACD_on_B/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ACD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_ACD_on_C/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ACD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_li_ACD_on_D/ --fixed-fpr=0.01 $TESTING_FLAG

# Evaluate ae_li_BCD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_BCD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_BCD_on_A/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_BCD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_li_BCD_on_B/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_BCD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_BCD_on_C/ --fixed-fpr=0.01 $TESTING_FLAG
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_BCD.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_li_BCD_on_D/ --fixed-fpr=0.01 $TESTING_FLAG

# Print job completion time
echo "Job finished at: $(date)"
