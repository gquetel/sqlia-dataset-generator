# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
# source venv-3.12.3/bin/activate

TESTING=false

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/ae_kakisim_enriched_generic/
RESULTS_DIR=./models/output/ae_kakisim_enriched_generic/

TESTING_FLAG=""
if [ "$TESTING" = "true" ]; then
    TESTING_FLAG="--testing"
fi

python3 models/training.py --dataset=$DATASETS_DIR/generic-OurAirports.csv --models ae_kakisim_enriched --subfolder=ae_kakisim_enriched_generic/generic-OurAirports-ae_kakisim_enriched --save-model-path=$MODELS_DIR/ae_kakisim_enriched_BCD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-sakila.csv --models ae_kakisim_enriched --subfolder=ae_kakisim_enriched_generic/generic-sakila-ae_kakisim_enriched --save-model-path=$MODELS_DIR/ae_kakisim_enriched_ACD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-AdventureWorks.csv --models ae_kakisim_enriched --subfolder=ae_kakisim_enriched_generic/generic-AdventureWorks-ae_kakisim_enriched --save-model-path=$MODELS_DIR/ae_kakisim_enriched_ABD $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/generic-OHR.csv --models ae_kakisim_enriched --subfolder=ae_kakisim_enriched_generic/generic-OHR-ae_kakisim_enriched --save-model-path=$MODELS_DIR/ae_kakisim_enriched_ABC $TESTING_FLAG

# Evaluate ae_kakisim_enriched_ABC on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ABC.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ABC_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ABC.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ABC_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ABC.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ABC_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ABC.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ABC_on_D/ --fixed-fpr=0.01

# Evaluate ae_kakisim_enriched_ABD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ABD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ABD_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ABD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ABD_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ABD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ABD_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ABD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ABD_on_D/ --fixed-fpr=0.01

# Evaluate ae_kakisim_enriched_ACD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ACD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ACD_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ACD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ACD_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ACD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ACD_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_ACD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_ACD_on_D/ --fixed-fpr=0.01

# Evaluate ae_kakisim_enriched_BCD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_BCD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-OurAirports.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_BCD_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_BCD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-sakila.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_BCD_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_BCD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_BCD_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_enriched_BCD.pth --model-type=ae_kakisim_enriched --test-dataset=$DATASETS_DIR/generic-OHR.csv --output-dir=$RESULTS_DIR/ae_kakisim_enriched_BCD_on_D/ --fixed-fpr=0.01

# Print job completion time
echo "Job finished at: $(date)"
