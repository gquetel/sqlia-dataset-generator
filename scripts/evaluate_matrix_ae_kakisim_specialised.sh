# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
# source venv-3.12.3/bin/activate

TESTING=false  

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/ae_kakisim_specialised/
RESULTS_DIR=./models/output/ae_kakisim_specialised/

TESTING_FLAG=""
if [ "$TESTING" = "true" ]; then
    TESTING_FLAG="--testing"
fi

python3 models/training.py --dataset=$DATASETS_DIR/specialised-OurAirports.csv --models ae_kakisim --subfolder=ae_kakisim_specialised/specialised-OurAirports-ae_kakisim --save-model-path=$MODELS_DIR/ae_kakisim_A $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/specialised-sakila.csv --models ae_kakisim --subfolder=ae_kakisim_specialised/specialised-sakila-ae_kakisim --save-model-path=$MODELS_DIR/ae_kakisim_B $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --models ae_kakisim --subfolder=ae_kakisim_specialised/specialised-AdventureWorks-ae_kakisim --save-model-path=$MODELS_DIR/ae_kakisim_C $TESTING_FLAG
python3 models/training.py --dataset=$DATASETS_DIR/specialised-OHR.csv --models ae_kakisim --subfolder=ae_kakisim_specialised/specialised-OHR-ae_kakisim --save-model-path=$MODELS_DIR/ae_kakisim_D $TESTING_FLAG

# Evaluate ae_kakisim_D on all test datasets (trained on OHR)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_D.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_kakisim_D_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_D.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_kakisim_D_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_D.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_kakisim_D_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_D.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_kakisim_D_on_D/ --fixed-fpr=0.01

# Evaluate ae_kakisim_C on all test datasets (trained on AdventureWorks)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_C.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_kakisim_C_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_C.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_kakisim_C_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_C.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_kakisim_C_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_C.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_kakisim_C_on_D/ --fixed-fpr=0.01

# Evaluate ae_kakisim_B on all test datasets (trained on sakila)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_B.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_kakisim_B_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_B.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_kakisim_B_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_B.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_kakisim_B_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_B.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_kakisim_B_on_D/ --fixed-fpr=0.01

# Evaluate ae_kakisim_A on all test datasets (trained on OurAirports)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_A.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_kakisim_A_on_A/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_A.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_kakisim_A_on_B/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_A.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_kakisim_A_on_C/ --fixed-fpr=0.01
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_kakisim_A.pth --model-type=ae_kakisim --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_kakisim_A_on_D/ --fixed-fpr=0.01

# Print job completion time
echo "Job finished at: $(date)"
