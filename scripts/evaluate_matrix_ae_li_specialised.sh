# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/
RESULTS_DIR=./models/output/

# Evaluate ae_li_ABCs on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABCs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_ABCs_on_A/ 
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABCs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_li_ABCs_on_B/  
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABCs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_ABCs_on_C/ 
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABCs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_li_ABCs_on_D/ 

# Evaluate ae_li_ABDs on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_ABDs_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_li_ABDs_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_ABDs_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ABDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_li_ABDs_on_D/

# Evaluate ae_li_ACDs on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ACDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_ACDs_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ACDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_li_ACDs_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ACDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_ACDs_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_ACDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_li_ACDs_on_D/

# Evaluate ae_li_BCDs on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_BCDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_BCDs_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_BCDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_li_BCDs_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_BCDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_BCDs_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_BCDs.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_li_BCDs_on_D/

# Print job completion time
echo "Job finished at: $(date)"
