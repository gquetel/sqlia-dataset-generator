# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/
RESULTS_DIR=./models/output/

# Evaluate ae_li_D on all test datasets (trained on OHR)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_D.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_D_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_D.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_li_D_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_D.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_D_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_D.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_li_D_on_D/ 

# Evaluate ae_li_C on all test datasets (trained on AdventureWorks)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_C.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_C_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_C.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_li_C_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_C.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_C_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_C.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_li_C_on_D/

# Evaluate ae_li_B on all test datasets (trained on sakila)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_B.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_B_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_B.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_li_B_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_B.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_B_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_B.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_li_B_on_D/

# Evaluate ae_li_A on all test datasets (trained on OurAirports)
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_A.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OurAirports.csv --output-dir=$RESULTS_DIR/ae_li_A_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_A.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-sakila.csv --output-dir=$RESULTS_DIR/ae_li_A_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_A.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_li_A_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_li_A.pth --model-type=ae_li --test-dataset=$DATASETS_DIR/specialised-OHR.csv --output-dir=$RESULTS_DIR/ae_li_A_on_D/

# Print job completion time
echo "Job finished at: $(date)"
