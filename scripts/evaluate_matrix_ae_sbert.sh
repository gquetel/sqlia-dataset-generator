# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate

DATASETS_DIR=$HOME/datasets/100k-training/
MODELS_DIR=./models/output/models/
RESULTS_DIR=./models/output/

# Evaluate ae_sbert_ABC on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABC.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_ABC_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABC.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_ABC_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABC.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_ABC_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABC.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_ABC_on_D/

# Evaluate ae_sbert_ABD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_ABD_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_ABD_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_ABD_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ABD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_ABD_on_D/

# Evaluate ae_sbert_ACD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ACD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_ACD_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ACD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_ACD_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ACD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_ACD_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_ACD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_ACD_on_D/

# Evaluate ae_sbert_BCD on all test datasets
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_BCD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/OurAirports.csv --output-dir=$RESULTS_DIR/ae_sbert_BCD_on_A/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_BCD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/sakila.csv --output-dir=$RESULTS_DIR/ae_sbert_BCD_on_B/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_BCD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/AdventureWorks.csv --output-dir=$RESULTS_DIR/ae_sbert_BCD_on_C/
python3 experiments/evaluate_model.py --model-path=$MODELS_DIR/ae_sbert_BCD.pth --model-type=ae_sbert --test-dataset=$DATASETS_DIR/OHR.csv --output-dir=$RESULTS_DIR/ae_sbert_BCD_on_D/

# Print job completion time
echo "Job finished at: $(date)"
