# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.13.11/bin/activate
cd models/

DATASET_PATH=$HOME/datasets/100k-training/
MODELS_DIR=./output/models/

# Generic experiments - train on 3 datasets, save model for matrix evaluation
# generic-OurAirports = trained on BCD, generic-sakila = trained on ACD, etc.
python3 training.py --dataset=$DATASET_PATH/generic-OurAirports.csv --models ae_li --subfolder=generic-OurAirports-ae_li --save-model-path=$MODELS_DIR/ae_li_BCD 
python3 training.py --dataset=$DATASET_PATH/generic-sakila.csv --models ae_li --subfolder=generic-sakila-ae_li --save-model-path=$MODELS_DIR/ae_li_ACD 
python3 training.py --dataset=$DATASET_PATH/generic-AdventureWorks.csv --models ae_li --subfolder=generic-AdventureWorks-ae_li --save-model-path=$MODELS_DIR/ae_li_ABD 
python3 training.py --dataset=$DATASET_PATH/generic-OHR.csv --models ae_li --subfolder=generic-OHR-ae_li --save-model-path=$MODELS_DIR/ae_li_ABC 

# Print job completion time
echo "Job finished at: $(date)"
