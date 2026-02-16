#!/bin/bash
# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment (CPU)
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate
cd models/

DATASET_PATH=$HOME/datasets/100k-training/
MODELS_DIR=./output/models/ae_li_specialised/

python3 training.py --dataset=$DATASET_PATH/specialised-OurAirports.csv --models ae_li --subfolder=specialised-OurAirports-ae_li --save-model-path=$MODELS_DIR/ae_li_A
python3 training.py --dataset=$DATASET_PATH/specialised-sakila.csv --models ae_li --subfolder=specialised-sakila-ae_li --save-model-path=$MODELS_DIR/ae_li_B
python3 training.py --dataset=$DATASET_PATH/specialised-AdventureWorks.csv --models ae_li --subfolder=specialised-AdventureWorks-ae_li --save-model-path=$MODELS_DIR/ae_li_C
python3 training.py --dataset=$DATASET_PATH/specialised-OHR.csv --models ae_li --subfolder=specialised-OHR-ae_li --save-model-path=$MODELS_DIR/ae_li_D                                                                                         
# Print job completion timecd 
echo "Job finished at: $(date)"
