#!/bin/bash
# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment (CPU)
cd ~/repos/sqlia-dataset/
source venv-3.13.11/bin/activate
cd models/

DATASET_PATH=$HOME/datasets/100k-training/

python3 training.py --dataset=$DATASET_PATH/specialised-OurAirports.csv --models ocsvm_li --subfolder=specialised-OurAirports-ocsvm_li                                                                              
python3 training.py --dataset=$DATASET_PATH/specialised-sakila.csv --models ocsvm_li --subfolder=specialised-sakila-ocsvm_li                                                                              
python3 training.py --dataset=$DATASET_PATH/specialised-AdventureWorks.csv --models ocsvm_li --subfolder=specialised-AdventureWorks-ocsvm_li                                                                              
python3 training.py --dataset=$DATASET_PATH/specialised-OHR.csv --models ocsvm_li --subfolder=specialised-OHR-ocsvm_li                                                                                             
# Print job completion timecd 
echo "Job finished at: $(date)"
