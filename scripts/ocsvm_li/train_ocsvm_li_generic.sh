# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
cd ~/repos/sqlia-dataset/
source venv-3.12.3/bin/activate
cd models/

DATASET_PATH=$HOME/datasets/100k-training/


# Generic experiments lame25    
python3 training.py --dataset=$DATASET_PATH/generic-OurAirports.csv --models ocsvm_li --subfolder=ocsvm_li_generic/generic-OurAirports-ocsvm_li
python3 training.py --dataset=$DATASET_PATH/generic-sakila.csv --models ocsvm_li --subfolder=ocsvm_li_generic/generic-sakila-ocsvm_li
python3 training.py --dataset=$DATASET_PATH/generic-AdventureWorks.csv --models ocsvm_li --subfolder=ocsvm_li_generic/generic-AdventureWorks-ocsvm_li
python3 training.py --dataset=$DATASET_PATH/generic-OHR.csv --models ocsvm_li --subfolder=ocsvm_li_generic/generic-OHR-ocsvm_li

# Print job completion time
echo "Job finished at: $(date)"
