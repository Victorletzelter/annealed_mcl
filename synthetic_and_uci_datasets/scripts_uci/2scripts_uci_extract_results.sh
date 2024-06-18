#!/bin/bash

declare -a dataset_names=("boston" "concrete" "energy" "kin8nm" "naval" "power_plant" "wine" "yacht" "protein" "year" "protein")

# Create the results directory if it does not exist
mkdir -p ${MY_HOME}/results_uci_annealing
# Go in the results directory
cd ${MY_HOME}/results_uci_annealing

# Delete the saved_csv directory if it exists
if [ -d "saved_csv" ]; then
    rm -r saved_csv
fi

mkdir saved_csv
cd ${MY_HOME}/scripts_uci
max_epochs="1000"
lr="0.01"

for dataset_name in "${dataset_names[@]}"
do
    python scripts_download_csv.py --experiment_name=A_0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} --save_dir=${MY_HOME}/results_uci_annealing/saved_csv
done