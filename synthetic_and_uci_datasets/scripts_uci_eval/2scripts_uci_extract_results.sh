#!/bin/bash

declare -a dataset_names=("boston" "concrete" "energy" "kin8nm" "naval" "power_plant" "wine" "yacht" "protein" "year")

# Create the results directory if it does not exist
mkdir -p ${MY_HOME}/results_uci_annealing
# Go in the results directory
cd ${MY_HOME}/results_uci_annealing

# Delete the saved_csv directory if it exists
if [ -d "saved_csv_eval" ]; then
    rm -r saved_csv_eval
fi

mkdir saved_csv_eval
cd ${MY_HOME}/scripts_uci_eval
max_epochs="1000"
lr="0.01"

for dataset_name in "${dataset_names[@]}"
do
    python scripts_download_csv.py --experiment_name=Eval_FULL_uci_${dataset_name}_${max_epochs}e_lr${lr} --save_dir=${MY_HOME}/results_uci_annealing/saved_csv_eval
done