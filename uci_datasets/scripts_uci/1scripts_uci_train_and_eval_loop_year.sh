#!/bin/bash
declare -a dataset_names=("year")
declare -a hyp_counts=(5)
declare -a split_nums=(0)
declare -a temperatures_ini=(0.5)
declare -a epsilons=(0.5)
max_epochs="1000"
lr="0.01"

MY_HOME="/home/victorletzelter/workspace/research-project/toy-rmcl" 
cd ${MY_HOME}/src
scheduler_mode="linear"

# Loop through the array of dataset names
for dataset_name in "${dataset_names[@]}"
do
    # Loop through the array of hypothesis counts
    for hyp_count in "${hyp_counts[@]}"
    do
        # Loop through the array of split numbers
        for split_num in "${split_nums[@]}"
        do
            python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} model.square_size=10 hydra.job.name=vanilla_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=A_0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.h_optimization=False model.hparams.training_wta_mode="wta" early_stopping.patience=1000 trainer.check_val_every_n_epoch=10

            for temperature_ini in "${temperatures_ini[@]}"
            do
                python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} model.square_size=10 hydra.job.name=annealed_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_temperature_${temperature_ini}_scheduler_${scheduler_mode} seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=A_0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.compute_nll=False model.hparams.h_optimization=False model.hparams.training_wta_mode="stable_awta" model.hparams.temperature_ini=${temperature_ini} model.hparams.scheduler_mode=${scheduler_mode} early_stopping.patience=1000 trainer.check_val_every_n_epoch=10
            done

            for epsilon in "${epsilons[@]}"
            do
                python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} model.square_size=10 hydra.job.name=epsilon_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_epsilon_${epsilon} seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=A_0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.compute_nll=False model.hparams.h_optimization=False model.hparams.training_wta_mode="wta-relaxed" model.hparams.training_epsilon=${epsilon} early_stopping.patience=1000 trainer.check_val_every_n_epoch=10
            done
        done
    done
done