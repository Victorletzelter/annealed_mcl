#!/bin/bash
declare -a dataset_names=("concrete" "energy" "kin8nm" "naval" "power_plant" "wine" "yacht") # Add your dataset names here
declare -a hyp_counts=(5)
declare -a split_nums=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19) # Add your split numbers here
declare -a temperatures_ini=(0.5)
declare -a epsilons=(0.1)
temperature_lim=0.0005
max_epochs="1000"
lr="0.01"
wta_after_temp_lim=True
scheduler_mode="exponential"
temperature_decay=0.95
optimizer=Adam

cd ${MY_HOME}/src

# Loop through the array of dataset names
for dataset_name in "${dataset_names[@]}"
do
    # Loop through the array of hypothesis counts
    for hyp_count in "${hyp_counts[@]}"
    do
        # Loop through the array of split numbers
        for split_num in "${split_nums[@]}"
        do
            # Vanilla WTA. The RMSE results should match those in Letzelter, Victor, et al. "Winner-takes-all learners are geometry-aware conditional density estimators." In ICML 2024.
            python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=vanilla_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=A_0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="wta" trainer.check_val_every_n_epoch=1

            for temperature_ini in "${temperatures_ini[@]}"
            do
                if [[ "$scheduler_mode" == "linear" ]] # aMCL with linear scheduler
                then
                    python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=annealed_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_temperature_${temperature_ini}_scheduler_${scheduler_mode}_optimizer_${optimizer}_wta_valid_temp_lim_${temperature_lim}_wta_lim seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=A_0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="stable_awta" model.hparams.temperature_ini=${temperature_ini} model.hparams.scheduler_mode=${scheduler_mode} model.hparams.optimizer=${optimizer} model.hparams.temperature_lim=${temperature_lim} model.hparams.wta_after_temperature_lim=True
                fi

                if [[ "$scheduler_mode" == "exponential" ]] # aMCL with exponential scheduler
                then
                    python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=annealed_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_temperature_${temperature_ini}_scheduler_${scheduler_mode}_optimizer_${optimizer}_decay_${temperature_decay}_wta_valid_temp_lim_${temperature_lim}_wta_lim seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=A_0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="stable_awta" model.hparams.temperature_ini=${temperature_ini} model.hparams.scheduler_mode=${scheduler_mode} model.hparams.temperature_decay=${temperature_decay} model.hparams.optimizer=${optimizer} model.hparams.temperature_lim=${temperature_lim} model.hparams.wta_after_temperature_lim=True
                fi
            done

            for epsilon in "${epsilons[@]}" # epsilon-WTA
            do
                python train.py data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=epsilon_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_epsilon_${epsilon} seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=A_0.8train_FULL_uci_${dataset_name}_hopt_scaled_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="wta-relaxed" model.hparams.training_epsilon=${epsilon}
            done
        done
    done
done