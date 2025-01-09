#!/bin/bash
declare -a dataset_names=("protein")
declare -a hyp_counts=(5)
declare -a split_nums=(0 1 2 3 4)
declare -a temperatures_ini=(0.5)
declare -a epsilons=(0.05 0.2 0.3 0.5)
temperature_lim=0.0005
max_epochs="1000"
lr="0.01"
wta_after_temp_lim=True
scheduler_mode="exponential"
temperature_decay=0.95
optimizer=Adam
CKPT_JSON=${MY_HOME}/scripts_uci_eval/checkpoints_paths.json
epsilon_annealed=True

# open the json file
CKPT_PATHS=$(cat ${CKPT_JSON})
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
            key=${dataset_name}_vanilla_mcl_vwta_split_${split_num}
            ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
            if [[ -n "${ckpt_path}" ]]; then
                python eval.py ckpt_path=${ckpt_path} data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=vanilla_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=Eval_FULL_uci_${dataset_name}_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="wta" trainer.check_val_every_n_epoch=1
            else
                echo "Checkpoint path for ${key} not found"
            fi

            for temperature_ini in "${temperatures_ini[@]}"
            do
                if [[ "$scheduler_mode" == "linear" ]] # aMCL with linear scheduler
                then
                    key=${dataset_name}_amcl_scheduler_linear_split_${split_num}
                    ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
                    python eval.py ckpt_path=${ckpt_path} data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=annealed_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_temperature_${temperature_ini}_scheduler_${scheduler_mode}_optimizer_${optimizer}_temp_lim_${temperature_lim}_wta_lim seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=Eval_FULL_uci_${dataset_name}_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="stable_awta" model.hparams.temperature_ini=${temperature_ini} model.hparams.scheduler_mode=${scheduler_mode} model.hparams.optimizer=${optimizer} model.hparams.temperature_lim=${temperature_lim} model.hparams.wta_after_temperature_lim=True
                fi

                if [[ "$scheduler_mode" == "exponential" ]] # aMCL with exponential scheduler
                then
                    key=${dataset_name}_amcl_scheduler_exponential_split_${split_num}
                    ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
                    if [[ -n "${ckpt_path}" ]]; then
                        python eval.py ckpt_path=${ckpt_path} data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=annealed_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_temperature_${temperature_ini}_scheduler_${scheduler_mode}_optimizer_${optimizer}_decay_${temperature_decay}_temp_lim_${temperature_lim}_wta_lim seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=Eval_FULL_uci_${dataset_name}_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="stable_awta" model.hparams.temperature_ini=${temperature_ini} model.hparams.scheduler_mode=${scheduler_mode} model.hparams.temperature_decay=${temperature_decay} model.hparams.optimizer=${optimizer} model.hparams.temperature_lim=${temperature_lim} model.hparams.wta_after_temperature_lim=True
                    else
                        echo "Checkpoint path for ${key} not found"
                    fi
                fi
            done

            for epsilon in "${epsilons[@]}" # epsilon-WTA
            do
                key=${dataset_name}_relaxed_wta_epsilon_${epsilon}_split_${split_num}
                ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
                if [[ -n "${ckpt_path}" ]]; then
                    python eval.py ckpt_path=${ckpt_path} data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=epsilon_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_epsilon_${epsilon} seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=Eval_FULL_uci_${dataset_name}_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="wta-relaxed" model.hparams.training_epsilon=${epsilon}
                else
                    echo "Checkpoint path for ${key} not found"
                fi
            done

            if [[ "$epsilon_annealed" == "True" ]] # epsilon-WTA with annealed epsilon
            then
                key=${dataset_name}_relaxed_wta_epsilon_annealed_split_${split_num}
                ckpt_path=$(echo ${CKPT_PATHS} | jq -r --arg key "${key}" '.[$key]')
                if [[ -n "${ckpt_path}" ]]; then
                    epsilon_ini=0.5
                    epsilon_decay=0.99
                    python eval.py ckpt_path=${ckpt_path} data=uci_data.yaml experiment=uci/uci_${dataset_name}.yaml data.hparams.data_root=${MY_HOME}/data/uci model=WTABased.yaml model.num_hypothesis=${hyp_count} hydra.job.name=epsilon_annealed_wta-${hyp_count}-hyps-seed-1-split-${split_num}_${max_epochs}e_lr${lr}_0.8train_epsilon_${epsilon_ini}_scheduler_linear_optimizer_Adam_temp_lim_${temperature_lim}_wta_lim seed=1 data.hparams.split_num=${split_num} logger.mlflow.experiment_name=Eval_FULL_uci_${dataset_name}_${max_epochs}e_lr${lr} trainer.max_epochs=${max_epochs} model.hparams.learning_rate=${lr} model.hparams.training_wta_mode="wta-relaxed" model.hparams.temperature_lim=${temperature_lim} model.hparams.wta_after_temperature_lim=True model.hparams.annealed_epsilon=True model.hparams.epsilon_ini=${epsilon_ini} model.hparams.epsilon_decay=${epsilon_decay} model.hparams.scheduler_mode=linear
                else
                    echo "Checkpoint path for ${key} not found"
                fi
            fi
        done
    done
done