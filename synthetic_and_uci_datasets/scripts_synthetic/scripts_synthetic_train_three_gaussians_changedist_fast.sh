#!/bin/bash

# TRAININGS

# Define experiments based on command-line argument
experiments=("three_gaussians_changedist")

declare -a temperatures_ini=(1.0)
cd ${MY_HOME}/src

max_epochs=101
hyps=("49 7 7")
train_linear=True
train_exp=False
optimizer=Adam

n_samples_train=1000
n_samples_val=1000
batch_size=1000
plot_training_freq=100
learning_rate=0.1

# Loop through each experiment
for experiment in "${experiments[@]}"; do
    # Loop through each hypothesis count and size pair
    for hyp in "${hyps[@]}"; do
        # Split hyp into number of hypotheses and histogram sizes
        IFS=' ' read -r num_hypotheses sizes <<< "$hyp"

        if [ "$train_exp" = "True" ]; then
            for temperature_ini in "${temperatures_ini[@]}" ; do
                for temperature_decay in "${temperatures_decay[@]}" ; do
                    # Score_based_WTA model with annealing exponential
                    python train.py data=synthetic_data.yaml model=WTABased.yaml experiment=A_${experiment}.yaml hydra.job.name=data_${experiment}_model_Score_based_WTA_${num_hypotheses}_hyps_annealed_exp_${temperature_ini}_${temperature_decay}_decay model.num_hypothesis=${num_hypotheses} trainer.max_epochs=${max_epochs} logger.mlflow.experiment_name=stable_Annealing_${experiment}_train_${max_epochs}_epochs model.hparams.training_wta_mode=stable_awta model.hparams.temperature_ini=${temperature_ini} model.hparams.temperature_decay=${temperature_decay} model.hparams.scheduler_mode=exponential model.hparams.plot_mode=True model.restrict_to_square=True model.hparams.learning_rate=${learning_rate} model.hparams.optimizer=${optimizer} data.hparams.n_samples_train=${n_samples_train} data.hparams.n_samples_val=${n_samples_val} data.batch_size=${batch_size} model.hparams.plot_training_frequency=${plot_training_freq}
                done
            done
        fi

        if [ "$train_linear" = "True" ]; then
            for temperature_ini in "${temperatures_ini[@]}" ; do
                # Score_based_WTA model with annealing linear
                python train.py data=synthetic_data.yaml model=WTABased.yaml experiment=A_${experiment}.yaml hydra.job.name=data_${experiment}_model_Score_based_WTA_${num_hypotheses}_hyps_annealed_linear_${temperature_ini}_ini model.num_hypothesis=${num_hypotheses} trainer.max_epochs=${max_epochs} logger.mlflow.experiment_name=stable_Annealing_${experiment}_train_${max_epochs}_epochs model.hparams.training_wta_mode=stable_awta model.hparams.temperature_ini=${temperature_ini} model.hparams.scheduler_mode=linear model.hparams.plot_mode=True model.restrict_to_square=True model.hparams.learning_rate=${learning_rate} model.hparams.optimizer=${optimizer} data.hparams.n_samples_train=${n_samples_train} data.hparams.n_samples_val=${n_samples_val} data.batch_size=${batch_size} model.hparams.plot_training_frequency=${plot_training_freq}
            done
        fi
    done
done