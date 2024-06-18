# Official code of *Annealed Multiple Choice Learning*

Source code for the source separation experiments in the paper **Annealed Multiple Choice Learning: Overcoming limitations of Winner-takes-all with annealing**.

## Structure of the repository

### Synthetic example 
We propose a python notebook `synthetic_experiment.ipynb` with a toy example of the aMCL framework.

### UCI Datasets

For reproducing the results on the UCI datasets, first create a conda environment where you install the needed dependencies:

```shell
conda create -n uci_env
conda activate uci_env
pip install -r uci_datasets/requirements.txt
```

Then, define your absolute home path as environment variable

```shell
export MY_HOME=<YOUR_PATH>/annealed_mcl/uci_datasets 
```

The results can be reproduced by running the following scripts:

```shell
cd ${MY_HOME}/scripts_uci ;
./1scripts_uci_train_and_eval_loop_protein.sh ;
./1scripts_uci_train_and_eval_loop_year.sh ;
./1scripts_uci_train_and_eval_loop.sh ;
./2scripts_uci_extract_results.sh
```

### Source separation 
The code to train source separation models on the WSJ0-mix dataset is available in the `source_separation` repository. It is structured as follows:
- `data.py`: pytorch Dataset to load every version of the Wall Street Jounal mix dataset. This dataset can load versions with a fixed number of sources, but also with a variable number of speakers.
- `losses.py`: every losses that are presented in  the paper and some additional ones are implemented here:
  - PIT
  - MCL
  - aMCL
  - EMD
- `models.py`: imlpements the seprartion models. Note that our systems have an additional scoring head which is only useful in the context of separating a variable number of sources. 
  - DPRNN used in the paper
  - Sudo Rm-Rf: SOTA separation architecture for low resources source separation used for additional experiments, not presented in the paper
- `system.py`: Pytorch Lightning system to train and evaluate the sepraration models
- `train.py`: code to launch to train the systems. It manages the configurations and the exeriments saving.
- `utils.py`: various utilitaries
- `config_dprnn.py`: configuration file for DPRNN with each configuration for the experiments conducted in the paper.
- `config_sudo.py`: same with SudoRm-Rf 

### Training model with a given configuration

To train a model, you just have to execute the followong command
```
python train.py --conf_id 001 --backbone dprnn
```

### Main dependencies

```
# Name                    Version      
asteroid                  0.6.1.dev0               
numpy                     1.24.4                   
pandas                    2.0.3                   
pot                       0.9.3                   
python                    3.8.18              
pytorch-lightning         1.7.7                    
scikit-learn              1.3.2                    
scipy                     1.10.1                   
torch                     1.13.1                   
torchaudio                0.13.1                   
```
