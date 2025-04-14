# Annealed Multiple Choice Learning (NeurIPS 2024)

Official implementation of **Annealed Multiple Choice Learning: Overcoming limitations of Winner-takes-all with annealing** (NeurIPS 2024).

## Overview

This repository contains the code for reproducing the experiments from our paper, including synthetic data experiments, UCI dataset experiments, and source separation tasks.

## Repository Structure

### Structure of the repository

``` shell
├── synthetic_experiments.ipynb    # Quick start notebook with synthetic data examples
├── synthetic_and_uci_datasets/   # Code for synthetic and UCI dataset experiments
├── source_separation/           # Source separation experiment implementation
└── images/                     # Visualizations and animations
```

## Getting Started

### Synthetic Experiments

We provide two ways to explore the aMCL framework:

1. **Quick Start**: Use `synthetic_experiment.ipynb` for interactive experimentation and fast prototyping.
2. **Full Pipeline**: Reproduce paper results using the following steps:

First create and activate a conda environment with 

```shell
conda create -n synth_env -y python=3.9.20
conda init
```

Then, re-launch the shell and run:
```shell
conda activate synth_env
pip install -r synthetic_and_uci_datasets/requirements.txt
```

Then, define your absolute home path as environment variable

```shell
export MY_HOME=<YOUR_PATH>/annealed_mcl/synthetic_and_uci_datasets
```

Then, the training to be performed for reproducing the Figures 1, 2 and 4 of the main paper can be performed through the following commands:

```shell
cd ${MY_HOME}/scripts_synthetic ;
./scripts_synthetic_train_three_gaussians_fast.sh ; # Run this to reproduce results with three fixed Gaussians
./scripts_synthetic_train_three_gaussians_changedist_fast.sh ; # Run this to reproduce results with three moving Gaussians
```

Note: The _fast suffix enables faster training than the original paper's setup, with visually similar plots. Remove _fast to train with the exact setup as described in the paper.

Below are animations comparing WTA and aMCL training dynamics:

![WTA Training](images/sgd_wta.gif)
*Winner-takes-all training dynamics with stochastic gradient descent (see Fig.1)*

![aMCL Training](images/sgd_amcl.gif)
*Annealed Multiple Choice Learning training dynamics with stochastic gradient descent (see Fig.1)*

### UCI Datasets Experiments

1. **Setup Environment**

For reproducing the results on the UCI datasets, first create and activate a conda environment where you install the needed dependencies:

```shell
conda create -n uci_env -y python=3.9.20
conda init
```

Then, after reloading the shell:

```shell
conda activate uci_env
pip install -r synthetic_and_uci_datasets/requirements.txt
```

Then, define your absolute home path as environment variable

```shell
export MY_HOME=<YOUR_PATH>/annealed_mcl/synthetic_and_uci_datasets
```

2. **Data Preparation**:

- Download UCI datasets from this [drive](https://drive.google.com/drive/folders/16L5Dy9qw3StCY4AvtP98KA5xDZrtcHV3?usp=drive_link) [C].
- Place datasets in `synthetic_and_uci_datasets/data/uci/`

3. **Run Experiments**:

The benchmark follows the experimental protocol of previous works [A,B].

**Evaluation pipeline**:

- Download checkpoints from this [drive](https://drive.google.com/file/d/1eH9yV5Lex_vKJzaEOQx-mf_UGTcGLTNx/view?usp=drive_link).
- Place the checkpoints folder (named `checkpoints_uci_amcl`) in the synthetic_and_uci_datasets/ folder.
- Run the python scriptin  `synthetic_and_uci_datasets/scripts_uci_eval/extract_ckpt_json.py` to extract the checkpoints paths in a json file with ```python synthetic_and_uci_datasets/scripts_uci_eval/extract_ckpt_json.py```
- Run the following commands to train and evaluate the models:
```shell
cd ${MY_HOME}/scripts_uci_eval ;
./1scripts_uci_train_and_eval_loop_protein.sh ;
./1scripts_uci_train_and_eval_loop_year.sh ;
./1scripts_uci_train_and_eval_loop.sh ;
./2scripts_uci_extract_results.sh
```

**Training pipeline**:

- To train the models, run the following commands:
```shell
cd ${MY_HOME}/scripts_uci ;
./1scripts_uci_train_and_eval_loop_protein.sh ;
./1scripts_uci_train_and_eval_loop_year.sh ;
./1scripts_uci_train_and_eval_loop.sh ;
./2scripts_uci_extract_results.sh
```

Note that the results of the MCL baseline should match those presented in [D].

[A] Hernandez-Lobato, J. M. and Adams, R. Probabilistic back-propagation for scalable learning of bayesian neural networks. In ICML, pp. 1861–1869. PMLR, 2015.

[B] Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep
ensembles. In NeurIPS, volume 30, 2017.

[C] Han, X., Zheng, H., & Zhou, M. Card: Classification and regression diffusion models. In NeurIPS, volume 35, 2022. 

[D] Letzelter, Victor, David Perera, Cédric Rommel, Mathieu Fontaine, Slim Essid, Gael Richard, and Patrick Pérez. "Winner-takes-all learners are geometry-aware conditional density estimators." In IMCL, 2024.

### Source separation 

The code to train source separation models on the WSJ0-mix dataset is available in the `source_separation` repository. It is structured as follows:
- `data.py`: pytorch Dataset to load every version of the Wall Street Jounal mix dataset. This dataset can load versions with a fixed number of sources, but also with a variable number of speakers.
- `losses.py`: every losses that are presented in  the paper and some additional ones are implemented here:
  - PIT
  - MCL
  - aMCL
  - EMD
- `models.py`: implements the separation models. Note that our systems have an additional scoring head which is only useful in the context of separating a variable number of sources. 
  - DPRNN used in the paper
  - Sudo Rm-Rf: SOTA separation architecture for low resources source separation used for additional experiments, not presented in the paper
- `system.py`: Pytorch Lightning system to train and evaluate the separation models
- `train.py`: code to launch to train the systems. It manages the configurations and the exeriments saving.
- `utils.py`: various utilitaries
- `config_dprnn.py`: configuration file for DPRNN with each configuration for the experiments conducted in the paper.
- `config_sudo.py`: same with SudoRm-Rf 

### Environment

To install the dependencies, please use the following commands:
```
conda create -n annealed_mcl python=3.8
conda activate annealed_mcl
conda install pytorch==2.0.0 pytorch-cuda==11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### Data preparation

Metadata for the wsj0-mix dataset are provided. In order to reproduce our experiments, download the wsj0-mix dataset, and update the metadata to specify the location of the downloaded audio files using the command:
```
python update_metadata.py /wsj0-mix/root/path
```

### Training model with a given configuration

To train a model:
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

### Contribution

We welcome contributions! Please feel free to:
- Submit issues for bugs or difficulties
- Create pull requests for improvements
- Suggest better organization or efficiency improvements

### Citation

If our work helped in your research, fell free to give us a star ⭐ or to cite us with the following bibtex code:

```
@article{amcl,
  title={Annealed Multiple Choice Learning: Overcoming limitations of Winner-takes-all with annealing},
  author={Perera, David and Letzelter, Victor and Mariotte, Th{\'e}o and Cort{\'e}s, Adrien and Chen, Mickael and Essid, Slim and Richard, Ga{\"e}l},
  journal={Advances in neural information processing systems},
  year={2024}
}
```
