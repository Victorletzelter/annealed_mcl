# Official code of *Annealed Multiple Choice Learning* (NeurIPS 2024)

Source code for the source separation experiments in the paper **Annealed Multiple Choice Learning: Overcoming limitations of Winner-takes-all with annealing** (NeurIPS 2024 Poster).

### Structure of the repository

```
.
├── README.md
├── synthetic_experiments.ipynb # Notebook for fast prototyping on the synthetic data experiments.
└── synthetic_and_uci_datasets # Full pipeline for reproducing the results on the synthetic and UCI datasets.
    └── ...
└── source_separation # Full pipeline for reproducing the results on the source separation experiments. 
    └── ...
└── images # Folder with gifs and images
    └── ...
...
```

### Synthetic example 
We propose a python notebook `synthetic_experiment.ipynb` with a toy example of the aMCL framework.

The full pipeline to reproduce the results in the paper on the synthetic datasets in given in the `synthetic_and_uci_datasets` folder. For reproducing the results, first create and activate a conda environment with 

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

The _fast suffix enables faster training than the original paper's setup, with visually similar plots. Remove _fast to train with the exact setup as described in the paper.

Please find below two animations that compares WTA and aMCL training dynamics and that shows how aMCL overcomes the limitations of the Winner-takes-all:

![](images/sgd_wta.gif)
*Winner-takes-all training dynamics with stochastic gradient descent (see Fig.1)*

![](images/sgd_amcl.gif)
*Annealed Multiple Choice Learning training dynamics with stochastic gradient descent (see Fig.1)*

### UCI Datasets

For reproducing the results on the UCI datasets, first create and activate a conda environment where you install the needed dependencies:

```shell
conda create -n uci_env -y -python=3.9.20
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

The benchmark follows the experimental protocol of preivous works [A,B]. The UCI regression datasets can be downloaded in this [drive](https://drive.google.com/drive/folders/16L5Dy9qw3StCY4AvtP98KA5xDZrtcHV3?usp=drive_link) [C].

Once the datasets are placed in the `synthetic_and_uci_datasets/data/uci' folder, the results can be reproduced from the following commands. 

```shell
cd ${MY_HOME}/scripts_uci ;
./1scripts_uci_train_and_eval_loop_protein.sh ;
./1scripts_uci_train_and_eval_loop_year.sh ;
./1scripts_uci_train_and_eval_loop.sh ;
./2scripts_uci_extract_results.sh
```

[A] Hernandez-Lobato, J. M. and Adams, R. Probabilistic back-propagation for scalable learning of bayesian neural networks. In ICML, pp. 1861–1869. PMLR, 2015.

[B] Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep
ensembles. In NeurIPS, volume 30, 2017.

[C] Han, X., Zheng, H., & Zhou, M. Card: Classification and regression diffusion models. In NeurIPS, volume 35, 2022. 

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

### Contribution

This code could be better organized and made more computationally efficient. Feel free to contribute to this repository or report any difficulties or bugs.

### Citation

If our work helped in your research, please consider citing us with the following bibtex code:

```
@article{amcl,
  title={Annealed Multiple Choice Learning: Overcoming limitations of Winner-takes-all with annealing},
  author={Perera, David and Letzelter, Victor and Mariotte, Th{\'e}o and Cort{\'e}s, Adrien and Chen, Mickael and Essid, Slim and Richard, Ga{\"e}l},
  journal={Advances in neural information processing systems},
  year={2024}
}
```
