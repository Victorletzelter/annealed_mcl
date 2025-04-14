"""
This is a configuration file used to forward desired parameters to
the different modules used during training.
"""
import os
import random
import socket
import string
from models import get_default_config

TASK = "source-separation"
STUDY = "001"
STUDY_TAGS = ["fixed_num_source",]
EXP = "001"

cluster = socket.gethostname()
slurm = "SLURM_JOB_ID" in os.environ

# place to save the logs (tb logs)
log_dir_path = "./log"
# data directory with the CSV files to generate the mixtures
data_dir_path = "./data"

# Data setup as a function of number of sources and sample rate
n_src=2 #maximum number of sources to separate. If n_src=3 it will train on wsj0{2-3}mix
sample_rate=8000
mode="min"
EXP_TAGS = ["wsj0mix","sepclean_dprnn"]
metadata_dir = os.path.join(data_dir_path,f"var_num_speakers/wav{int(sample_rate/1e3)}k/min/metadata")
test_metadata_dir = os.path.join(data_dir_path,f"/wav{int(sample_rate/1e3)}k/min/metadata")
# Assign job_id to the run132615
if slurm:
  job_id = os.environ["SLURM_JOB_ID"]
else:
  job_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))

common_parameters = {
  #"model_name": "sudo_sc",
  "model_name": "dprnn_sc",
  "model": {
    "n_src": n_src,
    "count_stage": "late",
  },
    "optim": {
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0},
    "training": {
        "seed": 42,
        "max_epochs": 200,
        "batch_size": 22,
        "num_workers": int(os.environ["SLURM_CPUS_ON_NODE"]) if slurm else 10,
        "early_stop": True,
        "half_lr": True,
        "gradient_clip_val": 5.,
        "noise_schedule": None,
        "overfitting": False,
        "num_sanity_val_steps": 2,
        "loss": {
          "name": "pit_sisdr",
          "return_score_loss": True,
          "score_alpha": 0.1,
        }
    },
    "validation": {
      "metric_names": ["pit_sisdr","pit_sisdr_inference","emd_sisdr","mcl_sisdr","mcl_sisdr_inference"],
      "score_threshold": 0.5,
    },
    "data": {
        "train": {
          "csv_dir": os.path.join(metadata_dir,"tr"),
          "shuffle": True,
        },
        "val": {
          "csv_dir": os.path.join(metadata_dir,"cv"),
          "shuffle": False,
        },
        "test": {
          "csv_dir": os.path.join(metadata_dir,"tt"),  
          "shuffle": False,
        },
        "params": {"segment": 5, # 3
                   "sample_rate": sample_rate,
                   "n_src": n_src,
                   "min_snr": -5,
                   "max_snr": 5,
                   "limit_train_batches": 1.0,
                   "limit_val_batches": 1.0,
                   "limit_test_batches": 1.0,
                   }
            },
    "exp_dir": os.path.join(log_dir_path, STUDY+"_"+"_".join(STUDY_TAGS),
                            EXP + "_"+"_".join(EXP_TAGS)),
    "job_id": job_id,
    "cluster": cluster,
}

conf = {
  ###
  # 2 sources // PIT, EMD, MCL, aMCL
  ###
  "001": { # pit - sisdr - 2 speakers - 2 predictions
    "training": { "loss": { "name": "pit_sisdr", }, },
  },
  "002": { # emd - sisdr - 2 speakers - 2 predictions
    "training": { "loss": { "name": "emd_sisdr", }, },
  },
  "003": { # mcl - sisdr - 2 speakers - 2 predictions
    "training": { "loss": { "name": "mcl_sisdr", }, },
  },
  "004": { # amcl T=0.05*Var - sisdr - 2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 0.05, "epoch_end": 100, }, }, },
  },

  ###
  # 3 sources // PIT, EMD, MCL, aMCL
  ###
  "005": { # pit - sisdr - 3 speakers - 3 predictions
    "training": { "loss": { "name": "pit_sisdr", }, },
    "model" : { "n_src" : 3, },
    "data": { "params" : { "n_src_min": 3 , "n_src": 3}, },
  },
  "006": { # emd - sisdr - 3 speakers - 3 predictions
    "training": { "loss": { "name": "emd_sisdr", }, },
    "model" : { "n_src" : 3, },
    "data": { "params" : { "n_src_min": 3 , "n_src": 3}, },
  },
  "007": { # mcl - sisdr - 3 speakers - 3 predictions
    "training": { "loss": { "name": "mcl_sisdr", }, },
    "model" : { "n_src" : 3, },
    "data": { "params" : { "n_src_min": 3 , "n_src": 3}, },
  },
  "008": { # amcl T=0.05*Var - sisdr - 3 speakers - 3 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 0.05, "epoch_end": 100, }, }, },
    "model" : { "n_src" : 3, },
    "data": { "params" : { "n_src_min": 3 , "n_src": 3}, },
  },

  ###
  # 4 sources // PIT, EMD, MCL, aMCL
  ###
  "009": { # pit - sisdr - 4 speakers - 4 predictions
    "training": { "loss": { "name": "pit_sisdr", }, },
    "model" : { "n_src" : 4, },
    "data": { "params" : { "n_src_min": 4 , "n_src": 4}, },
  },
  "010": { # emd - sisdr - 4 speakers - 4 predictions
    "training": { "loss": { "name": "emd_sisdr", }, },
    "model" : { "n_src" : 4, },
    "data": { "params" : { "n_src_min": 4 , "n_src": 4}, },
  },
  "011": { # mcl - sisdr - 4 speakers - 4 predictions
    "training": { "loss": { "name": "mcl_sisdr", }, },
    "model" : { "n_src" : 4, },
    "data": { "params" : { "n_src_min": 4 , "n_src": 4}, },
  },
  "012": { # amcl T=0.05*Var - sisdr - 4 speakers - 4 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 0.05, "epoch_end": 100, }, }, },
    "model" : { "n_src" : 4, },
    "data": { "params" : { "n_src_min": 4 , "n_src": 4}, },
  },

  ###
  # Varying the temperature parameter in aMCL
  ###
  "013": { # amcl T=0.05*Var - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 0.05, "epoch_end": 100, }, },"early_stop": False },
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },
  "014": { # amcl T=0.5*Var - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 0.5, "epoch_end": 100, }, },"early_stop": False},
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },
  "015": { # amcl T=2*Var - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 2.0, "epoch_end": 100, }, },"early_stop": False },
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },
  "016": { # amcl T=10*Var - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 10, "epoch_end": 100, }, },"early_stop": False },
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },
  "017": { # amcl T=0.05*Var - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "exponential", "temperature_start": 0.05, "epoch_end": 100, }, },"early_stop": False },
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },
  "018": { # amcl T=0.5*Var - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "exponential", "temperature_start": 0.5, "epoch_end": 100, }, },"early_stop": False},
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },
  "019": { # amcl T=2*Var - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "exponential", "temperature_start": 2.0, "epoch_end": 100, }, },"early_stop": False },
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },
  "020": { # amcl T=10*Var - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "exponential", "temperature_start": 10, "epoch_end": 100, }, },"early_stop": False },
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },

  # Change the seed for MCL (collapse with 42)
  "021": { # mcl - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "mcl_sisdr", }, },
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },
  "022": { # mcl - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "mcl_sisdr", }, "seed": 21},
    "model" : { "n_src" : 2, },
    "data": { "params" : { "n_src_min": 2 , "n_src": 2}, },
  },

  # Increasing the number of hypotheses
  "023": { # amcl t=0.05*var - sisdr - 3-3 speakers - 3 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 0.05, "epoch_end": 100, }, }, },
    "model" : { "n_src" : 5, },
    "data": { "params" : { "n_src_min": 3 , "n_src": 3}, },
    "validation": { "metric_names": ["emd_sisdr","mcl_sisdr","mcl_sisdr_inference"], },

  },
  "024": { # amcl t=0.05*var - sisdr - 3-3 speakers - 3 predictions
    "training": { "loss": { "name": "amcl_sisdr", "temperature_schedule" : { "name": "linear", "temperature_start": 0.05, "epoch_end": 100, }, }, "batch_size": 18,},
    "model" : { "n_src" : 10, },
    "data": { "params" : { "n_src_min": 3 , "n_src": 3}, },
    "validation": { "metric_names": ["emd_sisdr","mcl_sisdr","mcl_sisdr_inference"], },

  },
  "025": { # mcl - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "mcl_sisdr", }, },
    "model" : { "n_src" : 5, },
    "data": { "params" : { "n_src_min": 3 , "n_src": 3}, },
    "validation": { "metric_names": ["emd_sisdr","mcl_sisdr","mcl_sisdr_inference"], },

  },
  "026": { # mcl - sisdr - 2-2 speakers - 2 predictions
    "training": { "loss": { "name": "mcl_sisdr", }, "batch_size": 20,},
    "model" : { "n_src" : 10, },
    "data": { "params" : { "n_src_min": 3 , "n_src": 3}, },
  },


}