import os
import argparse
import json
from pprint import pprint
from config_sudo import common_parameters as sudo_param 
from config_sudo import conf as sudo_conf
from config_dprnn import common_parameters as dprnn_param 
from config_dprnn import conf as dprnn_conf

import yaml
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils import merge_dicts


from system import MCL_System

from models import select_model, get_default_config
from losses import select_loss, select_metrics
from data import Wsj0mixDatasetVarSpk

#from asteroid.data import Wsj0mixDataset
from asteroid.engine.optimizers import make_optimizer
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models.dprnn_tasnet import DPRNNTasNet
from asteroid.models.sudormrf import SuDORMRFNet

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
#parser = argparse.ArgumentParser()
#parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
#parser.add_argument("--ckpt", default="")



parser = argparse.ArgumentParser()
parser.add_argument("--conf_id", default="001",
          help="Conf tag, used to get the right config")
parser.add_argument("--debug", type=bool, default=False,
          help="If true save to specific directory")
parser.add_argument("--ckpt",default=None,type=str,help="Path to the checkpoint from which to start the training ('last' or specific path)")
parser.add_argument("--backbone", default="sudo", choices=["sudo","dprnn"],type=str)
def main(conf):

  conf_id = conf["conf_id"]
  if conf["debug"]:
    exp_dir = os.path.join(conf["exp_dir"], "DEBUG")
  else:
    exp_dir = os.path.join(conf["exp_dir"], f"conf_id_{conf_id}", conf["job_id"])
  pl.seed_everything(conf["training"]["seed"], workers=True)
  
  accelerator = "gpu" if torch.cuda.is_available() else "cpu"
  device = "cuda" if torch.cuda.is_available() else "cpu"

  train_set = Wsj0mixDatasetVarSpk(
    csv_dir=conf["data"]["train"]["csv_dir"],
    min_n_src=conf["data"]["params"].get("n_src_min",2),
    max_n_src=conf["data"]["params"]["n_src"],
    sample_rate=conf["data"]["params"]["sample_rate"],
    segment=conf["data"]["params"]["segment"],
    random_crop=(not conf["training"]["overfitting"]),
  )
  val_set = Wsj0mixDatasetVarSpk(
    csv_dir=conf["data"]["val"]["csv_dir"] if (not conf["training"]["overfitting"]) else conf["data"]["train"]["csv_dir"],
    min_n_src=conf["data"]["params"].get("n_src_min",2),
    max_n_src=conf["data"]["params"]["n_src"],
    sample_rate=conf["data"]["params"]["sample_rate"],
    segment=conf["data"]["params"]["segment"],
    random_crop=(not conf["training"]["overfitting"]),
  )
  train_loader = DataLoader(
    train_set,
    shuffle=conf["data"]["train"]["shuffle"],
    batch_size=conf["training"]["batch_size"],
    num_workers=conf["training"]["num_workers"],
    drop_last=True,
    prefetch_factor=2,
    pin_memory=True,
  )
  val_loader = DataLoader(
    val_set,
    shuffle=conf["data"]["val"]["shuffle"],
    batch_size=conf["training"]["batch_size"],
    num_workers=conf["training"]["num_workers"],
    drop_last=True,
  )

  # handle multiple test datasets
  n_src = conf["data"]["params"]["n_src"]
  n_src_min =     min_n_src=conf["data"]["params"].get("n_src_min",2)
  test_loader_list = []
  n_src_list = [(2, n_src)] + list(zip(range(2, n_src+1), range(2, n_src+1))) if n_src > n_src_min else [(n_src_min, n_src_min)]
  for min_n_src, max_n_src in n_src_list:
    test_set = Wsj0mixDatasetVarSpk(
        csv_dir=conf["data"]["test"]["csv_dir"],
        min_n_src=min_n_src, 
        max_n_src=max_n_src,
        sample_rate=conf["data"]["params"]["sample_rate"],
        segment=None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=conf["data"]["test"]["shuffle"],
        batch_size=1,
        num_workers=conf["training"]["num_workers"],
        drop_last=False,
    )
    test_loader_list.append(test_loader)
  
  conf["model"] = get_default_config(model_name=conf["model_name"], n_src=conf["model"]["n_src"], count_stage=conf["model"]["count_stage"])
  model = select_model(conf["model_name"],**conf["model"])

  optimizer = make_optimizer(model.parameters(), **conf["optim"])

  # Define scheduler
  scheduler = None
  if conf["training"]["half_lr"]:
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
  
  # Just after instantiating, save the args. Easy loading in the future.
  os.makedirs(exp_dir, exist_ok=True)
  conf_path = os.path.join(exp_dir, "conf.yml")
  with open(conf_path, "w") as outfile:
    yaml.safe_dump(conf, outfile)

  loss_func = select_loss(
    conf["training"]["loss"]["name"], 
    max_epochs=conf["training"]["max_epochs"], 
    dataloader=train_loader, 
    limit_batches=conf["data"]["params"]["limit_train_batches"], # limit batches for variance computation
    device=device,
    **conf["training"]["loss"])
  metric_dict = select_metrics(conf["validation"]["metric_names"], score_threshold=conf["validation"]["score_threshold"])
  
  if conf["training"]["noise_schedule"] == "linear":
    max_epochs, noise_duration, noise_start = conf["training"]["max_epochs"], conf["training"]["noise_duration"], conf["training"]["noise_start"]
    def noise_schedule(x, epoch, noise_duration=noise_duration, noise_start=noise_start):
      std = torch.linspace(noise_start, 0, int(noise_duration * max_epochs))[epoch] if epoch < noise_duration * max_epochs else 0
      noise = torch.normal(0, std, size=x.shape, device=x.device)         
      return x + noise
  else:
    noise_schedule = None

  system = MCL_System(
      model=model,
      loss_func=loss_func,
      metric_dict=metric_dict,
      optimizer=optimizer,
      train_loader=train_loader,
      val_loader=val_loader,
      test_loader=test_loader_list,
      scheduler=scheduler,
      noise_schedule=noise_schedule,
      score_alpha=conf["training"]["loss"]["score_alpha"],
      config=conf,
      batch_example=val_set[0],
    )

  # Define callbacks
  callbacks = []
  checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
  checkpoint = ModelCheckpoint(
    checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
  )
  callbacks.append(checkpoint)
  if conf["training"]["early_stop"]:
    callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

  if conf["ckpt"] is not None:
    
    if conf["ckpt"] == "last":
      # get last checkpoint of all run of current experiment
      conf_dir = "/".join(exp_dir.split("/")[:-1])
      ckpt_path_list = []
      for dirname in os.listdir(conf_dir):
        ckpt_folder_path = os.path.join(conf_dir, dirname, "checkpoints")
        if os.path.isdir(ckpt_folder_path):
          for ckpt_name in os.listdir(ckpt_folder_path):
            ckpt_path = os.path.join(ckpt_folder_path, ckpt_name)
            ckpt_epoch = int(ckpt_name.split("=")[1].split("-")[0])
            ckpt_path_list.append((ckpt_epoch, ckpt_path))
      _, ckpt_path = max(ckpt_path_list, key=lambda x: x[0])
    else:
      # load checkpoint statestate_dict = torch.load(conf["ckpt"])
      system.load_state_dict(state_dict['state_dict'])

  logger = TensorBoardLogger(os.path.join(exp_dir,"tb_logs"), name=args["conf_id"])
  trainer = pl.Trainer(
    overfit_batches= 1 if conf["training"]["overfitting"] else 0,
    max_epochs=conf["training"]["max_epochs"],
    callbacks=callbacks,
    default_root_dir=exp_dir,
    accelerator=accelerator,
    strategy="ddp",
    devices="auto",
    limit_train_batches=conf["data"]["params"]["limit_train_batches"],
    limit_val_batches=conf["data"]["params"]["limit_val_batches"],
    limit_test_batches=conf["data"]["params"]["limit_test_batches"],
    num_sanity_val_steps=conf["training"]["num_sanity_val_steps"],
    gradient_clip_val=conf["training"]["gradient_clip_val"],
    deterministic=conf["training"]["seed"] is not None,
    #resume_from_checkpoint=conf["ckpt"],
    logger=logger,
  )
  if conf["ckpt"] is not None and conf["ckpt"] == "last":
    trainer.fit(system, ckpt_path=ckpt_path)
  else:
    trainer.fit(system)

  
  trainer.test(ckpt_path="best")

  best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
  with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
    json.dump(best_k, f, indent=0)

  state_dict = torch.load(checkpoint.best_model_path)
  system.load_state_dict(state_dict=state_dict["state_dict"])
  system.cpu()

  to_save = system.model.serialize()
  to_save.update(train_set.get_infos())
  torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
  args = parser.parse_args()
  args = vars(args)
  args['conf_id'] = f"{int(args['conf_id']):03d}"
  if args["backbone"] == "sudo":
    conf = merge_dicts(sudo_param, sudo_conf[args["conf_id"]])
  else:
    conf = merge_dicts(dprnn_param, dprnn_conf[args["conf_id"]]) 
  conf = {**conf, **args}
  pprint(conf)
  main(conf)
