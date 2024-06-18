import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import yaml
from asteroid.utils import flatten_dict
import pandas as pd

class MCL_System(pl.LightningModule):
 
  default_monitor: str = "val_loss"

  def __init__(
      self,
      model,
      optimizer,
      loss_func,
      train_loader,
      score_alpha=0.1,
      val_loader=None,
      test_loader=None,
      scheduler=None,
      config=None,
      batch_example=None,
      noise_schedule=None,
      metric_dict=None,
  ):

    super().__init__()
    self.model = model
    self.optimizer = optimizer
    self.loss_func = loss_func
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.scheduler = scheduler
    self.batch_example = batch_example # Store a batch for visualisation
    self.config = config
    self.conf_id = config["conf_id"]
    self.sample_rate = config["data"]["params"]["sample_rate"]
    self.score_alpha=score_alpha
    self.noise_schedule = (lambda x, epoch : x) if noise_schedule is None else noise_schedule
    self.metric_dict = metric_dict if metric_dict is not None else {} 

  def common_step(self, batch, batch_idx, train=True, test=False, dataloader_idx=0):
        mixture, target_list, _ = batch
        
        # add noise to data
        if train and self.noise_schedule is not None:
          target = self.noise_schedule(target_list, self.current_epoch)

        # compute prediction        
        prediction_list, score_list = self.model(mixture) 

        # compute loss
        prediction_loss, score_loss = self.loss_func(prediction_list, score_list, target_list, self.current_epoch)

        # log everything
        log_prefix = "train" if train else ("validation" if not test else "test")
        if not train:
            metric_dict = {}
            for metric_name, metric in self.metric_dict.items():
                self.dataloader_idx = dataloader_idx
                res = metric(prediction_list, score_list, target_list)
                res = res[0].mean() if isinstance(res, tuple) else res # if training loss used as metric, take only the prediction_loss part
                self.log(f"{log_prefix}/{metric_name}", res)
                metric_dict[f"{metric_name}"] = res
            if test:
              self.scores_list[f"dataloader_{dataloader_idx}"].append(metric_dict)

        return prediction_loss, score_loss

  def forward(self, *args, **kwargs):
    return self.model(*args, **kwargs)
  
  def training_step(self, batch, batch_nb):
    prediction_loss, score_loss = self.common_step(batch, batch_nb, train=True)
    loss = prediction_loss + self.score_alpha * score_loss
    self.log("train_loss", loss, logger=True,prog_bar=True, on_step=True, sync_dist=False)
    self.log("train/prediction_loss", prediction_loss, logger=True, sync_dist=False)
    self.log("train/score_loss", score_loss, prog_bar=True, logger=True,sync_dist=False)
    cur_lr = self.optimizer.param_groups[0]['lr']
    cur_wd = self.optimizer.param_groups[0]["weight_decay"]
    self.log("lr", cur_lr, prog_bar=True, on_step=True, logger=True, sync_dist=False)
    self.log("wd", cur_wd, prog_bar=True, on_step=True, logger=True, sync_dist=False)
    self.logger.experiment.add_scalars("losses", {"train": loss}, global_step=self.global_step)
    return loss

  def validation_step(self, batch, batch_nb):
    prediction_loss, score_loss = self.common_step(batch, batch_nb, train=False)
    loss = prediction_loss + self.score_alpha * score_loss
    self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True,sync_dist=False)
    self.log("val/prediction_loss", prediction_loss, logger=True,sync_dist=False)
    self.log("val/score_loss", score_loss, prog_bar=True, logger=True,sync_dist=False)
    self.logger.experiment.add_scalars("losses", {"val": loss}, global_step=self.global_step)

  def on_validation_epoch_end(self):
    """Method called at the end of each validation epoch.

    This method adds audio and figures to Tensorboard for
    visualizing the model performance during validation.

    """
    # Set the example
    inputs, targets, _ = self.batch_example
    outputs = self(inputs.unsqueeze(0).unsqueeze(0).to(self.device))[0]
    outputs = outputs / torch.max(torch.abs(outputs))
    
    if outputs.shape[0] < outputs.shape[1] and "Scored" in self.model.__class__.__name__:
       outputs = outputs.permute(1,0,2)
    
    if outputs.ndim == 3:
       outputs = outputs.squeeze(0)

    self.logger.experiment.add_audio("target_0",
                                     targets[0].squeeze().cpu().squeeze(),
                                     global_step=0,
                                     sample_rate=self.sample_rate)
    self.logger.experiment.add_audio("target_1",
                                     targets[1].squeeze().cpu().squeeze(),
                                     global_step=0,
                                     sample_rate=self.sample_rate)
    self.logger.experiment.add_audio("input",
                                     inputs.cpu().squeeze(),
                                     global_step=0,
                                     sample_rate=self.sample_rate)
    self.logger.experiment.add_audio("output_0",
                                     outputs[0, :].squeeze(
                                     ).cpu().squeeze(),
                                     global_step=self.global_step,
                                     sample_rate=self.sample_rate)
    self.logger.experiment.add_audio("output_1",
                                     outputs[1, :].squeeze(
                                     ).cpu().squeeze(),
                                     global_step=self.global_step,
                                     sample_rate=self.sample_rate)

  def on_validation_end(self) -> None:
    """Logs hyperparameters to tensorboard"""
    # Log model params
    hparams = self.config
    self.logger.log_hyperparams(
        hparams, metrics=self.trainer.callback_metrics["val_loss"])

  def on_train_start(self) -> None:
    """Saves the config in a yml file """
    if os.path.exists(self.trainer.default_root_dir):
      conf_path = os.path.join(self.trainer.default_root_dir, "conf.yml")
      with open(conf_path, "w") as outfile: 
        yaml.safe_dump(self.config, outfile)
      return super().on_train_start()

  def on_test_start(self) -> None:
    self.scores_list = {f"dataloader_{i}": [] for i in range(len(self.test_loader))}
    #self. = []

  def on_test_epoch_end(self) -> None:
    for k in self.scores_list.keys():
      df = pd.DataFrame(self.scores_list[k])
      df.to_csv(os.path.join(self.logger.log_dir,f"all_metrics_{k}.csv"),index=False)


  def configure_optimizers(self):
    """Initialize optimizers, batch-wise and epoch-wise schedulers."""

    if self.scheduler is None:
      return self.optimizer

    if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
      interval = "step"
    else:
      interval = "epoch"

    return {
        "optimizer": self.optimizer,
        "lr_scheduler": {
            "scheduler": self.scheduler,
            "monitor": "val_loss",
            "interval": interval
        },
    }

  def train_dataloader(self):
    """Training dataloader"""
    return self.train_loader

  def val_dataloader(self):
    """Validation dataloader"""
    return self.val_loader

  def on_save_checkpoint(self, checkpoint):
    """Overwrite if you want to save more things in the checkpoint."""
    checkpoint["training_config"] = self.config
    return checkpoint

  def test_step(self, batch, batch_nb, dataloader_idx=0):
    prediction_loss, score_loss = self.common_step(batch, batch_nb, train=False, test=True, dataloader_idx=dataloader_idx)
    loss = prediction_loss + self.score_alpha * score_loss
    self.log(f"test_loss", loss, on_epoch=True, prog_bar=True)
  
  def test_dataloader(self):
    return self.test_loader
        