from typing import Any, Dict, Tuple
from lightning import LightningModule
import torch.optim as optim
import torch
import numpy as np
from src.utils.losses import mhloss, mhconfloss

from src.utils.eval_utils import (
    generate_plot_adapted
)
import os

from src.utils import (
    RankedLogger)

log = RankedLogger(__name__, rank_zero_only=True)

class methodsLighting(LightningModule):
    """A `LightningModule`, which implements 8 key methods:

    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.

    """

    def __init__(
        self,
        hparams,
    ) -> None:
        """Lightning module.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self._hparams = hparams

        if self.hparams['compute_mse'] is True :
            self.mse_accumulator = 0
            self.n_samples_mse = 0
            self.rmse_accumulator = 0

    def loss(self) -> torch.Tensor:
        """Compute the loss function.

        :param outputs: The model's predictions.
        :param targets: The target labels.
        :return: The loss value.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use
        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        optimizer = getattr(optim, self._hparams["optimizer"])(
            self.trainer.model.parameters(), lr=self._hparams["learning_rate"]
        )

        if "scheduler" in self.hparams and self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer}

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, data_target_position, data_source_activity_target = batch
        targets = (data_target_position, data_source_activity_target)

        # Forward pass
        outputs = self(x.float())#.reshape(-1, self.input_dim).float())

        # Compute the loss
        loss = self.loss()(predictions=outputs, targets=targets)

        return loss, outputs, targets

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        output = {"train_loss": loss}
        self.log_dict(output)

        return loss

    def on_train_epoch_start(self) -> None:
        "Lightning hook that is called when a training epoch starts."

        if "training_wta_mode" in self.hparams and "awta" in self.hparams.training_wta_mode :
            # set the temperature
            self.temperature = self.scheduler_temperature(self.current_epoch)

            if "temperature_lim" in self.hparams :
                temperature_lim = self.hparams["temperature_lim"]
            else:
                temperature_lim = 1e-10

            if self.temperature < temperature_lim :
                if 'wta_after_temperature_lim' in self.hparams and self.hparams['wta_after_temperature_lim'] is True :
                    self._hparams["training_wta_mode"] = 'wta'
                    self.hparams["training_wta_mode"] = 'wta'
                else :
                    self.temperature = temperature_lim

            # update the loss
            self.loss = mhconfloss(
            mode=self._hparams["training_wta_mode"],
            epsilon=self._hparams["training_epsilon"],
            output_dim=self.output_dim,
            temperature=self.temperature,
            distance=self._hparams["training_distance"]
        )
            self.log_dict({"temperature": self.temperature})

        elif "annealed_epsilon" in self.hparams and self.hparams["annealed_epsilon"] is True and "training_wta_mode" in self.hparams and "wta-relaxed" in self.hparams.training_wta_mode :
            if "epsilon_ini" in self.hparams and self.hparams["epsilon_ini"] is not None :
                self._hparams['training_epsilon'] = self.scheduler_epsilon(self.current_epoch)
                self.hparams['training_epsilon'] = self.scheduler_epsilon(self.current_epoch)
                self.log_dict({"epsilon": self.hparams['training_epsilon']})

        if 'plot_mode_training' in self.hparams and 'plot_training_frequency' in self.hparams : 
            if self.current_epoch%self.hparams['plot_training_frequency']==0 and self.hparams['plot_mode'] is True and (self.hparams['name'] == 'rmcl' or 'histogram' in self.hparams['name']) : 
                # we check if the folder with plots exists
                if not os.path.exists(self.trainer.default_root_dir+'/plots'):
                    os.makedirs(os.path.join(self.trainer.default_root_dir,'plots'))

                # check if temperature is a float 
                if self.temperature is not None and isinstance(self.temperature , float):
                    plot_title = self.hparams['name']+'temperature_'+str(np.round(self.temperature,10))+'_epoch_'+str(self.current_epoch)
                else :
                    plot_title = self.hparams['name']+'_epoch_'+str(self.current_epoch)

                generate_plot_adapted(self, dataset_ms = self.trainer.datamodule.dataset_ms_class, 
                dataset_ss = self.trainer.datamodule.dataset_ss_class,
                path_plot = os.path.join(self.trainer.default_root_dir, 'plots'),
                model_type='rMCL',
                list_x_values=[0.1,0.6,0.9],
                n_samples_gt_dist=3000,
                num_hypothesis=self.num_hypothesis,
                save_mode=True,
                device='cpu',
                plot_title=plot_title,
                plot_voronoi=False,
                plot_title_bool=True)
        
    def scheduler_temperature(self,epoch) :
        if self.hparams.scheduler_mode == 'constant' :
            return self.hparams.temperature_ini
        elif self.hparams.scheduler_mode == 'linear' :
            return self.hparams.temperature_ini - (self.hparams.temperature_ini) * epoch/self.trainer.max_epochs
        elif self.hparams.scheduler_mode == 'exponential' :
            return self.hparams.temperature_ini * self.hparams.temperature_decay**epoch
        
    def scheduler_epsilon(self,epoch) :
        if self.hparams.scheduler_mode == 'linear' :
                return self.hparams.epsilon_ini - (self.hparams.epsilon_ini) * epoch/self.trainer.max_epochs
        elif self.hparams.scheduler_mode == 'warmup_linear' :
            return self.hparams.epsilon_ini - (self.hparams.epsilon_ini) * 2* epoch/self.trainer.max_epochs
        elif self.hparams.scheduler_mode == 'warmup_constant_linear' :
            if epoch < 10 :
                return self.hparams.epsilon_ini
            else : 
                return self.hparams.epsilon_ini - (self.hparams.epsilon_ini) * epoch/self.trainer.max_epochs
        elif self.hparams.scheduler_mode == 'exponential' :
            return self.hparams.epsilon_ini * self.hparams.epsilon_decay**epoch

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        output = {"val_loss": loss}
        self.log_dict(output)

        if 'compute_risk_val' in self.hparams and self.hparams['compute_risk_val'] is True :
            risk = self.compute_risk(predictions=preds, targets=targets)
            self.log_dict({"val_risk": risk})

    def on_validation_epoch_start(self) -> None:
        "Lightning hook that is called when a validation epoch starts."
        if "training_wta_mode" in self.hparams and "awta" in self.hparams.training_wta_mode :
            self.loss = mhconfloss(
            mode='wta',
            epsilon=self._hparams["training_epsilon"],
            output_dim=self.output_dim,
            temperature=self.temperature,
            distance=self._hparams["training_distance"],
            conf_weight=0.0
        )
            
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # The loss should be reset to the original one in training_epoch_start in the case of awta.
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        if 'denormalize_predictions' in self.hparams and self.hparams['denormalize_predictions'] is True : 

            self.mean_train = self.trainer.datamodule.uci_dataset_train.scaler_y.mean_[0]
            self.std_train = np.sqrt(self.trainer.datamodule.uci_dataset_train.scaler_y.var_[0])

            ### Adjust the square size accordingly if needed for nll computation, here, it is large on purpose.
            self.square_size = np.abs(self.mean_train) + 8*self.std_train 

            preds = self.denormalize_predictions(preds, mean_scaler = self.mean_train, std_scaler = self.std_train)
            targets = self.denormalize_targets(targets, mean_scaler = self.mean_train, std_scaler = self.std_train) 
            # The test set was normalized using the train set statistics. We denormalize it.

        if self.hparams['compute_risk'] is True : 
            risk = self.compute_risk(predictions=preds, targets=targets)
        else : 
            risk = torch.tensor(float("nan"))

        if self.hparams['compute_mse'] is True : 
            predictions_mse = self.prepare_predictions_mse(predictions=preds)
            mse, rmse = self.compute_mse(predictions=predictions_mse, targets=targets)
            self.mse_accumulator += mse*batch[0].shape[0]
            self.rmse_accumulator += (rmse)*batch[0].shape[0]
            self.n_samples_mse += batch[0].shape[0]
        else : 
            mse = torch.tensor(float("nan"))

        output = {
            "test_loss": loss,
            "test_risk": risk,
            "test_mse": mse,
        }

        self.log_dict(output)

    def compute_mse(self, predictions, targets):
        # Compute the MSE
        # Assumes predictions of shape [batch, output_dim]
        # targets[0]: data_target_position of shape [batch, Max_sources, output_dim]
        # targets[1]: data_source_activity_target of shape [batch, Max_sources, 1]
        # Assumes one target is active here.
        targets = targets[0][:, 0, :] # shape [batch, output_dim]
        square_diff = (predictions - targets) ** 2
        return torch.mean(square_diff.sum(dim=-1)), torch.mean(square_diff.sum(dim=-1) ** 0.5)

    def compute_risk(self, predictions, targets):

        predictions = (predictions[0], None)
        # Compute the risk
        return mhloss(
            distance="euclidean-squared",
            output_dim=self.output_dim,
        )(predictions=predictions, targets=targets)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        if self.hparams['compute_mse'] is True : 
            # Compute the mean MSE over all test batches
            total_mse = self.mse_accumulator / self.n_samples_mse
            rmse = total_mse ** 0.5

            self.log_dict({"test_accumulated_rmse": rmse})

            instance_based_rmse = self.rmse_accumulator / self.n_samples_mse

            self.log_dict({"test_instance_based_rmse": instance_based_rmse})

        if 'plot_mode' in self.hparams :
            plot_mode = self.hparams['plot_mode']
        else : 
            plot_mode = True
        if 'dataset_name' in self.trainer.datamodule._hparams :
            plot_mode=False

        if plot_mode is True and (self.hparams['name'] == 'rmcl' or 'histogram' in self.hparams['name']) : 
            generate_plot_adapted(self, dataset_ms = self.trainer.datamodule.dataset_ms_class, 
            dataset_ss = self.trainer.datamodule.dataset_ss_class,
            path_plot = self.trainer.default_root_dir,
            model_type='rMCL',
            list_x_values=[0.01,0.6,0.9],
            n_samples_gt_dist=3000,
            num_hypothesis=self.num_hypothesis,
            save_mode=True,
            device='cpu',
            plot_title=self.hparams['name']+'_preds')

        if plot_mode is True and self.hparams['name'] == 'gauss_mix' : 
            generate_plot_adapted(self, dataset_ms = self.trainer.datamodule.dataset_ms_class, 
            dataset_ss = self.trainer.datamodule.dataset_ss_class,
            path_plot = self.trainer.default_root_dir,
            model_type='MDN',
            list_x_values=[0.1,0.6,0.9],
            n_samples_gt_dist=3000,
            num_hypothesis=self.num_hypothesis,
            log_var_pred=self.log_var_pred,
            save_mode=True,
            device='cpu',
            plot_title=self.hparams['name']+'_preds')