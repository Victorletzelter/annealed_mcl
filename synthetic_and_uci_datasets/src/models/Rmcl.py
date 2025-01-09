import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.losses import WTALoss
from .methodsLighting import methodsLighting

class Rmcl(methodsLighting):
    def __init__(
        self,
        hparams,
        num_hypothesis,
        restrict_to_square,
        hidden_layers,
        input_dim,
        output_dim
    ):
        """Constructor for the multi-hypothesis networks with score heads.
        Args:
            hparams (dict): Dictionary of hyperparameters, contains:
                name (str): Name of the model.
                training_wta_mode (str): Mode of the WTA loss.
                training_epsilon (float): Epsilon parameter for the WTA loss if the mode is "wta-relaxed".
                training_conf_weight (float): Scoring weight parameter for the Score part of theWTA loss.
                training_distance (str): Underlying distance type for the WTA loss.
                optimizer (str): Optimizer type (e.g., Adam).
                learning_rate (float): Learning rate.
                denormalize_predictions (bool): Whether to denormalize the predictions.
                compute_risk (bool): Whether to compute the risk.
                compute_mse (bool): Whether to compute the MSE.
                plot_mode (bool): Whether to perform a plot at the end of the training.
                plot_mode_training (bool): Whether to perform a plot at each training epoch.
                plot_training_frequency (int): Frequency of the plot training.
                temperature_ini (float): Initial temperature.
                scheduler_mode (str): Scheduler mode.
                temperature_decay (float): Temperature decay if the scheduler mode is "exponential".
                temperature_lim (float): Temperature limit, after which the scheduler is stopped.
                wta_after_temperature_lim (bool): Whether to apply WTA after temperature limit.
                annealed_epsilon (bool): Whether to annealed epsilon (if "wta-relaxed" mode is used).
                epsilon_ini (float): Initial epsilon.
                epsilon_decay (float): Epsilon decay.
                compute_risk_val (bool): Whether to compute the risk for the validation set.
            num_hypothesis (int): Number of output hypotheses.
            restrict_to_square (bool): Whether to restrict the output to the square [-1,1]^2.
            hidden_layers (list): List of hidden layers.
            input_dim (int): Dimension of the input space.
            output_dim (int): Dimension of the output space.
        """
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        methodsLighting.__init__(self, hparams)

        self._hparams = hparams
        self.num_hypothesis = num_hypothesis
        self.restrict_to_square = restrict_to_square

        # Initialize ModuleList for layers and ModuleList for activation functions
        self.layers = nn.ModuleList()
        self.activation_functions = nn.ModuleList()  
        self.final_hyp_layers = nn.ModuleDict()
        self.final_conf_layers = nn.ModuleDict()
            
        # Construct the architecture
        self.construct_layers(hidden_layers)

        # Construct the final layers
        self.construct_final_layers(hidden_layers[-1])

        # check which device is available
        if torch.cuda.is_available():
            self._hparams['device'] = torch.device("cuda")
        else:
            self._hparams['device'] = torch.device("cpu")

        self.mask_initialized_heads = torch.ones(self.num_hypothesis, dtype=torch.bool, device=self._hparams['device']) 

        if "training_wta_mode" in self.hparams and "awta" in self.hparams.training_wta_mode :
            self.temperature = self.hparams["temperature_ini"]
        else :
            self.temperature = None 

    def construct_layers(self, hidden_layers):
        """
        Constructs the sequence of layers based on the input and hidden layers configuration.
        """
        self.layers.append(nn.Linear(self.input_dim, hidden_layers[0]))
        self.activation_functions.append(nn.ReLU())

        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.activation_functions.append(nn.ReLU())

    def construct_final_layers(self, last_hidden_size):
        """
        Constructs the hypothesis and confidence layers based on the number of hypotheses.
        """
        for k in range(self.num_hypothesis):
            self.final_hyp_layers[f"hyp_{k}"] = nn.Linear(last_hidden_size, self.output_dim)
            self.final_conf_layers[f"hyp_{k}"] = nn.Linear(last_hidden_size, 1)

    def forward(self, x):
        """Forward pass of the multi-hypothesis network with score heads.

        Returns:
            hyp_stacked (torch.Tensor): Stacked hypotheses. Shape [batch,self.num_hypothesis,output_dim]
            confs (torch.Tensor): Confidence of each hypothesis. Shape [batch,self.num_hypothesis,1]
        """
        # Pass input through each layer
        for layer, activation in zip(self.layers, self.activation_functions):
            x = activation(layer(x))

        outputs_hyps = []
        confidences = []

        for k in range(self.num_hypothesis):
            if self.restrict_to_square:
                outputs_hyps.append(
                    F.tanh(self.final_hyp_layers[f"hyp_{k}"](x))
                )  # Size [batch,output_dim]
            else:
                outputs_hyps.append(
                    (self.final_hyp_layers[f"hyp_{k}"](x))
                )  # Size [batch,output_dim]
            confidences.append(
                F.sigmoid(self.final_conf_layers[f"hyp_{k}"](x))
            )  # Size [batch,1])

        hyp_stacked = torch.stack(
            outputs_hyps, dim=-2
        )  # Shape [batch,self.num_hypothesis,output_dim]
        assert hyp_stacked.shape == (x.shape[0], self.num_hypothesis, self.output_dim)
        conf_stacked = torch.stack(confidences, dim=-2)  # [batch,self.num_hypothesis,1]
        assert conf_stacked.shape == (x.shape[0], self.num_hypothesis, 1)

        return hyp_stacked, conf_stacked

    def loss(self):
        return WTALoss(
            mode=self._hparams["training_wta_mode"],
            epsilon=self._hparams["training_epsilon"],
            distance=self._hparams["training_distance"],
            conf_weight=self._hparams["training_conf_weight"],
            output_dim=self.output_dim,
            temperature=self.temperature,
        )
    
    def prepare_predictions_mse(self, predictions) :

        hyps = predictions[0] # shape [batch,self.num_hypothesis,output_dim]
        confs = predictions[1] / predictions[1].sum(dim=-2, keepdim=True) # shape [batch,self.num_hypothesis,1]

        # Return the ponderated mean of the hypotheses
        return (hyps * confs).sum(dim=-2) # shape [batchxoutput_dim]

    def denormalize_predictions(self, predictions, mean_scaler, std_scaler) : 
        # mean and std scalers are the ones used for the target variable
        # shape [batch,self.num_hypothesis,output_dim], [batch,self.num_hypothesis,1]
        hyps = predictions[0]
        confs = predictions[1]
        hyps = hyps * std_scaler + mean_scaler

        return (hyps, confs)

    def denormalize_targets(self ,targets, mean_scaler, std_scaler) :
        # targets[0] of shape [batch,Max_sources,output_dim]
        # targets[1] of shape [batch,Max_sources,1]
        target_position = targets[0]
        target_position = target_position * std_scaler + mean_scaler

        return target_position, targets[1]
