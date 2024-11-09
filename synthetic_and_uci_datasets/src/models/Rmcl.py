import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.losses import mhloss, mhconfloss
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
        """Constructor for the multi-hypothesis network with confidence (rMCL).

        Args:
            num_hypothesis (int): Number of output hypotheses.
        """
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        methodsLighting.__init__(self, hparams)

        self.name = "rmcl"
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
        """For pass of the multi-hypothesis network with confidence (rMCL).

        Returns:
            hyp_stacked (torch.Tensor): Stacked hypotheses. Shape [batchxself.num_hypothesisxoutput_dim]
            confs (torch.Tensor): Confidence of each hypothesis. Shape [batchxself.num_hypothesisx1]
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
                )  # Size [batchxoutput_dim]
            else:
                outputs_hyps.append(
                    (self.final_hyp_layers[f"hyp_{k}"](x))
                )  # Size [batchxoutput_dim]
            confidences.append(
                F.sigmoid(self.final_conf_layers[f"hyp_{k}"](x))
            )  # Size [batchx1])

        hyp_stacked = torch.stack(
            outputs_hyps, dim=-2
        )  # Shape [batchxself.num_hypothesisxoutput_dim]
        assert hyp_stacked.shape == (x.shape[0], self.num_hypothesis, self.output_dim)
        conf_stacked = torch.stack(confidences, dim=-2)  # [batchxself.num_hypothesisx1]
        assert conf_stacked.shape == (x.shape[0], self.num_hypothesis, 1)

        return hyp_stacked, conf_stacked

    def wta_risk(self, test_loader, device):

        risk_value = torch.tensor(0.0, device=device)

        criterion = mhloss(mode="wta", distance="euclidean-squared")

        for _, data in enumerate(test_loader):
            # Move the input and target tensors to the device

            data_t = data[0].to(device)
            data_target_position = data[1].to(device)
            data_source_activity_target = data[2].to(device)

            # Forward pass
            outputs = self(data_t.float().reshape(-1, 1))

            # Compute the loss
            risk_value += criterion(
                outputs, (data_target_position, data_source_activity_target)
            )

        return risk_value / len(test_loader)

    def loss(self):
        return mhconfloss(
            mode=self._hparams["training_wta_mode"],
            epsilon=self._hparams["training_epsilon"],
            distance=self._hparams["training_distance"],
            conf_weight=self._hparams["training_conf_weight"],
            output_dim=self.output_dim,
            temperature=self.temperature,
        )
    
    def prepare_predictions_mse(self, predictions) :

        hyps = predictions[0] # shape [batchself.num_hypothesisxoutput_dim]
        confs = predictions[1] / predictions[1].sum(dim=-2, keepdim=True) # shape [batchself.num_hypothesisx1]

        # Return the ponderated mean of the hypotheses
        return (hyps * confs).sum(dim=-2) # shape [batchxoutput_dim]

    def denormalize_predictions(self, predictions, mean_scaler, std_scaler) : 
        # mean and std scalers are the ones used for the target variable
        # shape [batchself.num_hypothesisxoutput_dim], [batchxself.num_hypothesisx1]
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
