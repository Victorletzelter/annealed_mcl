import torch
import torch.nn as nn
import itertools
import einops 
import ot
import numpy as np
import inspect

from utils import compute_variance, compute_sample_diameter, compute_batch_diameter

def select_loss(loss_name,**loss_kwargs):
    
  framework, distance = loss_name.split("_")
  
  if distance == "mse":
    distance_metric = pairwise_mse
  elif distance == "sisdr":
    distance_metric = asteroid_pairwise_negative_sisdr
  elif distance == "snr":
    distance_metric = asteroid_pairwise_negative_snr
  else:
    raise NotImplementedError(f"No distance metric found for {distance=}, try: mse, sisdr, or snr")

  if framework == "pit":
    loss_kwargs = { k: v for k,v in loss_kwargs.items() if k in inspect.signature(Scored_PIT.__init__).parameters.keys() }
    loss_func = Scored_PIT(distance_metric=distance_metric, **loss_kwargs)
  elif framework == "mcl":
    loss_kwargs = { k: v for k,v in loss_kwargs.items() if k in inspect.signature(Scored_MCL.__init__).parameters.keys() }
    loss_func = Scored_MCL(distance_metric=distance_metric, **loss_kwargs)
  elif framework == "amcl":
    max_epochs = loss_kwargs["max_epochs"]
    temperature_unit_name = loss_kwargs["temperature_schedule"]["unit"] if "unit" in loss_kwargs["temperature_schedule"] else "variance"
    compute_temperature_unit = { "variance": compute_variance, "sample_diameter": compute_sample_diameter, "batch_diameter": compute_batch_diameter, }[temperature_unit_name]
    variance_kwargs = { k: v for k,v in loss_kwargs.items() if k in inspect.signature(compute_temperature_unit).parameters.keys() }
    schedule_kwargs = { k: v for k,v in loss_kwargs["temperature_schedule"].items() if k in inspect.signature(select_temperature_schedule).parameters.keys() }
    loss_kwargs = { k: v for k,v in loss_kwargs.items() if k in inspect.signature(Annealed_MCL.__init__).parameters.keys() and k != "temperature_schedule" }
    temperature_unit = compute_temperature_unit(distance_metric=distance_metric, **variance_kwargs)
    temperature_schedule = select_temperature_schedule(max_epochs=max_epochs, temperature_unit=temperature_unit, **schedule_kwargs)
    loss_func = Annealed_MCL(distance_metric=distance_metric, temperature_schedule=temperature_schedule, **loss_kwargs)
  elif framework == "emd":
    loss_kwargs = { k: v for k,v in loss_kwargs.items() if k in inspect.signature(EMD.__init__).parameters.keys() }
    loss_func = EMD(distance_metric=distance_metric, **loss_kwargs)
  else:
    raise NotImplementedError(f"No framework found for {framework=}. Try: emd, pit, amcl or mcl.")

  return loss_func

def select_metrics(metric_name_list, score_threshold=0.5):
  metric_dict = {}
  for metric_name in metric_name_list:
    distance = metric_name.split("_")[1]
    current_score_threshold = score_threshold if "inference" in metric_name else None

    if distance == "mse":
      distance_metric = pairwise_mse
    elif distance == "sisdr":
      distance_metric = asteroid_pairwise_negative_sisdr
    elif distance == "snr":
      distance_metric = asteroid_pairwise_negative_snr
    else:
      raise NotImplementedError(f"No distance metric found for {distance=}, try: mse, sisdr, or snr")

    if "pit" in metric_name:
      metric_dict[metric_name] = Scored_PIT(distance_metric=distance_metric, return_score_loss=False, score_threshold=current_score_threshold)
    if "emd" in metric_name:
      metric_dict[metric_name] = EMD(distance_metric=distance_metric, return_score_loss=False, differentiable=False, beta=10)
    if "mcl" in metric_name:
      metric_dict[metric_name] = Scored_MCL(distance_metric=distance_metric, return_score_loss=False, score_threshold=current_score_threshold)
    if "entropy" in metric_name:
      metric_dict[metric_name] = Assignment_Entropy(distance_metric=distance_metric)

  return metric_dict

def select_temperature_schedule(name="constant", temperature_unit=1, temperature_start=1, temperature_end=0, epoch_start=0, epoch_end=None, max_epochs=None):

  if name == "constant":
    temperature_schedule = lambda epoch: temperature_start * temperature_unit
  elif name == "linear":
    epoch_end = max_epochs if epoch_end is None else epoch_end
    temperature_schedule = lambda epoch : max(temperature_start * temperature_unit * (epoch_end - epoch) / (epoch_end - epoch_start), 0) if epoch > epoch_start else temperature_start * temperature_unit
  elif name == "exponential":
    epoch_end = max_epochs if epoch_end is None else epoch_end
    def temperature_schedule(epoch):
      if epoch < epoch_start:
        return temperature_start * temperature_unit
      if epoch < epoch_end:
        return  temperature_start * temperature_unit * (0.9 ** (epoch - epoch_start))
      return 0
  else:
    raise NotImplementedError(f"No temperature schedule found for {name=}. Try: constant or linear.")
  
  return temperature_schedule

def pairwise_mse(prediction_list, target_list):
  return torch.square(prediction_list.unsqueeze(2)-target_list.unsqueeze(1)).sum(dim=-1)


def asteroid_pairwise_negative_snr(prediction_list, target_list, epsilon=1e-9):
  # extract shape
  n_prediction, n_target = prediction_list.shape[1], target_list.shape[1]

  # normalize & reshape
  target_list = (target_list - target_list.mean(axis=-1, keepdim=True)).unsqueeze(dim=1) # [batch_size, 1, n_target, n_sample]
  prediction_list = (prediction_list - prediction_list.mean(axis=-1, keepdim=True)).unsqueeze(dim=2) # [batch_size, n_prediction, 1, n_sample]

  # compute pairwise snr
  target_norm_list = torch.sum(target_list**2, dim=-1, keepdim=True) + epsilon # [batch_size, 1, n_target, n_sample]
  pairwise_difference = prediction_list - target_list # [batch_size, n_prediction, n_target, n_sample]
  pairwise_snr = 10 * torch.log10(torch.sum(target_norm_list**2, dim=-1) / (torch.sum(pairwise_difference**2, dim=-1) + epsilon) + epsilon) # [batch_size, n_prediction, n_target]
  return -pairwise_snr


def asteroid_pairwise_negative_sisdr(prediction_list, target_list, epsilon=1e-9):
  # extract shape
  n_prediction, n_target = prediction_list.shape[1], target_list.shape[1]

  # normalize & reshape
  target_list = (target_list - target_list.mean(axis=-1, keepdim=True)).unsqueeze(dim=1) # [batch_size, 1, n_target, n_sample]
  prediction_list = (prediction_list - prediction_list.mean(axis=-1, keepdim=True)).unsqueeze(dim=2) # [batch_size, n_prediction, 1, n_sample]

  # compute pairwise sisdr
  target_norm_list = torch.sum(target_list**2, dim=-1, keepdim=True) + epsilon # [batch_size, 1, n_target, n_sample]
  pairwise_product = torch.sum(prediction_list * target_list, dim=-1, keepdim=True) # [batch_size, n_prediction, n_target, n_sample]
  pairwise_projection = (pairwise_product * target_list) / target_norm_list  # [batch_size, n_prediction, n_target, n_sample]
  pairwise_difference = prediction_list - pairwise_projection # [batch_size, n_prediction, n_target, n_sample]
  pairwise_sisdr = 10 * torch.log10(torch.sum(pairwise_projection**2, dim=-1) / (torch.sum(pairwise_difference**2, dim=-1) + epsilon) + epsilon) # [batch_size, n_prediction, n_target]
  return -pairwise_sisdr


def sudo_pairwise_negative_sisdr(prediction_list, target_list, epsilon=1e-9):
  # extract shape
  n_prediction, n_target = prediction_list.shape[1], target_list.shape[1]

  # normalize
  target_list = target_list - target_list.mean(dim=-1, keepdim=True)
  prediction_list = prediction_list - prediction_list.mean(dim=-1, keepdim=True)
  
  # compute pairwise sisdr
  source_norm_list = torch.norm(target_list, p=2, dim=-1) ** 2 + epsilon
  pairwise_product = torch.einsum('bid,bjd->bij', prediction_list, target_list) # [batch_size, n_prediction, n_target]
  pairwise_projection = (pairwise_product / source_norm_list.unsqueeze(1)).unsqueeze(-1) * target_list.unsqueeze(1) # [batch_size, n_prediction, n_target, n_sample]
  pairwise_difference = prediction_list.unsqueeze(2) - pairwise_projection # [batch_size, n_prediction, n_target, n_sample]
  pairwise_sisdr = 20 * torch.log10(torch.norm(pairwise_projection, p=2, dim=-1) / (torch.norm(pairwise_difference, p=2, dim=-1) + epsilon)+epsilon) # [batch_size, n_prediction, n_target]
  return -pairwise_sisdr


class Annealed_MCL(nn.Module):
  def __init__(self, distance_metric, temperature_schedule, return_score_loss=True, sample_normalization=True, min_temperature=1e-4):
    super(Annealed_MCL, self).__init__()
    self.distance_metric = distance_metric
    self.temperature_schedule = temperature_schedule
    self.score_metric = torch.nn.BCELoss()
    self.return_score_loss = return_score_loss
    self.sample_normalization = sample_normalization
    self.min_temperature = min_temperature
    
  def forward(self, prediction_list, score_list, target_list, epoch=None):
    # extract shape
    batch_size, n_prediction, n_target = prediction_list.shape[0], prediction_list.shape[1], target_list.shape[1]

    # compute pairwise distance
    pairwise_distance = self.distance_metric(prediction_list, target_list) # [batch_size, n_prediction, n_target]
    pairwise_distance = einops.rearrange(pairwise_distance, "b p t -> b t p") # [batch_size, n_target, n_prediction]

    # soft assignation of source to closest prediction (& hard assignment for scoring purposes)
    temperature = self.temperature_schedule(epoch)
    amcl_sisdr = (torch.nn.functional.softmin(pairwise_distance / temperature, dim=2).detach() * pairwise_distance).sum(dim=2) if temperature > self.min_temperature else pairwise_distance.min(dim=2)[0] # [batch_size, n_target]  
    target_assignment = torch.argmin(pairwise_distance, dim=2) # [batch_size, n_target]
    
    # mask inactive target (normalize per sample)
    target_list = (target_list - target_list.mean(axis=-1, keepdim=True)).unsqueeze(dim=2) # [batch_size, n_target, 1, n_sample]
    target_mask = (target_list.abs().sum(dim=-1) > 0.).squeeze(-1) # [batch_size, n_target]
    amcl_sisdr = ((amcl_sisdr * target_mask).sum(dim=-1) / target_mask.sum(dim=-1)).mean() if self.sample_normalization else amcl_sisdr[target_mask].mean() # []

    # compute prediction -> target assignment
    if self.return_score_loss:
      prediction_assignment = torch.stack([torch.nn.functional.one_hot(target_assignment[:,target_index], num_classes=n_prediction).float() for target_index in range(n_target)], dim=-1) # [batch_size, n_prediction, n_target]
      masked_prediction_assignment = prediction_assignment * target_mask.unsqueeze(1) # [batch_size, n_prediction, n_target]
      score_taget = masked_prediction_assignment.any(dim=-1).float() # [batch_size, n_prediction]
      score_loss = self.score_metric(score_list, score_taget.detach()) # []

    return (amcl_sisdr, score_loss) if self.return_score_loss else amcl_sisdr


class Scored_MCL(nn.Module):
  def __init__(self, distance_metric, return_score_loss=True, sample_normalization=True, score_threshold=None):
      super(Scored_MCL, self).__init__()
      self.distance_metric = distance_metric
      self.score_metric = torch.nn.BCELoss()
      self.return_score_loss = return_score_loss
      self.sample_normalization = sample_normalization
      self.score_threshold = score_threshold

  def forward(self, prediction_list, score_list, target_list, epoch=None):
      # extract shape
      batch_size, n_prediction, n_target = prediction_list.shape[0], prediction_list.shape[1], target_list.shape[1]
      
      # compute pairwise distance
      pairwise_distance = self.distance_metric(prediction_list, target_list) # [batch_size, n_prediction, n_target]
      pairwise_distance = einops.rearrange(pairwise_distance, "b p t -> b t p") # [batch_size, n_target, n_prediction]
      
      # mask inactive predictions
      prediction_mask = (score_list <= self.score_threshold) if self.score_threshold is not None else torch.zeros_like(score_list).bool() # [batch_size, n_prediction]
      pairwise_distance[prediction_mask.unsqueeze(1).repeat(1, n_target, 1)] = 0 # [batch_size, n_target, n_prediction]

      # assign source to closest prediction  
      mcl_sisdr, target_assignment = pairwise_distance.min(dim=2) # [batch_size, n_target],  [batch_size, n_target]

      # mask inactive target (normalize per sample)
      target_list = (target_list - target_list.mean(axis=-1, keepdim=True)).unsqueeze(dim=2) # [batch_size, n_target, 1, n_sample]
      target_mask = (target_list.abs().sum(dim=-1) > 0.).squeeze(-1) # [batch_size, n_target]
      mcl_sisdr = ((mcl_sisdr * target_mask).sum(dim=-1) / target_mask.sum(dim=-1)).mean() if self.sample_normalization else mcl_sisdr[target_mask].mean() # []

      # compute prediction -> target assignment
      if self.return_score_loss:
        prediction_assignment = torch.stack([torch.nn.functional.one_hot(target_assignment[:,target_index], num_classes=n_prediction).float() for target_index in range(n_target)], dim=-1) # [batch_size, n_prediction, n_target]
        masked_prediction_assignment = prediction_assignment * target_mask.unsqueeze(1) * ~prediction_mask.unsqueeze(2) # [batch_size, n_prediction, n_target]
        score_target = masked_prediction_assignment.any(dim=-1).float() # [batch_size, n_prediction]
        score_loss = self.score_metric(score_list, score_target.detach()) # []
        
      return (mcl_sisdr, score_loss) if self.return_score_loss else mcl_sisdr


class EMD(nn.Module):
  def __init__(self, distance_metric, return_score_loss=False, differentiable=False, beta=10):
      super(EMD, self).__init__()
      self.distance_metric = distance_metric
      self.score_metric = torch.nn.BCELoss()
      self.ot_metric = (lambda p, q, cost: ot.sinkhorn2(p, q, cost, beta)) if differentiable else (lambda p, q, cost: ot.emd2(p, q, cost))
      self.return_score_loss = return_score_loss

  def forward(self, prediction_list, score_list, target_list, epoch=None):
      # extract shape 
      n_prediction = prediction_list.shape[1]

      # compute pairwise distance
      pairwise_distance = self.distance_metric(prediction_list, target_list) # [batch_size, n_prediction, n_target]

      # compute emd 
      prediction_probability, target_probability = score_list / score_list.sum(dim=-1, keepdim=True), (target_list.abs().sum(dim=-1) > 0) / (target_list.abs().sum(dim=-1) > 0).sum(dim=-1, keepdim=True) # [batch_size, n_prediction], [batch_size, n_target]
      nonzero_mask = ((score_list.sum(dim=-1, keepdim=True) > 0) & ((target_list.abs().sum(dim=-1) > 0).sum(dim=-1, keepdim=True) > 0)) # [batch_size]
      emd = torch.mean(torch.stack([(self.ot_metric(p,q,cost) if nonzero else torch.tensor(0.)) for p, q, cost, nonzero in zip(prediction_probability, target_probability, pairwise_distance, nonzero_mask)])) # []

      # compute score loss
      score_loss = torch.zeros_like(emd)
      return (emd, score_loss) if self.return_score_loss else emd


class Scored_PIT(nn.Module):
  def __init__(self, distance_metric, score_threshold=None, return_score_loss=True):
      super(Scored_PIT, self).__init__()
      self.distance_metric = distance_metric
      self.ot_metric = EMD(distance_metric, differentiable=False, return_score_loss=True)
      self.score_metric = torch.nn.BCELoss()
      self.score_threshold = score_threshold
      self.return_score_loss = return_score_loss

  def forward(self, prediction_list, score_list, target_list, epoch=None):
      # extract shape
      n_prediction, n_target = prediction_list.shape[1], target_list.shape[1]

      # compute pairwise distance
      pairwise_distance = self.distance_metric(prediction_list, target_list) # [batch_size, n_prediction, n_target]
      
      # mask inactive predictions & target
      target_activity = (target_list.abs().sum(dim=-1) > 0).unsqueeze(1) # [batch_size, 1, n_target]
      pairwise_distance, target_count = pairwise_distance * target_activity, target_activity.sum(dim=(1,2)) # [batch_size, n_prediction, n_target], [batch_size]
      
      # use pit when there are more predictions than targets
      if self.score_threshold is None:
        # find best permutation
        pit_sisdr, permutation_index = torch.stack([torch.diagonal(pairwise_distance[:, permutation, :], dim1=1, dim2=2).sum(dim=-1) / target_count for permutation in itertools.permutations(range(n_prediction))]).min(dim=0) # [batch_size]

        # compute prediction -> target assignment
        if self.return_score_loss:
          permutation = torch.argsort(torch.LongTensor(list(itertools.permutations(range(n_prediction)))), dim=1)[permutation_index.to("cpu")] # [batch_size, n_prediction]
          prediction_assignment = torch.eye(n_prediction, n_target)[permutation].to(pit_sisdr.device) * target_activity # [batch_size, n_prediction, n_target]
          score_target = prediction_assignment.any(dim=-1).float() # [batch_size, n_prediction]
          score_loss = self.score_metric(score_list, score_target) # []
          
      # use optimal transport otherwise
      else:
        score_list, nonzero_mask = (score_list > self.score_threshold).float(), (score_list > self.score_threshold).float().sum(dim=-1) > 0 # [batch_size, n_prediction], [batch_size]
        zero = torch.zeros(1, requires_grad=prediction_list.requires_grad, device=prediction_list.device) # []
        pit_sisdr, score_loss = self.ot_metric(prediction_list[nonzero_mask], score_list[nonzero_mask], target_list[nonzero_mask]) if nonzero_mask.any() else (zero, zero) # [], []
        
      return (pit_sisdr.mean(), score_loss) if self.return_score_loss else pit_sisdr.mean()


class Assignment_Entropy(nn.Module):
  def __init__(self, distance_metric, epsilon=1e-9):
      super(Assignment_Entropy, self).__init__()
      self.distance_metric = distance_metric
      self.epsilon = epsilon

  def forward(self, prediction_list, score_list, target_list):
      # extract shape
      n_prediction, n_target = prediction_list.shape[1], target_list.shape[1]
      
      # compute pairwise sisdr & prediction assigment
      pairwise_distance = self.distance_metric(prediction_list, target_list) # [batch_size, n_prediction, n_target]
      target_assignment = pairwise_distance.argmin(dim=2) # [batch_size, n_prediction]
      assignment_matrix = torch.stack([torch.nn.functional.one_hot(target_assignment[:,target_index], num_classes=n_prediction).float() for target_index in range(n_target)], dim=-1) # [batch_size, n_prediction, n_target]
      assignment_distribution = assignment_matrix.mean(dim=2) # [batch_size, n_prediction]

      # compute entropy
      assignment_entropy = -(assignment_distribution * torch.log(assignment_distribution + self.epsilon)).sum(dim=-1).mean() # []
      return assignment_entropy

