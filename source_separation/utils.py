import torch
from tqdm import tqdm


def merge_dicts(dict1, dict2):
  merged = dict1.copy()
  for key, value in dict2.items():
    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
      merged[key] = merge_dicts(merged[key], value)
    else:
      merged[key] = value
  return merged


def compute_variance(dataloader, distance_metric, device="cuda", limit_batches=1.):
    variance_sum = 0
    limit_batches = len(dataloader) * limit_batches if type(limit_batches) is float else limit_batches
    for batch_index, (source, target_list, _) in tqdm(enumerate(dataloader), total=limit_batches):
        if batch_index >= limit_batches:
            break
        target_list = target_list.to(device) # [batch_size, n_target, n_sample]
        barycenter = target_list.mean(dim=1).unsqueeze(1) # [batch_size, 1, n_sample]
        pairwise_distance = distance_metric(barycenter, target_list) # [batch_size, 1, n_target]
        intra_cluster_variance = pairwise_distance.abs().mean(dim=-1).squeeze(1).mean() # []
        variance_sum += intra_cluster_variance
    return variance_sum / len(dataloader)


def compute_sample_diameter(dataloader, distance_metric, device="cuda", limit_batches=1.):
    sample_diameter_sum = 0
    limit_batches = len(dataloader) * limit_batches if type(limit_batches) is float else limit_batches
    for batch_index, (source, target_list, _) in tqdm(enumerate(dataloader), total=limit_batches):
        if batch_index >= limit_batches:
            break
        target_list = target_list.to(device) # [batch_size, n_target, n_sample]
        pairwise_distance = distance_metric(target_list, target_list) # [batch_size, n_target, n_target]
        sample_diameter = pairwise_distance.max(dim=-1)[0].max(dim=-1)[0].mean() # []
        sample_diameter_sum += sample_diameter
    return sample_diameter_sum / len(dataloader)


def compute_batch_diameter(dataloader, distance_metric, device="cuda", limit_batches=1.):
    batch_diameter_sum = 0
    limit_batches = len(dataloader) * limit_batches if type(limit_batches) is float else limit_batches
    for batch_index, (source, target_list, _) in tqdm(enumerate(dataloader), total=limit_batches):
        if batch_index >= limit_batches:
            break
        target_list = target_list.to(device) # [batch_size, n_target, n_sample]
        pairwise_distance = distance_metric(target_list, target_list) # [batch_size, n_target, n_target]
        batch_diameter = pairwise_distance.max() # []
        batch_diameter_sum += batch_diameter
    return batch_diameter_sum / len(dataloader)
