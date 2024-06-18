import torch
import pandas as pd
from torch.utils import data
import os
import numpy as np
import soundfile as sf
import random as random

class Wsj0mixDatasetVarSpk(data.Dataset):
  """Dataset class for the wsj0-mix source separation dataset.

  Args:
    json_dir (str): The path to the directory containing the json files.
    sample_rate (int, optional): The sampling rate of the wav files.
    segment (float, optional): Length of the segments used for training,
      in seconds. If None, use full utterances (e.g. for test).
    n_src (int, optional): Number of sources in the training targets.

  References
    "Deep clustering: Discriminative embeddings for segmentation and
    separation", Hershey et al. 2015.
  """

  dataset_name = "wsj0-mix"

  def __init__(self, csv_dir, min_n_src=2, max_n_src=5, sample_rate=8000, segment=4.0, return_active_sources=True, one_hot_src_encoding=False, random_crop=True):
    super().__init__()
    # Task setting
    self.csv_dir = csv_dir
    self.sample_rate = sample_rate
    self.n_src = max_n_src
    # If true, returns the number of active source in the current segment (useful for loss calculation)
    self.return_active_sources = return_active_sources
    # If the number of sources is returned, it can be an int or a one-hot vector with source activity e.g. n_src=5 and 3 active sources -> [1,1,1,0,0]
    self.one_hot_src_encoding = one_hot_src_encoding
    # Load json files
    data_csv = os.path.join(csv_dir, f"{min_n_src}_{max_n_src}.csv")
    self.df = pd.read_csv(data_csv)
    if segment is not None:
      max_len = len(self.df)
      self.seg_len = int(segment * self.sample_rate)
      # Ignore the file shorter than the desired_length
      self.df = self.df[self.df["length"] >= self.seg_len]
      print(
        f"Drop {max_len - len(self.df)} utterances from {max_len} "
        f"(shorter than {segment} seconds)"
      )
    else:
      self.seg_len = None
    self.like_test = self.seg_len is None
    self.random_crop = random_crop

  def __len__(self):
    return len(self.df) 

  def __getitem__(self, idx):
    # Get the row in dataframe²
    row = self.df.iloc[idx]
    current_n_src = row["n_src"]
    # Get mixture path
    mixture_path = row["mix"]
    tot_samples = int(row["length"])
    self.mixture_path = mixture_path
    sources_list = []
    # If there is a seg start point is set randomly
    if self.seg_len is not None:
      start = random.randint(0, row["length"] - self.seg_len) if self.random_crop else 0
      stop = start + self.seg_len
    else:
      start = 0
      stop = row["length"]
    # If task is enh_both then the source is the clean mixture
    # Read sources
    for i in range(self.n_src):
      source_path = row[f"s{i + 1}"]
      #if len(str(source_path)) > 3:
      if i < current_n_src:
        source_path = row[f"s{i + 1}"]
        s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
      else:
        s = np.zeros(stop-start, dtype="float32") #+ 1e-10
      sources_list.append(s)
    # Read the mixture
    mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
    # Convert to torch tensor
    mixture = torch.from_numpy(mixture)
    # Stack sources
    sources = np.vstack(sources_list)
    # Convert sources to tensor
    sources = torch.from_numpy(sources)
    if not self.return_active_sources:
      return mixture, sources
    else:
      if self.one_hot_src_encoding:
        tmp=current_n_src
        current_n_src = torch.zeros((self.n_src,))
        current_n_src[:tmp] += 1
      return mixture, sources, current_n_src
      
    # Adapt to more than 2 sources
    #list_ids = mixture_path.split("/")[-1].split(".")[0].split("_")
    #return mixture, sources, list_ids

wsj0_license = dict(
    title="CSR-I (WSJ0) Complete",
    title_link="https://catalog.ldc.upenn.edu/LDC93S6A",
    author="LDC",
    author_link="https://www.ldc.upenn.edu/",
    license="LDC User Agreement for Non-Members",
    license_link="https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf",
    non_commercial=True,
)