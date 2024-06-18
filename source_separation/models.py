from asteroid.models.base_models import BaseEncoderMaskerDecoder
from asteroid.utils.torch_utils import pad_x_to_y, script_if_tracing
from asteroid_filterbanks import make_enc_dec
from asteroid.masknn.convolutional import SuDORMRF
from asteroid.masknn import DPRNN

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def select_model(model_name,**model_kwargs):
  if model_name == "dprnn_sc":
    model = ScoredDPRNN(**model_kwargs)
  elif model_name == "sudo_sc":
    model = ScoredSuDORMRF(**model_kwargs)

  return model

class ScoredSuDORMRF(nn.Module):
  """
  SuDORMRF model from the asteroid toolkit with an additional scoring head to predict sources' activity.
  """
  def __init__(self,
              in_chan,
              n_src,
              fb_name="free",
              n_filters=512,
              kernel_size=21,
              stride=10,
              bn_chan=128,
              num_blocks=16,
              upsampling_depth=4,
              sample_rate=8000,
              mask_act="softmax",
              count_stage=None
              ) -> None:
    super().__init__()

    enc, dec = make_enc_dec(
      fb_name=fb_name,
      kernel_size=kernel_size,
      n_filters=n_filters,
      stride=kernel_size // 2,
      sample_rate=sample_rate,
      padding=kernel_size // 2,
      output_padding=(kernel_size // 2) - 1,
    )
    n_feats = enc.n_feats_out
    enc = _Padder(enc, upsampling_depth=upsampling_depth, kernel_size=kernel_size)
    
    enc_activation = nn.ReLU()

    masker = SuDORMRF(
      n_feats,
      n_src,
      bn_chan=bn_chan,
      num_blocks=num_blocks,
      upsampling_depth=upsampling_depth,
      mask_act=mask_act,
    )

    if count_stage == 'late' or count_stage == "late_enc":
      # input has shape (batch, n_src, n_feat, n_frame)
      scoring = nn.Sequential(
        nn.Linear(in_features=n_feats,out_features=bn_chan),
        GlobalAvgPooling(dim=2),
        nn.ReLU(),
        nn.Linear(in_features=bn_chan,out_features=1),
        nn.Sigmoid(),
      )
    elif count_stage == 'pool_late' or count_stage == "pool_late_enc":
      # input has shape (batch, n_src, n_feat, n_frame)
      scoring = nn.Sequential(
        GlobalAvgPooling(dim=2),
        nn.Linear(in_features=n_feats,out_features=bn_chan),
        nn.ReLU(),
        nn.Linear(in_features=bn_chan,out_features=1),
        nn.Sigmoid(),
      )
    elif count_stage == 'early':
      scoring = nn.Sequential(
        nn.Linear(in_features=n_feats,out_features=bn_chan),
        GlobalAvgPooling(dim=1),
        nn.ReLU(),
        nn.Linear(in_features=bn_chan,out_features=n_src),
        nn.Sigmoid(),
      )
    elif count_stage == 'pool_early':
      scoring = nn.Sequential(
         GlobalAvgPooling(dim=1),
        nn.Linear(in_features=n_feats,out_features=bn_chan),
        nn.ReLU(),
        nn.Linear(in_features=bn_chan,out_features=n_src),
        nn.Sigmoid(),
      )
    else:
      scoring = None

    self.enc=enc
    self.dec =dec
    self.masker=masker
    self.scoring=scoring
    self.enc_activation=enc_activation
    self.count_stage=count_stage
    self.n_src = n_src

  def forward(self,wav):
    if wav.ndim < 3:
      wav = wav.unsqueeze(1)
    tf_rep = self.enc_activation(self.enc(wav)) #forward_encoder(wav)
    est_masks = self.masker(tf_rep)
    masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
    decoded = self.dec(masked_tf_rep)

    if self.count_stage == "late" or self.count_stage == "pool_late":
      speaker_activity = self.scoring(est_masks.permute(0,1,3,2)) # linear input should have shape (batch, n_src, n_frame, n_feat)
      speaker_activity=speaker_activity.squeeze(-1)
    elif self.count_stage == "early" or self.count_stage == "pool_early":
      speaker_activity = self.scoring(tf_rep.permute(0,2,1))
    elif self.count_stage == "late_enc" or self.count_stage == "pool_late_enc":
      speaker_activity = self.scoring(masked_tf_rep.permute(0,1,3,2))
      speaker_activity=speaker_activity.squeeze(-1)

    else:
      speaker_activity = torch.zeros((wav.shape[0], self.n_src), requires_grad=True).to(wav.device)

    reconstructed = pad_x_to_y(decoded, wav)
    return reconstructed, speaker_activity




class ScoredDPRNN(nn.Module):
  """DPRNN separatoin model with an additional scoring head to predict source activity.

  References
    - [1] "Dual-path RNN: efficient long sequence modeling for
      time-domain single-channel speech separation", Yi Luo, Zhuo Chen
      and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
  """

  def __init__(
      self,
      n_src,
      out_chan=None,
      bn_chan=128,
      hid_size=128,
      chunk_size=100,
      hop_size=None,
      n_repeats=6,
      norm_type="gLN",
      mask_act="sigmoid",
      bidirectional=True,
      rnn_type="LSTM",
      num_layers=1,
      dropout=0,
      in_chan=None,
      fb_name="free",
      kernel_size=16,
      n_filters=64,
      stride=8,
      encoder_activation=None,
      sample_rate=8000,
      use_mulcat=False,
      is_multilabel=True,
      count_stage='late'
      #**fb_kwargs,
   ):
    super().__init__()
    self.encoder, self.decoder = make_enc_dec(
        fb_name,
        kernel_size=kernel_size,
        n_filters=n_filters,
        stride=stride,
        sample_rate=sample_rate,
        #**fb_kwargs,
    )
    n_feats = self.encoder.n_feats_out
    if in_chan is not None:
        assert in_chan == n_feats, (
          "Number of filterbank output channels"
          " and number of input channels should "
          "be the same. Received "
          f"{n_feats} and {in_chan}"
        )
    # Update in_chan
    self.masker = DPRNN(
        n_feats,
        n_src,
        out_chan=out_chan,
        bn_chan=bn_chan,
        hid_size=hid_size,
        chunk_size=chunk_size,
        hop_size=hop_size,
        n_repeats=n_repeats,
        norm_type=norm_type,
        mask_act=mask_act,
        bidirectional=bidirectional,
        rnn_type=rnn_type,
        num_layers=num_layers,
        dropout=dropout,
        use_mulcat=use_mulcat,
    )
    self.is_multilabel=is_multilabel
    if is_multilabel:
        count_act = nn.Sigmoid()
    else:
        count_act = nn.Softmax(-1)

    if count_stage == 'late' or count_stage == "late_enc":
      # input has shape (batch, n_src, n_feat, n_frame)
      scoring = nn.Sequential(
        nn.Linear(in_features=n_feats,out_features=bn_chan),
        GlobalAvgPooling(dim=2),
        nn.ReLU(),
        nn.Linear(in_features=bn_chan,out_features=1),
        nn.Sigmoid(),
      )
    elif count_stage == 'pool_late' or count_stage == "pool_late_enc":
      # input has shape (batch, n_src, n_feat, n_frame)
      scoring = nn.Sequential(
        GlobalAvgPooling(dim=2),
        nn.Linear(in_features=n_feats,out_features=bn_chan),
        nn.ReLU(),
        nn.Linear(in_features=bn_chan,out_features=1),
        nn.Sigmoid(),
      )
    elif count_stage == 'early':
      scoring = nn.Sequential(
        nn.Linear(in_features=n_feats,out_features=bn_chan),
        GlobalAvgPooling(dim=1),
        nn.ReLU(),
        nn.Linear(in_features=bn_chan,out_features=n_src),
        nn.Sigmoid(),
      )
    elif count_stage == 'pool_early':
      scoring = nn.Sequential(
         GlobalAvgPooling(dim=1),
        nn.Linear(in_features=n_feats,out_features=bn_chan),
        nn.ReLU(),
        nn.Linear(in_features=bn_chan,out_features=n_src),
        nn.Sigmoid(),
      )
    elif count_stage == "crnn_late":
      scoring = CrnnWrapper(dims=4,n_feat=n_feats,n_src=n_src,bn_chan=64,hidden_dim=128)
    elif count_stage == "crnn_early":
      scoring = CrnnWrapper(dims=3,n_feat=n_feats,n_src=n_src,bn_chan=64,hidden_dim=128)
    else:
      scoring = None
    
    self.enc_activation = nn.ReLU()
    self.count_stage = count_stage
    self.n_src=n_src
    self.scoring=scoring

  def forward(self,wav):
    """Enc/Mask/Dec model forward

    Args:
        wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

    Returns:
        torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
    """
    # Real forward
    if wav.ndim < 3:
      wav = wav.unsqueeze(1)
    tf_rep = self.enc_activation(self.encoder(wav)) #forward_encoder(wav)
    est_masks = self.masker(tf_rep)
    masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
    decoded = self.decoder(masked_tf_rep)

    if self.count_stage == "late" or self.count_stage == "pool_late":
      speaker_activity = self.scoring(est_masks.permute(0,1,3,2)) # linear input should have shape (batch, n_src, n_frame, n_feat)
      speaker_activity=speaker_activity.squeeze(-1)
    elif self.count_stage == "early" or self.count_stage == "pool_early":
      speaker_activity = self.scoring(tf_rep.permute(0,2,1))
    elif self.count_stage == "late_enc" or self.count_stage == "pool_late_enc":
      speaker_activity = self.scoring(masked_tf_rep.permute(0,1,3,2))
      speaker_activity=speaker_activity.squeeze(-1)
    elif self.count_stage == "crnn_early":
      speaker_activity = self.scoring(tf_rep)
    elif self.count_stage == "crnn_late":
      speaker_activity = self.scoring(est_masks)


    reconstructed = pad_x_to_y(decoded, wav)
    return reconstructed, speaker_activity
  

class _Padder(nn.Module):
  def __init__(self, encoder, upsampling_depth=4, kernel_size=21):
      super().__init__()
      self.encoder = encoder
      self.upsampling_depth = upsampling_depth
      self.kernel_size = kernel_size

      # Appropriate padding is needed for arbitrary lengths
      self.lcm = abs(self.kernel_size // 2 * 2**self.upsampling_depth) // math.gcd(
          self.kernel_size // 2, 2**self.upsampling_depth
      )

      # For serialize
      self.filterbank = self.encoder.filterbank
      self.sample_rate = getattr(self.encoder.filterbank, "sample_rate", None)

  def forward(self, x):
      x = self.pad(x, self.lcm)
      return self.encoder(x)

  @staticmethod    
  def pad(x, lcm: int):
    values_to_pad = int(x.shape[-1]) % lcm
    if values_to_pad:
        appropriate_shape = x.shape
        padding = torch.zeros(
            list(appropriate_shape[:-1]) + [lcm - values_to_pad], dtype=x.dtype, device=x.device
        )
        padded_x = torch.cat([x, padding], dim=-1)
        return padded_x
    return x

class GlobalAvgPooling(nn.Module):
    def __init__(self,dim=-1) -> None:
        super().__init__()
        self.dim=dim
    def forward(self,x):
        return torch.mean(x,dim=self.dim)

class CrnnWrapper(nn.Module):
  def __init__(self, dims, n_feat, n_src, bn_chan=64, hidden_dim=128) -> None:
    super().__init__()

    if dims == 3:
      self.model = CRNN1D(n_feat=n_feat,num_classes=n_src,bn_chan=bn_chan,hidden_dim=hidden_dim)
    elif dims == 4:
      self.model = CRNN2D(n_feat=n_feat,num_classes=n_src,bn_chan=bn_chan,hidden_dim=hidden_dim)
    else:
      raise NotImplementedError("No model for this")
    
  def forward(self, x):
    return F.sigmoid(self.model(x)) #multilabel classification 

class CRNN1D(nn.Module):
  def __init__(self, 
               n_feat, 
               num_classes,
               bn_chan=64,
               hidden_dim=128):
    super(CRNN1D, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=n_feat, out_channels=bn_chan, kernel_size=3)
    self.conv2 = nn.Conv1d(in_channels=bn_chan, out_channels=hidden_dim, kernel_size=3)
    self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=bn_chan, batch_first=True)
    self.fc = nn.Linear(bn_chan, num_classes)

  def forward(self, x):
    x = nn.functional.relu(self.conv1(x))
    x = nn.functional.relu(self.conv2(x))
    x = x.permute(0, 2, 1)  # Swap dimensions for LSTM
    x, _ = self.lstm(x)
    x = F.tanh(x)
    x = x[:, -1, :]  # Take the output of the last time step
    x = self.fc(x)
    return x

class CRNN2D(nn.Module):
  def __init__(self, 
               n_feat, 
               num_classes,
               hidden_dim=128,
               bn_chan=128,
               ):
    super(CRNN2D, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=n_feat*num_classes, out_channels=bn_chan, kernel_size=3)
    self.conv2 = nn.Conv1d(in_channels=bn_chan, out_channels=hidden_dim, kernel_size=3)
    self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=bn_chan, batch_first=True)
    self.fc = nn.Linear(bn_chan, num_classes)

  def forward(self, x):
    n_batch, n_src, n_feat, n_frame = x.size()
    x = x.view(n_batch,  n_src*n_feat, n_frame)  # Reshape to (B*N, F, T) for 1D convolution
    x = nn.functional.relu(self.conv1(x))
    x = nn.functional.relu(self.conv2(x))
    x = x.permute(0, 2, 1)  # Swap dimensions for LSTM
    x, _ = self.lstm(x)
    x = F.tanh(x)
    x = x[:, -1, :]  # Take the output of the last time step
    x = self.fc(x)
    return x

def get_default_config(model_name: str,
                       n_src:int ,
                       count_stage: str="late"):
  if model_name == "dprnn_sc":
    cfg = dict(
              n_src=n_src,
              out_chan=None,
              bn_chan=128,
              hid_size=128,
              chunk_size=100,
              hop_size=None,
              n_repeats=6,
              norm_type="gLN",
              mask_act="sigmoid",
              bidirectional=True,
              rnn_type="LSTM",
              num_layers=1,
              dropout=0,
              in_chan=None,
              fb_name="free",
              kernel_size=16,
              n_filters=64,
              stride=8,
              encoder_activation=None,
              sample_rate=8000,
              use_mulcat=False,
              is_multilabel=True,
              count_stage=count_stage,
          )
  elif model_name == "sudo_sc":
    cfg = {"fb_name": "free",
            "n_filters": 512,
            "kernel_size": 21,
            "stride": 10,
            "bn_chan": 128,
            "num_blocks": 16,
            "upsampling_depth": 4,
            "mask_act": "softmax",
            "n_src": n_src,
            "in_chan": 512,
            "count_stage": count_stage}
  
  return cfg
