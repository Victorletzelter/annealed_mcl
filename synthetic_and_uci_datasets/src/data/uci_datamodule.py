from lightning import LightningDataModule
from .uci_dataset import UCI_Dataset
from torch.utils.data import DataLoader

class UCI_datamodule(LightningDataModule):
    """
   Generator of datasets 
    """
    def __init__(
        self,
        batch_size,
        num_workers,
        hparams
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.hparams['batch_size'] = batch_size
        self.hparams['num_workers'] = num_workers

        for key in hparams:
            self.hparams[key] = hparams[key]

        self.uci_dataset_train = UCI_Dataset(dataset_name = self.hparams['dataset_name'], 
                                  data_root = self.hparams['data_root'], 
                                  split = 'train', 
                                  split_num = self.hparams['split_num'],
                                  train_ratio = self.hparams['train_ratio'], 
                                  seed = self.hparams['seed'], 
                                  normalize_x = self.hparams['normalize_x'], 
                                  normalize_y = self.hparams['normalize_y'], 
                                  y_dim = self.hparams['y_dim'])
        
        self.uci_dataset_val = UCI_Dataset(dataset_name = self.hparams['dataset_name'], 
                                  data_root = self.hparams['data_root'], 
                                  split = 'val', 
                                  split_num = self.hparams['split_num'],
                                  train_ratio = self.hparams['train_ratio'], 
                                  seed = self.hparams['seed'], 
                                  normalize_x = self.hparams['normalize_x'], 
                                  normalize_y = self.hparams['normalize_y'], 
                                  y_dim = self.hparams['y_dim'])
        
        self.uci_dataset_test = UCI_Dataset(dataset_name = self.hparams['dataset_name'], 
                                  data_root = self.hparams['data_root'], 
                                  split = 'test', 
                                  split_num = self.hparams['split_num'],
                                  train_ratio = self.hparams['train_ratio'], 
                                  seed = self.hparams['seed'], 
                                  normalize_x = self.hparams['normalize_x'], 
                                  normalize_y = self.hparams['normalize_y'], 
                                  y_dim = self.hparams['y_dim'])
    
    def __len__(self):
        return self.uci_dataset_train.__len__()

    def train_dataloader(self) :
        batch_size = min(self.hparams['batch_size'], len(self.uci_dataset_train))
        data_loader = DataLoader(self.uci_dataset_train, 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=self.hparams['num_workers'])

        return data_loader
        
    def val_dataloader(self) :
        batch_size = min(self.hparams['batch_size'], len(self.uci_dataset_val))
        data_loader = DataLoader(self.uci_dataset_val, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=self.hparams['num_workers'])
        
        return data_loader

    def test_dataloader(self) :
        batch_size = min(self.hparams['batch_size'], len(self.uci_dataset_test))
        data_loader = DataLoader(self.uci_dataset_test, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 num_workers=self.hparams['num_workers'])

        return data_loader
    
    def prepare_targets_emd(self, inputs, targets) :
        return targets

