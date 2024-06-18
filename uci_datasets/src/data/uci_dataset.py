import torch
import os
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd

class UCI_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, data_root, split, split_num, train_ratio, seed, normalize_x, normalize_y, y_dim):
        # global variables for reading data files
        _DATA_DIRECTORY_PATH = os.path.join(data_root, dataset_name, "data")
        _DATA_FILE = os.path.join(_DATA_DIRECTORY_PATH, "data.txt")
        _INDEX_FEATURES_FILE = os.path.join(_DATA_DIRECTORY_PATH, "index_features.txt")
        _INDEX_TARGET_FILE = os.path.join(_DATA_DIRECTORY_PATH, "index_target.txt")

        # set random seed 1 -- same setup as MC Dropout
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        # load the data
        data = np.loadtxt(_DATA_FILE)
        # load feature and target indices
        index_features = np.loadtxt(_INDEX_FEATURES_FILE)
        index_target = np.loadtxt(_INDEX_TARGET_FILE)
        # load feature and target as X and y
        X = data[:, [int(i) for i in index_features.tolist()]].astype(np.float32)
        y = data[:, int(index_target.tolist())].astype(np.float32)
        # preprocess feature set X

        # assert dataset_name not in ['bostonHousing', 'energy', 'naval-propulsion-plant']
        one_hot_encoding = False
        # one_hot_encoding = True if dataset_name in ['bostonHousing', 'energy', 'naval-propulsion-plant'] else False
        X, dim_cat = self.preprocess_uci_feature_set(X=X, dataset_name=dataset_name, one_hot_encoding=one_hot_encoding)
        self.dim_cat = dim_cat

        # load the indices of the train and test sets
        # split_num = 0 # 1 if dataset_name == 'YearPredictionMSD' else (5 if dataset_name == 'protein-tertiary-structure' else 20)
        index_train = np.loadtxt(self._get_index_train_test_path(_DATA_DIRECTORY_PATH, split_num, train=True))
        index_test = np.loadtxt(self._get_index_train_test_path(_DATA_DIRECTORY_PATH, split_num, train=False))

        # read in data files with indices
        x_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]].reshape(-1, 1)
        x_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]].reshape(-1, 1)

        # split train set further into train and validation set for hyperparameter tuning
        if split == "validation":
            num_training_examples = int(train_ratio * x_train.shape[0]) # config.diffusion.nonlinear_guidance.train_ratio
            x_test = x_train[num_training_examples:, :]
            y_test = y_train[num_training_examples:]
            x_train = x_train[0:num_training_examples, :]
            y_train = y_train[0:num_training_examples]

        self.x_train = x_train if type(x_train) is torch.Tensor else torch.from_numpy(x_train)
        self.y_train = y_train if type(y_train) is torch.Tensor else torch.from_numpy(y_train)
        self.x_test = x_test if type(x_test) is torch.Tensor else torch.from_numpy(x_test)
        self.y_test = y_test if type(y_test) is torch.Tensor else torch.from_numpy(y_test)

        self.train_n_samples = x_train.shape[0]
        self.train_dim_x = self.x_train.shape[1]  # dimension of training data input
        self.train_dim_y = self.y_train.shape[1]  # dimension of training regression output

        self.test_n_samples = x_test.shape[0]
        self.test_dim_x = self.x_test.shape[1]  # dimension of testing data input
        self.test_dim_y = self.y_test.shape[1]  # dimension of testing regression output

        assert self.train_dim_x == self.test_dim_x
        assert self.train_dim_y == self.test_dim_y

        self.normalize_x = normalize_x # config.data.normalize_x
        self.normalize_y = normalize_y # config.data.normalize_y
        self.scaler_x, self.scaler_y = None, None

        if self.normalize_x:
            self.normalize_train_test_x()
        if self.normalize_y:
            self.normalize_train_test_y()

        data = self.return_dataset(split=split)
        self.x = data[:, :-y_dim] if y_dim > 0 else data
        self.y = data[:, -y_dim:] if y_dim > 0 else torch.zeros(self.x.shape[0]).long()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        # return self.x[index], self.y[index]
        return self.x[index], self.y[index].unsqueeze(1), torch.ones_like(self.y[index].unsqueeze(1),dtype=torch.bool)

    def normalize_train_test_x(self):
        """
        When self.dim_cat > 0, we have one-hot encoded number of categorical variables,
            on which we don't conduct standardization. They are arranged as the last
            columns of the feature set.
        """
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        if self.dim_cat == 0:
            self.x_train = torch.from_numpy(
                self.scaler_x.fit_transform(self.x_train).astype(np.float32))
            self.x_test = torch.from_numpy(
                self.scaler_x.transform(self.x_test).astype(np.float32))
        else:  # self.dim_cat > 0
            x_train_num, x_train_cat = self.x_train[:, :-self.dim_cat], self.x_train[:, -self.dim_cat:]
            x_test_num, x_test_cat = self.x_test[:, :-self.dim_cat], self.x_test[:, -self.dim_cat:]
            x_train_num = torch.from_numpy(
                self.scaler_x.fit_transform(x_train_num).astype(np.float32))
            x_test_num = torch.from_numpy(
                self.scaler_x.transform(x_test_num).astype(np.float32))
            self.x_train = torch.from_numpy(np.concatenate([x_train_num, x_train_cat], axis=1))
            self.x_test = torch.from_numpy(np.concatenate([x_test_num, x_test_cat], axis=1))

    def normalize_train_test_y(self):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.y_train = torch.from_numpy(
            self.scaler_y.fit_transform(self.y_train).astype(np.float32))
        self.y_test = torch.from_numpy(
            self.scaler_y.transform(self.y_test).astype(np.float32))

    def return_dataset(self, split="train"):
        if split == "train":
            train_dataset = torch.cat((self.x_train, self.y_train), dim=1)
            return train_dataset
        else:
            test_dataset = torch.cat((self.x_test, self.y_test), dim=1)
            return test_dataset

    def summary_dataset(self, split="train"):
        if split == "train":
            return {'n_samples': self.train_n_samples, 'dim_x': self.train_dim_x, 'dim_y': self.train_dim_y}
        else:
            return {'n_samples': self.test_n_samples, 'dim_x': self.test_dim_x, 'dim_y': self.test_dim_y}
    
    def onehot_encode_cat_feature(self, X, cat_var_idx_list):
        """
        Apply one-hot encoding to the categorical variable(s) in the feature set,
            specified by the index list.
        """
        # select numerical features
        X_num = np.delete(arr=X, obj=cat_var_idx_list, axis=1)
        # select categorical features
        X_cat = X[:, cat_var_idx_list]
        X_onehot_cat = []
        for col in range(X_cat.shape[1]):
            X_onehot_cat.append(pd.get_dummies(X_cat[:, col], drop_first=True))
        X_onehot_cat = np.concatenate(X_onehot_cat, axis=1).astype(np.float32)
        dim_cat = X_onehot_cat.shape[1]  # number of categorical feature(s)
        X = np.concatenate([X_num, X_onehot_cat], axis=1)
        return X, dim_cat

    def preprocess_uci_feature_set(self, X, dataset_name, one_hot_encoding=True):
        """
        Obtain preprocessed UCI feature set X (one-hot encoding applied for categorical variable)
            and dimension of one-hot encoded categorical variables.
        """
        dim_cat = 0
        if one_hot_encoding:
            if dataset_name == 'bostonHousing':
                X, dim_cat = self.onehot_encode_cat_feature(X, [3])
            elif dataset_name == 'energy':
                X, dim_cat = self.onehot_encode_cat_feature(X, [4, 6, 7])
            elif dataset_name == 'naval-propulsion-plant':
                X, dim_cat = self.onehot_encode_cat_feature(X, [0, 1, 8, 11])
            else:
                pass
        return X, dim_cat

    def _get_index_train_test_path(self, data_directory_path, split_num, train=True):
        """
        Method to generate the path containing the training/test split for the given
        split number (generally from 1 to 20).
        @param split_num      Split number for which the data has to be generated
        @param train          Is true if the data is training data. Else false.
        @return path          Path of the file containing the requried data
        """
        if train:
            return os.path.join(data_directory_path, "index_train_" + str(split_num) + ".txt")
        else:
            return os.path.join(data_directory_path, "index_test_" + str(split_num) + ".txt")