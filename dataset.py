import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class TaggingDataSet(Dataset):
    def __init__(self, data, seq_len=50):
        columns_nodes = ["node " + str(i) for i in range(1, 17)]
        self.data = torch.tensor(data[columns_nodes].values, dtype=torch.float32)

        labels = data["label"].replace(["benign", "malicious"], [0, 1]).values
        self.labels = torch.tensor(labels, dtype=torch.float32)

        self.idc = np.arange(0, len(self.labels), seq_len)
        self.len = len(self.idc) - 1
        self.seq_len = seq_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start = self.idc[idx]
        stop = start + self.seq_len
        next_item = self.data[start:stop], self.labels[start:stop]
        return next_item


class TaggingDataModule(LightningDataModule):
    def __init__(self, data_train, batch_size=512, num_workers=0):
        super().__init__()

        self.data_train = data_train
        self._already_called = False
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if self._already_called:
            return

        # Standardize data
        columns_nodes = ["node " + str(i) for i in range(1, 17)]
        data_train = self.data_train[columns_nodes].values
        data_train_flattened = data_train.flatten()
        data_train_len = len(data_train_flattened)
        data_mean = data_train.sum() / data_train_len
        data_std_dev = np.sqrt(np.sum((data_train - data_mean) ** 2) / data_train_len)
        self.data_train[columns_nodes] = (data_train - data_mean) / data_std_dev

        self.data_train = TaggingDataSet(self.data_train)

        self._already_called = True

        return data_mean, data_std_dev

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
