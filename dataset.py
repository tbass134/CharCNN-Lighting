from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import re
import torch
import config
import utils

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)

        self.text = self.data.loc[:, "text"]
        self.labels = self.data.loc[:, "target"]

    def preprocess(self):
        pass

    def __getitem__(self, index):
        text = self.text[index]
        text = utils.lower(text)
        text = utils.remove_hashtags(text)
        text = utils.remove_user_mentions(text)
        text = utils.remove_urls(text)

        data = np.array([config.identity_mat[config.vocabulary.index(i)] for i in list(text)[::-1] if i in config.vocabulary],
                        dtype=np.float32)
        if len(data) > config.max_length:
            data = data[:config.max_length]
        elif 0 < len(data) < config.max_length:
            data = np.concatenate(
                (data, np.zeros((config.max_length - len(data), config.number_of_characters), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros(
                (config.max_length, config.number_of_characters), dtype=np.float32)

        label = self.labels[index]
        data = torch.Tensor(data)

        return data, label
    
    def __len__(self):
        return len(self.data)

    