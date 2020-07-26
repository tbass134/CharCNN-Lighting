from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import re
import torch
class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)

        self.text = self.data.loc[:, "text"]
        self.labels = self.data.loc[:, "target"]

        self.vocabulary =  "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        self.number_of_characters = 69
        self.max_length = 150
        self.identity_mat = np.identity(self.number_of_characters)


    def __getitem__(self, index):
        text = self.text[index]
        text = self.lower(text)
        text = self.remove_hashtags(text)
        text = self.remove_user_mentions(text)
        text = self.remove_urls(text)

        raw_text = text

        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text)[::-1] if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), self.number_of_characters), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros(
                (self.max_length, self.number_of_characters), dtype=np.float32)

        label = self.labels[index]
        data = torch.Tensor(data)

        return data, label
    
    def __len__(self):
        return len(self.data)

    def lower(self, text):
        return text.lower()

    def remove_hashtags(self,text):
        clean_text = re.sub(r'#[A-Za-z0-9_]+', "", text)
        return clean_text


    def remove_user_mentions(self, text):
        clean_text = re.sub(r'@[A-Za-z0-9_]+', "", text)
        return clean_text

    def remove_urls(self,text):
        clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        return clean_text
