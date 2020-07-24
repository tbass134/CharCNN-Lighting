import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
from torch.utils.data.dataset import Dataset
import re
import numpy as np

class CharCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.characters = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        
        self.loss_function =  torch.nn.CrossEntropyLoss()
        self.dropout_input = nn.Dropout2d(0.1)

        self.conv1 = nn.Sequential(nn.Conv1d(69 + 0,
                                             256,
                                             kernel_size=7,
                                             padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=0),
                                   nn.ReLU()
                                   )

        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=0),
                                   nn.ReLU()
                                   )

        self.conv5 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=0),
                                   nn.ReLU()
                                   )

        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3)
                                   )

        input_shape = (128,150,69)
        self.output_dimension = self._get_conv_output(input_shape)

        # define linear layers
 
        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Linear(1024, 2)

        # initialize weights

        self._create_weights()


    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    def forward(self, x):
        x = self.dropout_input(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        return dict(
            loss=loss,
            log=dict(
                train_loss=loss
            )
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

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

validation_split = .2
shuffle_dataset = True

dataset = CustomDatasetFromCSV("train.csv")

# Creating data indices for training and validation splits:
seed = 42
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.seed(seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset,sampler=valid_sampler)


seed_everything(seed)
train_loader = DataLoader(dataset, batch_size=128, num_workers=4)
model = CharCNN()
trainer = Trainer(deterministic=True, max_epochs=10)
trainer.fit(model, train_loader)
trainer.test()
        
