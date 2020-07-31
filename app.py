import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import CustomDatasetFromCSV



import pandas as pd
import re
import numpy as np
from model import CharCNN


# import shutil
# try:
#     shutil.rmtree("lightning_logs")
# except:
#     pass

seed_everything(42)
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

train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=128, num_workers=4)
validation_loader = torch.utils.data.DataLoader(dataset,sampler=valid_sampler, num_workers=4)
       

model = CharCNN(train_ds=train_loader, val_ds=validation_loader)
trainer = Trainer(gpus=1,fast_dev_run=False)
trainer.fit(model)
        
