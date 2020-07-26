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
import re
import numpy as np
from model import CharCNN



seed_everything(42)
model = CharCNN()
trainer = Trainer(gpus=1, deterministic=True, max_epochs=5, fast_dev_run=False)
trainer.fit(model)
# trainer.test(test_dataloaders=validation_loader, ckpt_path=None)
        
