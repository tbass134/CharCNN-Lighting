import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataset import CustomDatasetFromCSV
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning.metrics import Accuracy



class CharCNN(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.metric = Accuracy(num_classes=2)
        self.learning_rate = learning_rate
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
        acc = self.metric(y_hat, y)
        
        return {
            "loss":loss,
            "acc":acc,
            "log":{
                "train_loss":loss,
                "train_acc":acc
            }
        }
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        acc = self.metric(y_hat, y)

        return {
            "loss":loss,
            "acc":acc,
            "log":{
                "test_loss":loss,
                "test_acc":acc
            }
        }

    def training_epoch_end(self, outputs):
        train_acc_mean = torch.stack([x['acc'] for x in outputs]).mean()
        print("training_epoch_end",train_acc_mean)
        return {'train_acc': train_acc_mean}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        acc = self.metric(y_hat, y)
        
        return {
            "loss":loss,
            "acc":acc,
            "log":{
                "val_loss":loss,
                "val_acc":acc
            }
        }

    def validation_epoch_end(self, outputs):
        val_acc_mean = torch.stack([x['acc'] for x in outputs]).mean()
        print("validation_epoch_end: acc",val_acc_mean)
        return {'val_acc': val_acc_mean}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)

    def _configure_dataloaders(self,train=True):
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
        if train:
            return train_loader
        else:
            return validation_loader

    def train_dataloader(self):
        return self._configure_dataloaders(train=True)

    def val_dataloader(self):
        return self._configure_dataloaders(train=False)