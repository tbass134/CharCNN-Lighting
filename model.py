import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import config
from pytorch_lightning.metrics import Accuracy

class CharCNN(pl.LightningModule):
    def __init__(self, train_ds=None, val_ds=None, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.metric = Accuracy()
        self.characters = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        
        self.loss_function =  torch.nn.CrossEntropyLoss()
        self.dropout_input = nn.Dropout2d(0.1)

        self.conv1 = nn.Sequential(nn.Conv1d(config.number_of_characters,
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

        input_shape = (128,config.max_length,config.number_of_characters)
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
            "train_acc":acc,
            "log":{
                "train_acc":acc
            }
        }
        
    def training_epoch_end(self, outputs):
        train_acc_mean = 0
        for output in outputs:
            train_acc_mean += output['train_acc']

        train_acc_mean /= len(outputs)

        # log training accuracy at the end of an epoch
        results = {
            'log': {'train_acc_mean': train_acc_mean.item()},
            'progress_bar': {'train_acc_mean': train_acc_mean},
        }
        return results

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        acc = self.metric(y_hat, y)

        return {
            "test_loss":loss,
            "test_acc":acc,
            "log":{
                "test_loss":loss,
                "test_acc":acc
            }
        }

    def test_epoch_end(self, outputs):
        test_acc_mean = 0
        for output in outputs:
            test_acc_mean += output['test_acc']

        test_acc_mean /= len(outputs)

        # log training accuracy at the end of an epoch
        results = {
            'log': {'test_acc_mean': test_acc_mean.item()},
            'progress_bar': {'test_acc_mean': test_acc_mean},
        }
        return results
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        acc = self.metric(y, y_hat)
        
        return {
            "val_loss":loss,
            "val_acc":acc,
            "log":{
                "val_loss":loss,
                "val_acc":acc
            }
        }

    def validation_epoch_end(self, outputs):
        val_acc_mean = 0
        for output in outputs:
            val_acc_mean += output['val_acc']

        val_acc_mean /= len(outputs)

        # log training accuracy at the end of an epoch
        results = {
            'log': {'val_acc_mean': val_acc_mean.item()},
            'progress_bar': {'val_acc_mean': val_acc_mean},
        }
        return results
   

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)        

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.val_ds