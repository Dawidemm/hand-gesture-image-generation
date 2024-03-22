import torch
import lightning
import torch.nn as nn
from torch.nn.functional import softmax
from torchmetrics import Accuracy
from typing import Tuple


class HandsClassifier(lightning.LightningModule):
    def __init__(self, input_size: Tuple[int, ...], num_classes: int=3, learning_rate: float=0.1):
        super(HandsClassifier, self).__init__()

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=num_classes)
        
        feature_extractor_layers = [
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        fe_output_size = self._feature_extractor_output_size(input_size, feature_extractor_layers)
        classifier_layers = [
            nn.Linear(fe_output_size, 32),
            nn.LeakyReLU(inplace=True, negative_slope=0.05),
            nn.Linear(32, num_classes)
        ]

        self.feature_extractor = nn.Sequential(*feature_extractor_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    def _feature_extractor_output_size(self, input_size, layers):
        x = torch.rand(1, *input_size)
        for layer in layers:
            x = layer(x)

        return x.size(1) * x.size(2) * x.size(3)

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = softmax(x, dim=1)

        return x
    
    def _shared_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        # print(f'preds: {preds}')
        # print(f'preds argmax: {preds.argmax(dim=1)}')
        # print(f'y" {y}')

        step_loss = self.criterion(preds, y)
        step_acc = self.acc(preds.argmax(dim=1), y)

        return step_loss, step_acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     loss, acc = self._shared_step(batch, batch_idx)

    #     self.log('val_loss', loss, prog_bar=True)
    #     self.log('val_acc', acc, prog_bar=True)

    #     return loss
    
    # def test_step(self, batch, batch_idx):
    #     loss, acc = self._shared_step(batch, batch_idx)

    #     self.log('train_loss', loss)
    #     self.log('train_acc', acc)

    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer