import torch
import lightning
import torch.nn as nn
from torchvision import models
from torchmetrics import Accuracy


class HandsClassifier(lightning.LightningModule):
    def __init__(self, num_classes: int=3, learning_rate: float=0.01, classifier_hidden_units: int=512):
        super(HandsClassifier, self).__init__()

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=num_classes)

        self.feature_extractor= models.resnet18(weights='IMAGENET1K_V1')
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        classifier_layers = [
            nn.Linear(1000, classifier_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(classifier_hidden_units, num_classes)
        ]
        self.classsifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classsifier(x)

        return x
    
    def _shared_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        step_loss = self.criterion(preds, y)
        step_acc = self.acc(preds.argmax(dim=1), y)

        return step_loss, step_acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)

        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer