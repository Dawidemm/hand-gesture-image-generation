import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.classifier import HandsClassifier
from src.utils.hands_dataset import HandsDataset


DATASET_DIR = 'dataset'

def train_classifier():

    transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = HandsDataset(
        root_dir=DATASET_DIR,
        transform=transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True
    )

    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator='auto', 
        logger=False
    )

    model = HandsClassifier(input_size=(3, 256, 256))

    trainer.fit(model=model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    train_classifier()