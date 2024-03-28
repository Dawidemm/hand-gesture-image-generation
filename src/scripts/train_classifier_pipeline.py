import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping

from src.models.classifier import HandsClassifier
from src.utils.lit_hands_datamodule import LightningHandsDatamodule


DATASET_DIR = 'dataset'

def train_classifier():

    datamodule = LightningHandsDatamodule(
        root_directory=DATASET_DIR,
        batch_size=16
    )
    datamodule.setup(stage='train')

    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=3
    )

    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator='auto',
        callbacks=[early_stopping],
        logger=True
    )

    model = HandsClassifier()

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    train_classifier()