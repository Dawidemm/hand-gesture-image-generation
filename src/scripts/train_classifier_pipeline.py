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

    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator='auto', 
        logger=False
    )

    model = HandsClassifier()

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    train_classifier()