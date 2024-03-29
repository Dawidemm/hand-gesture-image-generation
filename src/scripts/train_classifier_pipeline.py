import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping

from src.models.classifier import HandsClassifier
from src.utils.lit_hands_datamodule import LightningHandsDatamodule


DATASET_DIR = 'dataset'
BATCH_SIZE = 16
MAX_EPOCHS = 10

def train_classifier(
        dataset_directory: str,
        batch_size: int,
        max_epochs: int
    ):

    datamodule = LightningHandsDatamodule(
        root_directory=dataset_directory,
        batch_size=batch_size
    )
    datamodule.setup(stage='train')

    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=3
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        accelerator='auto',
        callbacks=[early_stopping],
        logger=True
    )

    model = HandsClassifier()

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    train_classifier(
        dataset_directory=DATASET_DIR,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS
    )