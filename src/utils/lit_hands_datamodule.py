from torch.utils.data import DataLoader
from torch.utils.data import random_split
from lightning import LightningDataModule
from torchvision import transforms

from src.utils.hands_dataset import HandsDataset

class LightningHandsDatamodule(LightningDataModule):
    def __init__(
            self, 
            root_directory: str, 
            batch_size: int,
            split: float = 0.2,
            transform=None
    ):
        
        super().__init__()

        self.root_directory = root_directory
        self.split = split
        self.batch_size = batch_size

        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def setup(self, stage: str):

        if stage == 'fit':
            dataset = HandsDataset(root_dir=self.root_directory, transform=self.transform)
            train_dataset_samples = int(len(dataset)* (1-self.split))
            val_dataset_samples = len(dataset) - train_dataset_samples
            self.train_dataset, self.val_dataset = random_split(dataset, [train_dataset_samples, val_dataset_samples])

        if stage == 'test':
            dataset = HandsDataset(root_dir=self.root_directory)
            self.test_dataset = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)