from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class HandsDatset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.dataset = ImageFolder(root=root_dir)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform == None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256))
            ])
            image = transform(image)
        else:
            image = self.transform(image)

        return image, label