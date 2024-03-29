from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

class HandsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.dataset = ImageFolder(root=root_dir)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform == None:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = transform(image)
        else:
            image = self.transform(image)

        return image, label