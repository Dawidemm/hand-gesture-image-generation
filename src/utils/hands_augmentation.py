import os
from torchvision import transforms
from torchvision.utils import save_image
from src.utils.hands_dataset import HandsDataset

class HandsAugmentation():
    def __init__(
            self,
            root_dir: str,
            labels: dict,
            multiply_scaler: int
    ):

        self.root_dir = root_dir
        self.dataset = HandsDataset(root_dir=self.root_dir)

        self.labels = labels
        self.multiply_scaler = multiply_scaler

    def gen_augmented_images(self):

        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomHorizontalFlip()
        ])

        for i in range(len(self.dataset)):
            for j in range(self.multiply_scaler):
            
                image, label = self.dataset[i]
                image_pil = transforms.ToPILImage()(image) 
                augmented_image = transform(image_pil)
                augmented_image_tensor = transforms.ToTensor()(augmented_image)
                
                label = self.labels[label]
                augmented_image_filepath = os.path.join(self.root_dir, f'{label}/aug-img{i}-v{j}.png')
                save_image(augmented_image_tensor, augmented_image_filepath)