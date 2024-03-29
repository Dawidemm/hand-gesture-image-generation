import pytest
import torch
from torchvision import transforms
from src.utils.hands_dataset import HandsDataset

@pytest.fixture
def dataset():
    return HandsDataset(root_dir='dataset', transform=None)

def test_hands_dataset_creation(dataset):
    assert dataset is not None

def test_hands_dataset_getitem(dataset):
    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image is not None
        assert label is not None

def test_hands_dataset_len(dataset):
    assert len(dataset) > 0

def test_hands_dataset_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])
    dataset = HandsDataset(root_dir='dataset', transform=transform)

    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.size() == (3, 64, 64)
        assert label is not None

def test_hands_dataset_image_size(dataset):
    size = (3, 256, 256)

    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.size() == size
        assert label is not None

def test_hands_dataset_image_type(dataset):
    for i in range(len(dataset)):
        image, _ = dataset[i]
        assert isinstance(image, torch.Tensor)