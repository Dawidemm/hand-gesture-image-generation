import pytest
from src.utils.lit_hands_datamodule import LightningHandsDatamodule

@pytest.fixture
def hands_data_module():
    root_directory = 'dataset'
    batch_size = 32
    return LightningHandsDatamodule(root_directory, batch_size)

def test_lightning_hands_data_module_initialization(hands_data_module):
    assert hands_data_module.root_directory == 'dataset'
    assert hands_data_module.batch_size == 32
    assert hands_data_module.split == 0.2
    assert hands_data_module.transform is not None

def test_lightning_hands_data_module_setup_fit(hands_data_module):
    hands_data_module.setup(stage='fit')
    assert len(hands_data_module.train_dataset) > 0
    assert len(hands_data_module.val_dataset) > 0

def test_lightning_hands_data_module_setup_test(hands_data_module):
    hands_data_module.setup(stage='train')
    assert len(hands_data_module.test_dataset) > 0

def test_lightning_hands_data_module_train_dataloader(hands_data_module):
    hands_data_module.setup(stage='fit')
    train_dataloader = hands_data_module.train_dataloader()
    assert train_dataloader.batch_size == 32
    assert len(train_dataloader.dataset) > 0

def test_lightning_hands_data_module_val_dataloader(hands_data_module):
    hands_data_module.setup(stage='fit')
    val_dataloader = hands_data_module.val_dataloader()
    assert val_dataloader.batch_size == 32
    assert len(val_dataloader.dataset) > 0

def test_lightning_hands_data_module_test_dataloader(hands_data_module):
    hands_data_module.setup(stage='train')
    test_dataloader = hands_data_module.test_dataloader()
    assert test_dataloader.batch_size == 32
    assert len(test_dataloader.dataset) > 0