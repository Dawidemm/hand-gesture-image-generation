import pytest
from src.scripts.train_classifier_pipeline import train_classifier

def test_train_classifier():
    try:
        train_classifier(
            dataset_directory='dataset',
            batch_size=16,
            max_epochs=2
        )
        assert True
    except Exception as e:
        pytest.fail(f"Classifier training failed with error: {str(e)}")