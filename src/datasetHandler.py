from aeon.datasets import load_classification
from src.utils import logger

class UCRDataset:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        logger.info(f"Initialized UCRDataset with name: {self.name}, path: {self.path}")

    def load(self):
        logger.info(f"Loading dataset: {self.name}")
        X_train, y_train, meta = load_classification(
            name=self.name, split="train", return_metadata=True, extract_path=self.path
        )
        X_test, y_test = load_classification(
            name=self.name, split="test", return_metadata=False, extract_path=self.path
        )
        logger.info(f"Dataset {self.name} loaded successfully")
        return X_train, y_train, X_test, y_test, meta
