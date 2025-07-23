from aeon.datasets import load_classification
from ..utils.logger import get_logger


class UCRDataset:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.logger = get_logger(__name__)
        self.logger.info(f"Initialized UCRDataset with name: {self.name}, path: {self.path}")

    def load(self):
        self.logger.info(f"Loading dataset: {self.name}")
        X_train, y_train, meta = load_classification(
            name=self.name, split="train", return_metadata=True, extract_path=self.path
        )
        X_test, y_test = load_classification(
            name=self.name, split="test", return_metadata=False, extract_path=self.path
        )
        self.logger.info(f"Dataset {self.name} loaded successfully")
        return X_train, y_train, X_test, y_test, meta
