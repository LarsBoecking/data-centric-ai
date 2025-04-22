from tsml_eval.publications.y2023.tsc_bakeoff.set_bakeoff_classifier import (
    _set_bakeoff_classifier,
)
from src.utils import logger

### Classifier Wrapper ###
class BakeoffClassifier:
    def __init__(self, name: str, random_state: int = 0):
        self.name = name
        self.random_state = random_state
        logger.info(
            f"Initializing BakeoffClassifier with name: {self.name}, random_state: {self.random_state}"
        )
        self.model = _set_bakeoff_classifier(
            name, random_state=self.random_state, n_jobs=1
        )

    def fit(self, X, y):
        logger.info(f"Fitting classifier: {self.name}")
        self.model.fit(X, y)
        logger.info(f"Classifier {self.name} fitted successfully")

    def predict(self, X):
        logger.info(f"Making predictions with classifier: {self.name}")
        return self.model.predict(X)
