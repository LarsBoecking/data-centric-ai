import numpy as np
from typing import Any, Dict
from src.utils import logger

class DataCentricStrategy:
    def apply(self, X, y):
        raise NotImplementedError

    @staticmethod
    def from_config(conf: Dict[str, Any]) -> "DataCentricStrategy":
        logger.info(f"Creating strategy from config: {conf}")
        strategy_registry = {
            ("label_flipping", "random"): RandomLabelFlipping,
            ("label_flipping", "systematic"): SystematicLabelFlipping,
            ("number_instances", "random"): NumberInstanceStrategy,
            ("length_reduction", "random"): LengthReductionStrategy,
            ("baseline", "none"): BaselineStrategy,
        }
        key = (conf["type"], conf.get("mode"))
        StrategyClass = strategy_registry.get(key)
        if StrategyClass is None:
            logger.error(f"Unknown strategy configuration: {key}")
            raise ValueError(f"Unknown strategy configuration: {key}")
        logger.info(f"Strategy {StrategyClass.__name__} created successfully")
        return StrategyClass(**conf.get("params", {}))


class RandomLabelFlipping(DataCentricStrategy):
    def __init__(self, flip_ratio: float):
        self.flip_ratio = flip_ratio
        logger.info(
            f"Initialized RandomLabelFlipping with flip_ratio: {self.flip_ratio}"
        )

    def apply(self, X, y):
        logger.info(f"Applying RandomLabelFlipping with flip_ratio: {self.flip_ratio}")
        y_flipped = y.copy()
        n_samples = len(y)
        n_flip = int(self.flip_ratio * n_samples)
        flip_indices = np.random.choice(n_samples, size=n_flip, replace=False)
        unique_labels = np.unique(y)

        for idx in flip_indices:
            y_flipped[idx] = np.random.choice(unique_labels[unique_labels != y[idx]])

        logger.info("RandomLabelFlipping applied successfully")
        return X, y_flipped


class SystematicLabelFlipping(DataCentricStrategy):
    def __init__(self, confusion_matrix: Dict[str, Dict[str, float]]):
        self.confusion_matrix = confusion_matrix
        logger.info(
            f"Initialized SystematicLabelFlipping with confusion_matrix: {self.confusion_matrix}"
        )

    def apply(self, X, y):
        logger.info("Applying SystematicLabelFlipping")
        y_flipped = y.copy()

        for idx, label in enumerate(y):
            label_str = str(label)
            if label_str in self.confusion_matrix:
                probs = self.confusion_matrix[label_str]
                target_classes = list(probs.keys())
                probabilities = list(probs.values())
                y_flipped[idx] = np.random.choice(target_classes, p=probabilities)

        logger.info("SystematicLabelFlipping applied successfully")
        return X, y_flipped

class NumberInstanceStrategy(DataCentricStrategy):
    def __init__(self, reduction_ratio: float):
        if not (0.0 < reduction_ratio <= 1.0):
            raise ValueError("reduction_ratio must be between 0 and 1 (exclusive).")
        self.reduction_ratio = reduction_ratio
        logger.info(
            f"Initialized NumberInstanceStrategy with reduction_ratio: {self.reduction_ratio}"
        )

    def apply(self, X, y):
        logger.info(f"Applying NumberInstanceStrategy with reduction_ratio: {self.reduction_ratio}")
        n_samples = len(X)
        n_reduced = int(self.reduction_ratio * n_samples)
        selected_indices = np.random.choice(n_samples, size=n_reduced, replace=False)

        X_reduced = X[selected_indices]
        y_reduced = y[selected_indices]

        logger.info("NumberInstanceStrategy applied successfully")
        return X_reduced, y_reduced
    
class LengthReductionStrategy(DataCentricStrategy):
    def __init__(self, reduction_fraction: float, take_from_end: bool = False):
        if not (0.0 < reduction_fraction <= 1.0):
            raise ValueError("reduction_fraction must be between 0 and 1 (exclusive).")
        self.reduction_fraction = reduction_fraction
        self.take_from_end = take_from_end
        logger.info(
            "Initialized LengthReductionStrategy with reduction_fraction: %s, take_from_end: %s",
            self.reduction_fraction,
            self.take_from_end,
        )

    def apply(self, X, y):
        logger.info(
            f"Applying LengthReductionStrategy with reduction_fraction: {self.reduction_fraction}"
        )
        # len(X) gives the number of samples which is not appropriate here.
        # We need to reduce the temporal dimension of each series based on its
        # original length (the last axis).
        series_length = X.shape[-1]
        reduced_length = int(series_length * self.reduction_fraction)
        # Support numpy arrays directly for efficiency
        if self.take_from_end:
            X_reduced = X[..., -reduced_length:]
        else:
            X_reduced = X[..., :reduced_length]
        
        logger.info("LengthReductionStrategy applied successfully")
        return X_reduced, y

class BaselineStrategy(DataCentricStrategy):
    def __init__(self):
        logger.info("Initialized BaselineStrategy (no data-centric adaptation)")

    def apply(self, X, y):
        logger.info("Applying BaselineStrategy (no changes)")
        return X, y
