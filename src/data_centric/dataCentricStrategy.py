import numpy as np
from typing import Any, Dict, Tuple
from ..utils.logger import get_logger


class DataCentricStrategy:
    """
    Base class for data-centric adaptation strategies in time series classification.

    This abstract base class defines the interface for implementing various data-centric
    strategies that modify training data to improve model performance. All concrete
    strategy implementations must inherit from this class and implement the apply method.

    The class provides a factory method to create strategy instances from configuration
    dictionaries, enabling flexible experiment setup through YAML configuration files.

    Methods:
        apply(X, y) -> Tuple[np.ndarray, np.ndarray]:
            Abstract method that applies the data-centric strategy to input data.
            Must be implemented by all subclasses.

        from_config(conf: Dict[str, Any]) -> DataCentricStrategy:
            Static factory method that creates strategy instances from configuration dictionaries.

    Raises:
        NotImplementedError: If a subclass doesn't implement the apply method.

    Example:
        ```python
        # Create strategy from configuration
        config = {
            "type": "label_flipping",
            "mode": "random",
            "params": {"flip_ratio": 0.1}
        }
        strategy = DataCentricStrategy.from_config(config)
        X_modified, y_modified = strategy.apply(X_train, y_train)
        ```
    """

    def __init__(self):
        """Initialize the base DataCentricStrategy with logging."""
        self.logger = get_logger(__name__)
        self.logger.info("Initialized DataCentricStrategy base class")
        if not hasattr(self, "apply"):
            raise NotImplementedError("Subclasses must implement the apply method")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the data-centric strategy to the input data.

        Args:
            X (np.ndarray): Input time series data of shape (n_samples, n_features) or (n_samples, n_timesteps).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Modified data (X_modified, y_modified).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def from_config(conf: Dict[str, Any]) -> "DataCentricStrategy":
        """
        Factory method to create strategy instances from configuration dictionaries.

        This method uses a registry pattern to map configuration keys to strategy classes,
        enabling flexible instantiation of different strategies based on type and mode.

        Args:
            conf (Dict[str, Any]): Configuration dictionary with keys:
                - type (str): Strategy type ('label_flipping', 'number_instances', 'length_reduction', 'baseline')
                - mode (str): Strategy mode ('random', 'systematic', 'none')
                - params (Dict, optional): Strategy-specific parameters

        Returns:
            DataCentricStrategy: Instantiated strategy object.

        Raises:
            ValueError: If the strategy configuration is unknown or invalid.

        Example:
            ```python
            config = {
                "type": "label_flipping",
                "mode": "random",
                "params": {"flip_ratio": 0.15}
            }
            strategy = DataCentricStrategy.from_config(config)
            ```
        """
        logger = get_logger(__name__)
        logger.info(f"Creating strategy from config: {conf}")

        strategy_registry = {
            ("label_flipping", "random"): RandomLabelFlipping,
            ("number_instances", "random"): NumberInstanceStrategy,
            ("length_reduction", "random"): LengthReductionStrategy,
            ("feature_quality", "normal"): FeatureQualityStrategy,
            ("baseline", "none"): BaselineStrategy,
        }

        key = (conf["type"], conf.get("mode"))
        StrategyClass = strategy_registry.get(key)

        if StrategyClass is None:
            logger.error(f"Unknown strategy configuration: {key}")
            raise ValueError(f"Unknown strategy configuration: {key}")

        logger.info(f"Strategy {StrategyClass.__name__} created successfully")
        return StrategyClass(**conf.get("params", {}))


class BaselineStrategy(DataCentricStrategy):
    """
    Baseline strategy that applies no data-centric modifications.

    This strategy serves as a control condition in experiments, returning the
    input data unchanged. It's useful for establishing baseline performance
    against which other data-centric strategies can be compared.

    Methods:
        apply(X, y) -> Tuple[np.ndarray, np.ndarray]:
            Returns input data unchanged.

    Example:
        ```python
        strategy = BaselineStrategy()
        X_unchanged, y_unchanged = strategy.apply(X_train, y_train)
        # Data remains exactly the same
        ```
    """

    def __init__(self):
        """Initialize BaselineStrategy with no parameters."""
        super().__init__()
        self.logger.info("Initialized BaselineStrategy (no data-centric adaptation)")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply baseline strategy (no modifications).

        Args:
            X (np.ndarray): Input time series data.
            y (np.ndarray): Target labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Unchanged input data.
        """
        self.logger.info("Applying BaselineStrategy (no changes)")
        return X, y


class RandomLabelFlipping(DataCentricStrategy):
    """
    Random label flipping strategy for introducing noise in training labels.

    This strategy randomly flips a specified fraction of training labels to different
    classes, simulating noisy labeling scenarios commonly encountered in real-world
    datasets. The flipping is performed uniformly at random among all available classes.

    Args:
        flip_ratio (float): Fraction of labels to flip (between 0.0 and 1.0).

    Methods:
        apply(X, y) -> Tuple[np.ndarray, np.ndarray]:
            Applies random label flipping to the training data.

    Example:
        ```python
        strategy = RandomLabelFlipping(flip_ratio=0.1)
        X_modified, y_noisy = strategy.apply(X_train, y_train)
        # 10% of labels will be randomly flipped to other classes
        ```
    """

    def __init__(self, flip_ratio: float):
        """
        Initialize RandomLabelFlipping strategy.

        Args:
            flip_ratio (float): Fraction of labels to flip (between 0.0 and 1.0).
        """
        super().__init__()
        self.flip_ratio = flip_ratio
        self.logger.info(
            f"Initialized RandomLabelFlipping with flip_ratio: {self.flip_ratio}"
        )

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random label flipping to the training data.

        Randomly selects a fraction of samples and flips their labels to different
        classes chosen uniformly at random from the available classes (excluding
        the original class).

        Args:
            X (np.ndarray): Input time series data (unchanged).
            y (np.ndarray): Target labels to be modified.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Original X data and modified labels.
        """
        self.logger.info(
            f"Applying RandomLabelFlipping with flip_ratio: {self.flip_ratio}"
        )

        y_flipped = y.copy()
        n_samples = len(y)
        n_flip = int(self.flip_ratio * n_samples)
        flip_indices = np.random.choice(n_samples, size=n_flip, replace=False)
        unique_labels = np.unique(y)

        for idx in flip_indices:
            # Choose a different label than the current one
            available_labels = unique_labels[unique_labels != y[idx]]
            y_flipped[idx] = np.random.choice(available_labels)

        self.logger.info(
            f"RandomLabelFlipping applied successfully: {n_flip}/{n_samples} labels flipped"
        )
        return X, y_flipped



class NumberInstanceStrategy(DataCentricStrategy):
    """
    Instance reduction strategy for reducing the number of training samples.

    This strategy randomly selects a subset of training instances, effectively
    reducing the training set size. This can be used to study the impact of
    training data quantity on model performance or to simulate scenarios with
    limited labeled data.

    Args:
        reduction_ratio (float): Fraction of instances to remove (between 0.0 and 1.0).

    Methods:
        apply(X, y) -> Tuple[np.ndarray, np.ndarray]:
            Randomly selects a subset of training instances.

    Raises:
        ValueError: If reduction_ratio is not between 0 and 1 (exclusive).

    Example:
        ```python
        strategy = NumberInstanceStrategy(reduction_ratio=0.3)
        X_reduced, y_reduced = strategy.apply(X_train, y_train)
        # Keep only 70% of original training instances
        ```
    """

    def __init__(self, reduction_ratio: float):
        """
        Initialize NumberInstanceStrategy.

        Args:
            reduction_ratio (float): Fraction of instances to remove (0 <= ratio < 1).

        Raises:
            ValueError: If reduction_ratio is not in valid range.
        """
        if not (0.0 <= reduction_ratio < 1.0):
            raise ValueError("reduction_ratio must be between 0 (inclusive) and 1 (exclusive).")

        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.logger.info(
            f"Initialized NumberInstanceStrategy with reduction_ratio: {self.reduction_ratio}"
        )

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply instance reduction by randomly selecting a subset of samples.

        Args:
            X (np.ndarray): Input time series data.
            y (np.ndarray): Target labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Reduced dataset with fewer instances.
        """
        self.logger.info(
            f"Applying NumberInstanceStrategy with reduction_ratio: {self.reduction_ratio}"
        )

        n_samples = len(X)
        n_reduced = int((1-self.reduction_ratio) * n_samples)
        selected_indices = np.random.choice(n_samples, size=n_reduced, replace=False)

        X_reduced = X[selected_indices]
        y_reduced = y[selected_indices]

        self.logger.info(
            f"NumberInstanceStrategy applied successfully: {n_reduced}/{n_samples} instances kept"
        )
        return X_reduced, y_reduced


class LengthReductionStrategy(DataCentricStrategy):
    """
    Length reduction strategy for truncating time series to shorter sequences.

    This strategy reduces the temporal length of time series by keeping only a
    fraction of the original sequence length during training, then pads the reduced
    sequences back to original length with zeros to maintain consistent shapes.
    This simulates scenarios with partial observations or shorter measurement windows.

    Args:
        reduction_fraction (float): Fraction of original length to remove (between 0.0 and 1.0, exclusive).
        take_from_end (bool, optional): If True, keep the end portion of the series;
            if False, keep the beginning. Defaults to False.

    Methods:
        apply(X, y) -> Tuple[np.ndarray, np.ndarray]:
            Truncates time series to the specified fraction and pads back to original length.

    Raises:
        ValueError: If reduction_fraction is not between 0 and 1 (exclusive).

    Example:
        ```python
        strategy = LengthReductionStrategy(reduction_fraction=0.5, take_from_end=True)
        X_shortened, y_unchanged = strategy.apply(X_train, y_train)
        # Keep only the last 50% of each time series, pad the rest with zeros
        # Output shape remains the same as input shape
        ```
    """

    def __init__(self, reduction_fraction: float, take_from_end: bool = True):
        """
        Initialize LengthReductionStrategy.

        Args:
            reduction_fraction (float): Fraction of original length to remove (0 < fraction <= 1).
            take_from_end (bool): Whether to keep the end portion (True) or beginning (False).

        Raises:
            ValueError: If reduction_fraction is not in valid range.
        """
        if not (0.0 <= reduction_fraction < 1.0):
            raise ValueError("reduction_fraction must be between 0 (inclusive) and 1 (exclusive).")

        super().__init__()
        self.reduction_fraction = reduction_fraction
        self.take_from_end = take_from_end
        self.logger.info(
            f"Initialized LengthReductionStrategy with reduction_fraction: {self.reduction_fraction}, "
            f"take_from_end: {self.take_from_end}"
        )

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply length reduction by truncating time series sequences and padding back to original length.

        Args:
            X (np.ndarray): Input time series data of shape (..., sequence_length).
            y (np.ndarray): Target labels (unchanged).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Modified X data with same shape as input, original labels.
        """
        self.logger.info(
            f"Applying LengthReductionStrategy with reduction_fraction: {self.reduction_fraction}"
        )

        original_shape = X.shape
        series_length = original_shape[-1]
        reduced_length = int(series_length * (1-self.reduction_fraction))

        # Create output array filled with zeros, same shape as input
        X_reduced = np.zeros_like(X)

        if self.take_from_end:
            # Keep the end portion, pad at the beginning with zeros
            X_reduced[..., -reduced_length:] = X[..., -reduced_length:]
        else:
            # Keep the beginning portion, pad at the end with zeros
            X_reduced[..., :reduced_length] = X[..., :reduced_length]

        self.logger.info(
            f"LengthReductionStrategy applied successfully: {reduced_length}/{series_length} timesteps kept, "
            f"remaining positions filled with zeros. Output shape: {X_reduced.shape}"
        )
        return X_reduced, y


class FeatureQualityStrategy(DataCentricStrategy):
    """
    Feature quality degradation strategy for adding noise to time series data.

    This strategy simulates noisy sensor measurements or data collection issues by
    adding Gaussian noise to the time series features. The noise is sampled from
    a normal distribution with mean determined by the mode and standard deviation
    controlled by the noise_level parameter.

    Args:
        noise_level (float): Standard deviation of the Gaussian noise to add.
        mode (str, optional): Mode determining the noise distribution.

    Methods:
        apply(X, y) -> Tuple[np.ndarray, np.ndarray]:
            Adds Gaussian noise to the time series features.

    Raises:
        ValueError: If mode is not one of the supported options.

    Example:
        ```python
        strategy = FeatureQualityStrategy(noise_level=0.1, mode="normal")
        X_noisy, y_unchanged = strategy.apply(X_train, y_train)
        # Adds zero-mean Gaussian noise with std=0.1 to all time series features
        ```
    """

    def __init__(self, noise_level: float, mode: str = "normal"):
        """
        Initialize FeatureQualityStrategy.

        Args:
            noise_level (float): Standard deviation of the Gaussian noise (must be >= 0).
            mode (str): Mode for noise distribution mean ("normal").

        Raises:
            ValueError: If noise_level is negative or mode is invalid.
        """
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative.")

        valid_modes = ["normal"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got: {mode}")

        super().__init__()
        self.noise_level = noise_level
        self.mode = mode
        self.logger.info(
            f"Initialized FeatureQualityStrategy with noise_level: {self.noise_level}, "
            f"mode: {self.mode}"
        )

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply feature quality degradation by adding Gaussian noise to time series data.

        Args:
            X (np.ndarray): Input time series data.
            y (np.ndarray): Target labels (unchanged).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Noisy X data and original labels.
        """
        self.logger.info(
            f"Applying FeatureQualityStrategy with noise_level: {self.noise_level}, mode: {self.mode}"
        )

        if self.noise_level == 0:
            self.logger.info("Noise level is 0, returning unchanged data")
            return X, y

        # Generate noise with same shape as X
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=X.shape)

        # Add noise to the data
        X_noisy = X + noise

        self.logger.info(
            f"FeatureQualityStrategy applied successfully. "
            f"Noise std: {np.std(noise):.4f}, "
            f"Output shape: {X_noisy.shape}"
        )

        return X_noisy, y
