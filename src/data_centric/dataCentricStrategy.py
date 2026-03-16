import numpy as np
from typing import Any, Dict, Tuple, Optional
from scipy.spatial.distance import euclidean
from ..utils.logger import get_logger


class DataCentricStrategy:
    """
    Base class for data-centric adaptation strategies in time series classification.

    This abstract base class defines the interface for implementing various data-centric
    strategies that modify training data to improve model performance. All concrete
    strategy implementations must inherit from this class and implement the apply method.

    Unified Parameter Interface:
        All strategy subclasses follow a common initialization pattern with these parameters:
        - mode (str): Implementation variant selection (e.g., "gaussian", "flip", "distance").
            Different modes implement different algorithmic approaches for the same strategy.
        - p (float): Degradation/modification intensity in [0, 1]. Controls the strength of
            the strategy application (0 = no modification, 1 = maximum modification).
        - Additional strategy-specific parameters (e.g., sigma_max for noise magnitude).

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
            "type": "feature_quality",
            "mode": "gaussian",
            "params": {"p": 0.1}
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

        Args:
            conf (Dict[str, Any]): Configuration dictionary with keys:
                - type (str): Strategy type
                - mode (str): Strategy mode/variant
                - params (Dict, optional): Strategy-specific parameters

        Returns:
            DataCentricStrategy: Instantiated strategy object.

        Raises:
            ValueError: If the strategy configuration is unknown or invalid.
        """
        logger = get_logger(__name__)
        logger.info(f"Creating strategy from config: {conf}")

        strategy_registry = {
            ("baseline", None): BaselineStrategy,
            ("baseline", "none"): BaselineStrategy,
            ("feature_quality", "gaussian"): FeatureQualityStrategy,
            ("feature_quality", "drift"): FeatureQualityStrategy,
            ("feature_quality", "correlated"): FeatureQualityStrategy,
            ("label_quality", "flip"): LabelQualityStrategy,
            ("label_quality", "confusion"): LabelQualityStrategy,
            ("difficulty_selection", "distance"): DifficultySelectionStrategy,
            ("difficulty_selection", "density"): DifficultySelectionStrategy,
            ("difficulty_selection", "uncertainty"): DifficultySelectionStrategy,
            ("volume_reduction", "noise"): VolumeReductionStrategy,
            ("volume_reduction", "drift"): VolumeReductionStrategy,
            ("volume_reduction", "label_feature"): VolumeReductionStrategy,
            ("temporal_downsampling", "uniform"): TemporalDownsamplingStrategy,
            ("temporal_downsampling", "truncation"): TemporalDownsamplingStrategy,
            ("temporal_downsampling", "random_gaps"): TemporalDownsamplingStrategy,
            ("temporal_padding", "gaussian"): TemporalPaddingStrategy,
            ("temporal_padding", "random_walk"): TemporalPaddingStrategy,
            ("temporal_padding", "asymmetric"): TemporalPaddingStrategy,
        }

        key = (conf["type"], conf.get("mode"))
        StrategyClass = strategy_registry.get(key)

        if StrategyClass is None:
            logger.error(f"Unknown strategy configuration: {key}")
            raise ValueError(f"Unknown strategy configuration: {key}")

        logger.info(f"Strategy {StrategyClass.__name__} created successfully")
        mode = conf.get("mode")
        if mode == "none":
            mode = None
        return StrategyClass(mode=mode, **conf.get("params", {}))


class BaselineStrategy(DataCentricStrategy):
    """Baseline strategy that applies no data-centric modifications."""

    def __init__(self, mode: Optional[str] = None, **kwargs):
        super().__init__()
        self.logger.info("Initialized BaselineStrategy (no data-centric adaptation)")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply baseline strategy (no modifications)."""
        self.logger.info("Applying BaselineStrategy (no changes)")
        return X, y


class FeatureQualityStrategy(DataCentricStrategy):
    """Improve feature quality via noise addition.
    
    Degrades features by adding different types of noise to simulate real-world sensor
    variability and measurement errors.
    
    Modes:
        - "gaussian" (default): Pure Gaussian noise with σ(p) = p·σ_max.
        - "drift": Sensor drift with linear trend, β(p) = p·0.1.
        - "correlated": AR(1) temporally correlated noise with φ=0.7.
    
    Args:
        mode (str): Implementation variant (see Modes above).
        p (float): Degradation intensity in [0, 1].
        sigma_max (float): Maximum noise standard deviation. Default: 1.0.
        **kwargs: Additional arguments passed to parent class.
    
    Raises:
        ValueError: If p is not in [0, 1].
    """

    def __init__(self, mode: str = "gaussian", p: float = 0.0, sigma_max: float = 1.0, **kwargs):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        
        self.mode = mode
        self.p = p
        self.sigma_max = sigma_max
        self.logger.info(f"FeatureQualityStrategy initialized (mode={mode}, p={p})")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature degradation based on mode."""
        if self.mode == "gaussian":
            return self._apply_gaussian_noise(X, y)
        elif self.mode == "drift":
            return self._apply_sensor_drift(X, y)
        elif self.mode == "correlated":
            return self._apply_correlated_noise(X, y)
        else:
            raise ValueError(f"Unknown feature_quality mode: {self.mode}")

    def _apply_gaussian_noise(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Primary variant: Pure Gaussian noise."""
        if self.p == 0:
            return X.copy(), y.copy()
        
        sigma = self.p * self.sigma_max
        noise = np.random.normal(0, sigma, X.shape)
        X_noisy = X + noise
        
        self.logger.info(f"FeatureQualityStrategy.gaussian applied: σ={sigma:.4f}")
        return X_noisy, y.copy()

    def _apply_sensor_drift(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Variant 1: Sensor drift with linear trend."""
        if self.p == 0:
            return X.copy(), y.copy()
        
        T = X.shape[-1]  # Last dimension is temporal
        beta = self.p * 0.1
        t_array = np.arange(T)
        drift = beta * t_array
        sigma_noise = self.p * self.sigma_max
        noise = np.random.normal(0, sigma_noise, X.shape)
        
        # Broadcast drift to match shape: reshape to (1, 1, ..., 1, T)
        drift_shape = [1] * (X.ndim - 1) + [T]
        drift_broadcast = drift.reshape(drift_shape)
        X_degraded = X + drift_broadcast + noise
        
        self.logger.info(f"FeatureQualityStrategy.drift applied: p={self.p}")
        return X_degraded, y.copy()

    def _apply_correlated_noise(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Variant 2: AR(1) temporally correlated noise φ=0.7."""
        if self.p == 0:
            return X.copy(), y.copy()
        
        phi = 0.7
        sigma_noise = self.p * self.sigma_max
        noise = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            zeta = np.random.normal(0, sigma_noise, X.shape[1])
            noise[i, 0] = zeta[0]
            for t in range(1, X.shape[1]):
                noise[i, t] = phi * noise[i, t-1] + zeta[t]
        
        X_degraded = X + noise
        self.logger.info(f"FeatureQualityStrategy.correlated applied: p={self.p}")
        return X_degraded, y.copy()


class LabelQualityStrategy(DataCentricStrategy):
    """Improve label quality via label flipping.
    
    Degrades labels to simulate annotation errors and class confusion.
    
    Modes:
        - "flip" (default): Uniform random label flipping with flip_prob = p.
        - "confusion": Confusion matrix based corruption (TODO).
    
    Args:
        mode (str): Implementation variant (see Modes above).
        p (float): Degradation intensity in [0, 1].
        **kwargs: Additional arguments passed to parent class.
    
    Raises:
        ValueError: If p is not in [0, 1].
    """

    def __init__(self, mode: str = "flip", p: float = 0.0, **kwargs):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        
        self.mode = mode
        self.p = p
        self.logger.info(f"LabelQualityStrategy initialized (mode={mode}, p={p})")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply label degradation based on mode."""
        if self.mode == "flip":
            return self._apply_label_flip(X, y)
        elif self.mode == "confusion":
            return self._apply_confusion_matrix(X, y)
        else:
            raise ValueError(f"Unknown label_quality mode: {self.mode}")

    def _apply_label_flip(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Primary variant: Uniform random label flipping."""
        if self.p == 0:
            return X.copy(), y.copy()
        
        y_flipped = y.copy()
        n_samples = len(y)
        n_flip = int(np.ceil(self.p * n_samples))
        flip_indices = np.random.choice(n_samples, size=min(n_flip, n_samples), replace=False)
        
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            self.logger.warning("Cannot flip labels with only 1 class")
            return X.copy(), y_flipped
        
        for idx in flip_indices:
            available = unique_labels[unique_labels != y[idx]]
            y_flipped[idx] = np.random.choice(available)
        
        self.logger.info(f"LabelQualityStrategy.flip applied: p={self.p}, {len(flip_indices)}/{n_samples} flipped")
        return X.copy(), y_flipped

    def _apply_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Confusion matrix based label corruption."""
        self.logger.warning("LabelQualityStrategy.confusion not implemented, returning unchanged")
        return X.copy(), y.copy()


class DifficultySelectionStrategy(DataCentricStrategy):
    """Increase volume of high-relevance instances by removing easy cases.
    
    Selects hard/difficult instances by keeping only the most challenging samples
    per class. Removes (1-p)*N_k instances, keeping the hardest ones.
    
    Modes:
        - "distance" (default): Keep hardest instances by distance to class centroid.
        - "density": Density-based instance selection (TODO).
        - "uncertainty": Uncertainty-based instance selection (TODO).
    
    Args:
        mode (str): Implementation variant (see Modes above).
        p (float): Reduction intensity in [0, 1].
        **kwargs: Additional arguments passed to parent class.
    
    Raises:
        ValueError: If p is not in [0, 1].
    """

    def __init__(self, mode: str = "distance", p: float = 0.0, **kwargs):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        
        self.mode = mode
        self.p = p
        self.logger.info(f"DifficultySelectionStrategy initialized (mode={mode}, p={p})")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply difficulty-based instance selection based on mode."""
        if self.mode == "distance":
            return self._apply_distance_based(X, y)
        elif self.mode == "density":
            return self._apply_density_based(X, y)
        elif self.mode == "uncertainty":
            return self._apply_uncertainty_based(X, y)
        else:
            raise ValueError(f"Unknown difficulty_selection mode: {self.mode}")

    def _apply_distance_based(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Primary variant: Keep hardest instances by distance to centroid."""
        if self.p == 0:
            return X.copy(), y.copy()
        
        # Flatten to 2D if needed (handle shape (n_samples, n_channels, n_timesteps))
        original_shape = X.shape
        if X.ndim > 2:
            X_2d = X.reshape(X.shape[0], -1)
        else:
            X_2d = X
        
        X_selected = []
        y_selected = []
        
        for label in np.unique(y):
            mask = y == label
            X_class = X_2d[mask]
            y_class = y[mask]
            
            # Centroid distance: closer to centroid = easier (remove these)
            centroid = np.mean(X_class, axis=0)
            distances = np.array([euclidean(xi, centroid) for xi in X_class])
            
            # Keep hardest (1-p)·N_k instances (smallest distances = closest to boundary)
            n_keep = max(1, int(np.ceil((1 - self.p) * len(X_class))))
            keep_idx = np.argsort(distances)[:n_keep]  # Keep smallest distances (hardest)
            
            X_selected.append(X[mask][np.sort(keep_idx)])
            y_selected.append(y_class[np.sort(keep_idx)])
        
        X_result = np.vstack(X_selected)
        y_result = np.concatenate(y_selected)
        
        self.logger.info(f"DifficultySelectionStrategy.distance applied: {len(X_result)}/{len(X)} kept")
        return X_result, y_result

    def _apply_density_based(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Density-based instance selection."""
        self.logger.warning("DifficultySelectionStrategy.density not implemented, returning unchanged")
        return X.copy(), y.copy()

    def _apply_uncertainty_based(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Uncertainty-based instance selection."""
        self.logger.warning("DifficultySelectionStrategy.uncertainty not implemented, returning unchanged")
        return X.copy(), y.copy()


class VolumeReductionStrategy(DataCentricStrategy):
    """Reduce volume of low-quality instances by augmenting with corrupted copies.
    
    Augments the dataset with degraded versions of existing instances to increase
    effective training volume and robustness. Adds p*N corrupted copies.
    
    Modes:
        - "noise" (default): Gaussian noise corrupted copies.
        - "drift": Sensor drift corrupted copies (TODO).
        - "label_feature": Joint label and feature corruption (TODO).
    
    Args:
        mode (str): Implementation variant (see Modes above).
        p (float): Augmentation intensity in [0, 1] (fraction of new copies).
        sigma_max (float): Noise scale for corruption. Default: 1.0.
        **kwargs: Additional arguments passed to parent class.
    
    Raises:
        ValueError: If p is not in [0, 1].
    """

    def __init__(self, mode: str = "noise", p: float = 0.0, sigma_max: float = 1.0, **kwargs):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        
        self.mode = mode
        self.p = p
        self.sigma_max = sigma_max
        self.logger.info(f"VolumeReductionStrategy initialized (mode={mode}, p={p})")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply volume reduction via augmentation based on mode."""
        if self.mode == "noise":
            return self._apply_noise_augmentation(X, y)
        elif self.mode == "drift":
            return self._apply_drift_augmentation(X, y)
        elif self.mode == "label_feature":
            return self._apply_label_feature_augmentation(X, y)
        else:
            raise ValueError(f"Unknown volume_reduction mode: {self.mode}")

    def _apply_noise_augmentation(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Primary variant: Augment with Gaussian noise corrupted copies."""
        if self.p == 0:
            return X.copy(), y.copy()
        
        n_corrupt = int(np.ceil(self.p * len(X)))
        corrupt_idx = np.random.choice(len(X), size=n_corrupt, replace=False)
        
        X_copies = X[corrupt_idx].copy()
        y_copies = y[corrupt_idx].copy()
        sigma = self.p * self.sigma_max
        
        noise = np.random.normal(0, sigma, X_copies.shape)
        X_copies += noise
        
        X_aug = np.vstack([X, X_copies])
        y_aug = np.concatenate([y, y_copies])
        
        self.logger.info(f"VolumeReductionStrategy.noise applied: {len(X)} → {len(X_aug)} instances")
        return X_aug, y_aug

    def _apply_drift_augmentation(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Augment with sensor drift corrupted copies."""
        self.logger.warning("VolumeReductionStrategy.drift not implemented, returning unchanged")
        return X.copy(), y.copy()

    def _apply_label_feature_augmentation(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Augment with both label and feature corruption."""
        self.logger.warning("VolumeReductionStrategy.label_feature not implemented, returning unchanged")
        return X.copy(), y.copy()


class TemporalDownsamplingStrategy(DataCentricStrategy):
    """Increase volume of relevant features via temporal downsampling.
    
    Reduces temporal resolution through downsampling, creating new feature views.
    Simulates different sampling rates and helps evaluate stability.
    
    Modes:
        - "uniform" (default): Uniform downsampling with s(p) = 1 + floor(15p).
        - "truncation": Truncate to (1-p)*T timesteps (TODO).
        - "random_gaps": Random segment removal creating gaps (TODO).
    
    Args:
        mode (str): Implementation variant (see Modes above).
        p (float): Degradation intensity in [0, 1].
        **kwargs: Additional arguments passed to parent class.
    
    Raises:
        ValueError: If p is not in [0, 1].
    """

    def __init__(self, mode: str = "uniform", p: float = 0.0, **kwargs):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        
        self.mode = mode
        self.p = p
        self.logger.info(f"TemporalDownsamplingStrategy initialized (mode={mode}, p={p})")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal downsampling based on mode."""
        if self.mode == "uniform":
            return self._apply_uniform_downsampling(X, y)
        elif self.mode == "truncation":
            return self._apply_truncation(X, y)
        elif self.mode == "random_gaps":
            return self._apply_random_gaps(X, y)
        else:
            raise ValueError(f"Unknown temporal_downsampling mode: {self.mode}")

    def _apply_uniform_downsampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Primary variant: Uniform downsampling with s(p) = 1 + floor(15p)."""
        if self.p == 0:
            return X.copy(), y.copy()
        
        s = 1 + int(np.floor(self.p * 15))
        # Downsample the last dimension (temporal)
        X_down = X[..., ::s]
        
        self.logger.info(f"TemporalDownsamplingStrategy.uniform applied: {X.shape[-1]} → {X_down.shape[-1]} timesteps")
        return X_down, y.copy()

    def _apply_truncation(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Truncate sequences to (1-p)·T length."""
        self.logger.warning("TemporalDownsamplingStrategy.truncation not implemented, returning unchanged")
        return X.copy(), y.copy()

    def _apply_random_gaps(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Random temporal segment removal."""
        self.logger.warning("TemporalDownsamplingStrategy.random_gaps not implemented, returning unchanged")
        return X.copy(), y.copy()


class TemporalPaddingStrategy(DataCentricStrategy):
    """Reduce volume of unmeaningful features via temporal padding.
    
    Adds temporal noise padding to sequences to increase effective feature volume
    and test robustness to irrelevant temporal information. Pad length: ceil(p*T/2).
    
    Modes:
        - "gaussian" (default): Gaussian noise padding on both sides.
        - "random_walk": Random walk padding on both sides (TODO).
        - "asymmetric": Asymmetric prefix/suffix padding (TODO).
    
    Args:
        mode (str): Implementation variant (see Modes above).
        p (float): Padding intensity in [0, 1].
        **kwargs: Additional arguments passed to parent class.
    
    Raises:
        ValueError: If p is not in [0, 1].
    """

    def __init__(self, mode: str = "gaussian", p: float = 0.0, **kwargs):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        
        self.mode = mode
        self.p = p
        self.logger.info(f"TemporalPaddingStrategy initialized (mode={mode}, p={p})")

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal padding based on mode."""
        if self.mode == "gaussian":
            return self._apply_gaussian_padding(X, y)
        elif self.mode == "random_walk":
            return self._apply_random_walk_padding(X, y)
        elif self.mode == "asymmetric":
            return self._apply_asymmetric_padding(X, y)
        else:
            raise ValueError(f"Unknown temporal_padding mode: {self.mode}")

    def _apply_gaussian_padding(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Primary variant: Gaussian noise padding on both sides."""
        if self.p == 0:
            return X.copy(), y.copy()
        
        T = X.shape[-1]  # Last dimension is temporal
        pad_len = int(np.ceil(self.p * T / 2))
        
        # Pad the last dimension (temporal) with Gaussian noise
        pad_shape = list(X.shape[:-1]) + [pad_len]
        prefix = np.random.normal(0, 1.0, pad_shape)
        suffix = np.random.normal(0, 1.0, pad_shape)
        
        X_result = np.concatenate([prefix, X, suffix], axis=-1)
        
        self.logger.info(f"TemporalPaddingStrategy.gaussian applied: {T} → {X_result.shape[-1]} timesteps")
        return X_result, y.copy()

    def _apply_random_walk_padding(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Random walk padding on both sides."""
        self.logger.warning("TemporalPaddingStrategy.random_walk not implemented, returning unchanged")
        return X.copy(), y.copy()

    def _apply_asymmetric_padding(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """TODO: Asymmetric prefix/suffix padding."""
        self.logger.warning("TemporalPaddingStrategy.asymmetric not implemented, returning unchanged")
        return X.copy(), y.copy()
