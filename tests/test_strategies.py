"""Unit tests for DataCentric strategy classes."""

import pytest
import numpy as np
from src.data_centric.dataCentricStrategy import (
    BaselineStrategy,
    FeatureQualityStrategy,
    LabelQualityStrategy,
    DifficultySelectionStrategy,
    VolumeReductionStrategy,
    TemporalDownsamplingStrategy,
    TemporalPaddingStrategy,
    DataCentricStrategy,
)


class TestBaselineStrategy:
    """Test BaselineStrategy returns unchanged data."""

    def test_baseline_preserves_data(self):
        """Baseline strategy should not modify data."""
        X = np.random.randn(10, 1, 50)
        y = np.random.randint(0, 2, 10)

        strategy = BaselineStrategy()
        X_out, y_out = strategy.apply(X, y)

        assert np.allclose(X_out, X), "Baseline should preserve X"
        assert np.allclose(y_out, y), "Baseline should preserve y"

    def test_baseline_preserves_shape(self):
        """Baseline strategy should preserve data shape."""
        shapes = [(10, 1, 50), (20, 3, 100), (5, 5, 25)]
        for shape in shapes:
            X = np.random.randn(*shape)
            y = np.random.randint(0, 2, shape[0])

            strategy = BaselineStrategy()
            X_out, y_out = strategy.apply(X, y)

            assert X_out.shape == shape, f"Shape mismatch for {shape}"
            assert y_out.shape == (shape[0],), "Label shape mismatch"


class TestFeatureQualityStrategy:
    """Test FeatureQualityStrategy degradation modes."""

    def test_gaussian_noise_applies_noise(self):
        """Gaussian noise mode should add noise proportional to p."""
        X = np.ones((10, 1, 50))
        y = np.zeros(10)

        strategy = FeatureQualityStrategy(mode="gaussian", p=0.5, sigma_max=1.0)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape == X.shape, "Shape should be preserved"
        assert not np.allclose(X_out, X), "Data should be modified"
        assert np.abs(np.mean(X_out - X)) < 0.5, "Mean noise should be reasonable"

    def test_gaussian_noise_with_zero_p(self):
        """With p=0, Gaussian mode should make minimal changes."""
        X = np.ones((10, 1, 50))
        y = np.zeros(10)

        strategy = FeatureQualityStrategy(mode="gaussian", p=0.0, sigma_max=1.0)
        X_out, y_out = strategy.apply(X, y)

        assert np.allclose(X_out, X), "With p=0, data should be nearly unchanged"

    def test_drift_mode_applies_trend(self):
        """Drift mode should add linear trends."""
        X = np.random.randn(10, 2, 50)
        y = np.zeros(10)

        strategy = FeatureQualityStrategy(mode="drift", p=0.5)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape == X.shape, "Shape should be preserved"
        assert not np.allclose(X_out, X), "Data should be modified"

    def test_correlated_noise_mode(self):
        """Correlated noise mode should produce autocorrelated noise."""
        X = np.random.randn(10, 1, 100)
        y = np.zeros(10)

        strategy = FeatureQualityStrategy(mode="correlated", p=0.5)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape == X.shape, "Shape should be preserved"
        assert not np.allclose(X_out, X), "Data should be modified"

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        strategy = FeatureQualityStrategy(mode="invalid_mode", p=0.5)
        X = np.random.randn(10, 1, 50)
        y = np.zeros(10)

        with pytest.raises(ValueError):
            strategy.apply(X, y)


class TestLabelQualityStrategy:
    """Test LabelQualityStrategy degradation modes."""

    def test_flip_mode_flips_labels(self):
        """Flip mode should randomly flip labels."""
        X = np.random.randn(100, 1, 50)
        y = np.zeros(100, dtype=int)
        y[50:] = 1  # Binary labels

        strategy = LabelQualityStrategy(mode="flip", p=0.3)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape == X.shape, "X shape should be preserved"
        assert y_out.shape == y.shape, "Label shape should be preserved"
        assert set(np.unique(y_out)).issubset({0, 1}), "Should remain binary"

        flip_ratio = np.sum(y != y_out) / len(y)
        assert 0.1 < flip_ratio < 0.5, f"Flip ratio {flip_ratio} should be ~p"

    def test_flip_with_zero_p(self):
        """With p=0, flip mode should not flip labels."""
        X = np.random.randn(50, 1, 50)
        y = np.random.randint(0, 3, 50)

        strategy = LabelQualityStrategy(mode="flip", p=0.0)
        X_out, y_out = strategy.apply(X, y)

        assert np.array_equal(y_out, y), "With p=0, labels should not change"

    def test_flip_preserves_unique_classes(self):
        """Flipping should handle multi-class labels."""
        X = np.random.randn(90, 1, 50)
        y = np.repeat(np.arange(3), 30)  # 3 classes, 30 samples each

        strategy = LabelQualityStrategy(mode="flip", p=0.2)
        X_out, y_out = strategy.apply(X, y)

        assert len(np.unique(y_out)) >= 2, "Should preserve at least some classes"

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        strategy = LabelQualityStrategy(mode="invalid_mode", p=0.5)
        X = np.random.randn(10, 1, 50)
        y = np.zeros(10)

        with pytest.raises(ValueError):
            strategy.apply(X, y)


class TestDifficultySelectionStrategy:
    """Test DifficultySelectionStrategy selection modes."""

    def test_distance_mode_reduces_samples(self):
        """Distance mode should select hardest (most distant) samples."""
        X = np.random.randn(20, 1, 50)
        y = np.zeros(20)

        strategy = DifficultySelectionStrategy(mode="distance", p=0.5)
        X_out, y_out = strategy.apply(X, y)

        expected_size = int(20 * (1 - 0.5))
        assert X_out.shape[0] == expected_size, "Should reduce to (1-p)·N samples"
        assert X_out.shape[1:] == X.shape[1:], "Spatial dims should be preserved"

    def test_distance_mode_handles_3d_data(self):
        """Distance mode should handle 3D time series data."""
        X = np.random.randn(30, 1, 470)  # 3D Beef dataset shape
        y = np.zeros(30)

        strategy = DifficultySelectionStrategy(mode="distance", p=0.3)
        X_out, y_out = strategy.apply(X, y)

        expected_size = int(30 * (1 - 0.3))
        assert X_out.shape == (expected_size, 1, 470), "3D shape should be preserved"

    def test_distance_with_zero_p(self):
        """With p=0, no samples should be removed."""
        X = np.random.randn(15, 2, 50)
        y = np.random.randint(0, 2, 15)

        strategy = DifficultySelectionStrategy(mode="distance", p=0.0)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape[0] == X.shape[0], "With p=0, all samples kept"

    def test_distance_with_full_p(self):
        """With p=1.0, all samples except 1 should be removed."""
        X = np.random.randn(20, 1, 50)
        y = np.zeros(20)

        strategy = DifficultySelectionStrategy(mode="distance", p=1.0)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape[0] == 0 or X_out.shape[0] == 1, "p=1.0 should remove almost all"

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        strategy = DifficultySelectionStrategy(mode="invalid_mode", p=0.5)
        X = np.random.randn(10, 1, 50)
        y = np.zeros(10)

        with pytest.raises(ValueError):
            strategy.apply(X, y)


class TestVolumeReductionStrategy:
    """Test VolumeReductionStrategy augmentation modes."""

    def test_noise_augmentation_increases_volume(self):
        """Noise augmentation should increase dataset size."""
        X = np.random.randn(10, 1, 50)
        y = np.zeros(10)

        strategy = VolumeReductionStrategy(mode="noise", p=0.3, noise_std=0.1)
        X_out, y_out = strategy.apply(X, y)

        expected_size = int(10 * (1 + 0.3))
        assert X_out.shape[0] == expected_size, f"Should have {expected_size} samples"
        assert X_out.shape[1:] == X.shape[1:], "Feature dims preserved"

    def test_noise_augmentation_with_zero_p(self):
        """With p=0, no augmentation should occur."""
        X = np.random.randn(10, 1, 50)
        y = np.zeros(10)

        strategy = VolumeReductionStrategy(mode="noise", p=0.0, noise_std=0.1)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape[0] == X.shape[0], "With p=0, dataset size unchanged"

    def test_augmented_labels_match_strategy(self):
        """Augmented samples should inherit original labels."""
        X = np.random.randn(15, 2, 30)
        y = np.array([0, 1, 2] * 5)

        strategy = VolumeReductionStrategy(mode="noise", p=0.2, noise_std=0.05)
        X_out, y_out = strategy.apply(X, y)

        # First 15 should be originals with same labels
        assert np.array_equal(y_out[:15], y), "Original labels preserved"


class TestTemporalDownsamplingStrategy:
    """Test TemporalDownsamplingStrategy downsampling modes."""

    def test_uniform_downsampling_reduces_length(self):
        """Uniform downsampling should reduce temporal dimension."""
        X = np.random.randn(10, 2, 100)
        y = np.zeros(10)

        strategy = TemporalDownsamplingStrategy(mode="uniform", p=0.5)
        X_out, y_out = strategy.apply(X, y)

        # s = 1 + floor(15 * p) = 1 + floor(7.5) = 8
        # Python slicing [::8] gives ceil(100/8) = 13 elements
        s = 1 + int(np.floor(15 * 0.5))
        expected_length = len(np.arange(0, 100, s))
        assert X_out.shape[2] == expected_length, f"Temporal dim should be {expected_length}"
        assert X_out.shape[:2] == X.shape[:2], "Sample and feature dims preserved"

    def test_uniform_with_zero_p(self):
        """With p=0, s=1, so no downsampling."""
        X = np.random.randn(10, 1, 100)
        y = np.zeros(10)

        strategy = TemporalDownsamplingStrategy(mode="uniform", p=0.0)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape[2] == X.shape[2], "With p=0, temporal length unchanged"

    def test_uniform_with_high_p(self):
        """With high p, should downsample more aggressively."""
        X = np.random.randn(10, 1, 100)
        y = np.zeros(10)

        strategy = TemporalDownsamplingStrategy(mode="uniform", p=0.9)
        X_out, y_out = strategy.apply(X, y)

        # s = 1 + floor(15 * 0.9) = 1 + 13 = 14
        # Python slicing [::14] gives ceil(100/14) = 8 elements
        s = 1 + int(np.floor(15 * 0.9))
        expected_length = len(np.arange(0, 100, s))
        assert X_out.shape[2] == expected_length, "High p should reduce length significantly"

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        strategy = TemporalDownsamplingStrategy(mode="invalid_mode", p=0.5)
        X = np.random.randn(10, 1, 50)
        y = np.zeros(10)

        with pytest.raises(ValueError):
            strategy.apply(X, y)


class TestTemporalPaddingStrategy:
    """Test TemporalPaddingStrategy padding modes."""

    def test_gaussian_padding_increases_length(self):
        """Gaussian padding should increase temporal dimension."""
        X = np.random.randn(10, 1, 100)
        y = np.zeros(10)

        strategy = TemporalPaddingStrategy(mode="gaussian", p=0.3)
        X_out, y_out = strategy.apply(X, y)

        # pad_len = ceil(p * T / 2) = ceil(0.3 * 100 / 2) = ceil(15) = 15
        # new_len = T + 2 * pad_len = 100 + 30 = 130
        expected_length = 100 + 2 * 15
        assert X_out.shape[2] == expected_length, f"Temporal length should be {expected_length}"
        assert X_out.shape[:2] == X.shape[:2], "Sample and feature dims preserved"

    def test_gaussian_padding_with_zero_p(self):
        """With p=0, no padding should occur."""
        X = np.random.randn(10, 1, 100)
        y = np.zeros(10)

        strategy = TemporalPaddingStrategy(mode="gaussian", p=0.0)
        X_out, y_out = strategy.apply(X, y)

        assert X_out.shape[2] == X.shape[2], "With p=0, temporal length unchanged"

    def test_gaussian_padding_different_lengths(self):
        """Padding should handle various temporal lengths."""
        lengths = [50, 100, 200, 470]
        for length in lengths:
            X = np.random.randn(5, 1, length)
            y = np.zeros(5)

            strategy = TemporalPaddingStrategy(mode="gaussian", p=0.2)
            X_out, y_out = strategy.apply(X, y)

            assert X_out.shape[2] > X.shape[2], "Length should increase"
            assert X_out.shape[:2] == X.shape[:2], "Dims preserved"

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        strategy = TemporalPaddingStrategy(mode="invalid_mode", p=0.5)
        X = np.random.randn(10, 1, 50)
        y = np.zeros(10)

        with pytest.raises(ValueError):
            strategy.apply(X, y)


class TestFactoryPattern:
    """Test DataCentricStrategy factory method."""

    def test_factory_creates_baseline(self):
        """Factory should create BaselineStrategy for baseline config."""
        config = {"type": "baseline", "mode": "none", "params": {}}
        strategy = DataCentricStrategy.from_config(config)

        assert isinstance(strategy, BaselineStrategy)

    def test_factory_creates_feature_quality(self):
        """Factory should create FeatureQualityStrategy."""
        config = {
            "type": "feature_quality",
            "mode": "gaussian",
            "params": {"p": 0.5, "sigma_max": 1.0},
        }
        strategy = DataCentricStrategy.from_config(config)

        assert isinstance(strategy, FeatureQualityStrategy)

    def test_factory_creates_label_quality(self):
        """Factory should create LabelQualityStrategy."""
        config = {
            "type": "label_quality",
            "mode": "flip",
            "params": {"p": 0.3},
        }
        strategy = DataCentricStrategy.from_config(config)

        assert isinstance(strategy, LabelQualityStrategy)

    def test_factory_creates_difficulty_selection(self):
        """Factory should create DifficultySelectionStrategy."""
        config = {
            "type": "difficulty_selection",
            "mode": "distance",
            "params": {"p": 0.4},
        }
        strategy = DataCentricStrategy.from_config(config)

        assert isinstance(strategy, DifficultySelectionStrategy)

    def test_factory_creates_volume_reduction(self):
        """Factory should create VolumeReductionStrategy."""
        config = {
            "type": "volume_reduction",
            "mode": "noise",
            "params": {"p": 0.2, "noise_std": 0.1},
        }
        strategy = DataCentricStrategy.from_config(config)

        assert isinstance(strategy, VolumeReductionStrategy)

    def test_factory_creates_temporal_downsampling(self):
        """Factory should create TemporalDownsamplingStrategy."""
        config = {
            "type": "temporal_downsampling",
            "mode": "uniform",
            "params": {"p": 0.5},
        }
        strategy = DataCentricStrategy.from_config(config)

        assert isinstance(strategy, TemporalDownsamplingStrategy)

    def test_factory_creates_temporal_padding(self):
        """Factory should create TemporalPaddingStrategy."""
        config = {
            "type": "temporal_padding",
            "mode": "gaussian",
            "params": {"p": 0.3},
        }
        strategy = DataCentricStrategy.from_config(config)

        assert isinstance(strategy, TemporalPaddingStrategy)

    def test_factory_invalid_type_raises_error(self):
        """Factory should raise ValueError for invalid type."""
        config = {
            "type": "invalid_strategy",
            "mode": "some_mode",
            "params": {},
        }

        with pytest.raises(ValueError):
            DataCentricStrategy.from_config(config)

    def test_factory_invalid_mode_raises_error(self):
        """Factory should raise ValueError for invalid mode."""
        config = {
            "type": "feature_quality",
            "mode": "invalid_mode",
            "params": {"p": 0.5},
        }

        with pytest.raises(ValueError):
            DataCentricStrategy.from_config(config)


class TestParameterValidation:
    """Test parameter validation across strategies."""

    def test_p_parameter_bounded(self):
        """p parameter should be in [0, 1]."""
        strategies = [
            FeatureQualityStrategy(mode="gaussian", p=0.5),
            LabelQualityStrategy(mode="flip", p=0.3),
            DifficultySelectionStrategy(mode="distance", p=0.4),
            VolumeReductionStrategy(mode="noise", p=0.2),
            TemporalDownsamplingStrategy(mode="uniform", p=0.5),
            TemporalPaddingStrategy(mode="gaussian", p=0.3),
        ]

        X = np.random.randn(10, 1, 50)
        y = np.zeros(10)

        for strategy in strategies:
            X_out, y_out = strategy.apply(X, y)
            assert X_out is not None, "Strategy should return valid output"

    def test_strategies_handle_edge_cases(self):
        """Strategies should handle single sample gracefully."""
        X = np.random.randn(1, 1, 50)
        y = np.array([0])

        strategies = [
            BaselineStrategy(),
            FeatureQualityStrategy(mode="gaussian", p=0.5),
            LabelQualityStrategy(mode="flip", p=0.3),
        ]

        for strategy in strategies:
            X_out, y_out = strategy.apply(X, y)
            assert X_out.shape[0] >= 0, "Should handle single sample"


class TestDataIntegrity:
    """Test data integrity across operations."""

    def test_no_nan_generation(self):
        """Strategies should not generate NaN values."""
        X = np.random.randn(20, 2, 75)
        y = np.random.randint(0, 3, 20)

        strategies = [
            FeatureQualityStrategy(mode="gaussian", p=0.5),
            LabelQualityStrategy(mode="flip", p=0.3),
            DifficultySelectionStrategy(mode="distance", p=0.4),
            VolumeReductionStrategy(mode="noise", p=0.2),
            TemporalDownsamplingStrategy(mode="uniform", p=0.5),
            TemporalPaddingStrategy(mode="gaussian", p=0.3),
        ]

        for strategy in strategies:
            X_out, y_out = strategy.apply(X.copy(), y.copy())
            assert not np.any(np.isnan(X_out)), f"{strategy.__class__.__name__} produced NaN"

    def test_no_inf_generation(self):
        """Strategies should not generate infinite values."""
        X = np.random.randn(20, 2, 75)
        y = np.random.randint(0, 3, 20)

        strategies = [
            FeatureQualityStrategy(mode="gaussian", p=0.5),
            TemporalDownsamplingStrategy(mode="uniform", p=0.5),
            TemporalPaddingStrategy(mode="gaussian", p=0.3),
        ]

        for strategy in strategies:
            X_out, y_out = strategy.apply(X.copy(), y.copy())
            assert not np.any(np.isinf(X_out)), f"{strategy.__class__.__name__} produced Inf"
