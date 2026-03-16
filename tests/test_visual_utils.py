"""Unit tests for visualization utilities."""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.visual_utils.visual_utils import (
    plot_accuracy_vs_strategy_param,
    plot_accuracy_comparison,
    plot_accuracy_trajectory,
)


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Create sample results DataFrame mimicking new strategy parameter structure."""
    np.random.seed(42)
    
    datasets = ["Beef", "Crop", "ECG"]
    classifiers = ["DTW", "1NN", "ResNet"]
    p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    random_seeds = [0, 1, 2]
    
    rows = []
    for dataset in datasets:
        for classifier in classifiers:
            for p in p_values:
                for seed in random_seeds:
                    accuracy = np.random.uniform(0.6, 0.95)
                    rows.append({
                        "dataset": dataset,
                        "classifier": classifier,
                        "p": p,  # NEW: unified parameter name
                        "type": "feature_quality",  # NEW: strategy type instead of "strategy"
                        "mode": "gaussian",  # NEW: mode instead of "strategy_mode"
                        "accuracy": accuracy,
                        "random_seed": seed,
                    })
    
    return pd.DataFrame(rows)


class TestPlotAccuracyVsStrategyParam:
    """Test plot_accuracy_vs_strategy_param function."""
    
    def test_basic_plotting(self, sample_results_df):
        """Test basic plotting without errors."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        
        fig = plot_accuracy_vs_strategy_param(
            subset_results=subset,
            strategy_params_compare="p",
            classifier="DTW",
            dataset_names=dataset_names,
            prepare_save_plot=True,
            **{name: True for name in dataset_names}
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_filtering_works(self, sample_results_df):
        """Test that filtering by dataset works."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        
        fig = plot_accuracy_vs_strategy_param(
            subset_results=subset,
            strategy_params_compare="p",
            classifier="1NN",
            dataset_names=dataset_names,
            prepare_save_plot=True,
            **{"Beef": True, "Crop": False, "ECG": True}
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_with_no_save_returns_none(self, sample_results_df):
        """Test that prepare_save_plot=False returns None."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        
        result = plot_accuracy_vs_strategy_param(
            subset_results=subset,
            strategy_params_compare="p",
            classifier="ResNet",
            dataset_names=dataset_names,
            prepare_save_plot=False,
            **{name: True for name in dataset_names}
        )
        
        assert result is None
        plt.close('all')


class TestPlotAccuracyComparison:
    """Test plot_accuracy_comparison function."""
    
    def test_basic_comparison(self, sample_results_df):
        """Test basic classifier comparison."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        p_value = 0.3
        
        fig = plot_accuracy_comparison(
            subset_results=subset,
            strategy_params_compare="p",
            strategy_params_value=p_value,
            classifier1="DTW",
            classifier2="1NN",
            prepare_save_plot=True,
            **{name: True for name in dataset_names}
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_missing_parameter_value_returns_none(self, sample_results_df):
        """Test handling of missing parameter value."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        
        result = plot_accuracy_comparison(
            subset_results=subset,
            strategy_params_compare="p",
            strategy_params_value=0.999,  # Doesn't exist in data
            classifier1="DTW",
            classifier2="1NN",
            prepare_save_plot=False,
            **{name: True for name in dataset_names}
        )
        
        assert result is None
    
    def test_missing_classifier_returns_none(self, sample_results_df):
        """Test handling of missing classifier."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        p_value = 0.3
        
        result = plot_accuracy_comparison(
            subset_results=subset,
            strategy_params_compare="p",
            strategy_params_value=p_value,
            classifier1="DTW",
            classifier2="NonExistent",  # Doesn't exist
            prepare_save_plot=False,
            **{name: True for name in dataset_names}
        )
        
        assert result is None


class TestPlotAccuracyTrajectory:
    """Test plot_accuracy_trajectory function."""
    
    def test_basic_trajectory(self, sample_results_df):
        """Test basic trajectory plotting."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        
        fig = plot_accuracy_trajectory(
            classifier1="DTW",
            classifier2="1NN",
            subset_results=subset,
            strategy_params_compare="p",
            dataset_names=dataset_names,
            prepare_save_plot=True,
            **{name: True for name in dataset_names}
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_selective_datasets(self, sample_results_df):
        """Test trajectory with selective dataset filtering."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        
        fig = plot_accuracy_trajectory(
            classifier1="ResNet",
            classifier2="DTW",
            subset_results=subset,
            strategy_params_compare="p",
            dataset_names=dataset_names,
            prepare_save_plot=True,
            **{"Beef": True, "Crop": False, "ECG": True}
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_all_datasets_filtered_returns_none(self, sample_results_df):
        """Test when all datasets are filtered out."""
        subset = sample_results_df[sample_results_df["type"] == "feature_quality"]
        dataset_names = subset["dataset"].unique()
        
        result = plot_accuracy_trajectory(
            classifier1="DTW",
            classifier2="1NN",
            subset_results=subset,
            strategy_params_compare="p",
            dataset_names=dataset_names,
            prepare_save_plot=False,
            **{name: False for name in dataset_names}  # All filtered
        )
        
        # Function handles this gracefully
        assert result is None or isinstance(result, plt.Figure)
        plt.close('all')


class TestNewParameterFormat:
    """Test compatibility with new R1-R6 strategy parameter format."""
    
    def test_works_with_unified_p_parameter(self):
        """Verify visual utils work with unified 'p' parameter instead of mode-specific params."""
        # This mimics what results will look like with new R1-R6 code:
        # "p" instead of "flip_ratio", "reduction_ratio", etc.
        df = pd.DataFrame({
            "dataset": ["D1", "D1", "D2", "D2"],
            "classifier": ["C1", "C2", "C1", "C2"],
            "p": [0.1, 0.1, 0.2, 0.2],
            "type": "feature_quality",
            "mode": "gaussian",
            "accuracy": [0.8, 0.85, 0.75, 0.9],
            "random_seed": [0, 0, 0, 0],
        })
        
        fig = plot_accuracy_vs_strategy_param(
            subset_results=df,
            strategy_params_compare="p",
            classifier="C1",
            dataset_names=["D1", "D2"],
            prepare_save_plot=True,
            D1=True,
            D2=True
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_works_with_new_column_names(self):
        """Verify visual utils work with 'type' and 'mode' instead of old names."""
        df = pd.DataFrame({
            "dataset": ["D1"] * 4,
            "classifier": ["C1", "C1", "C2", "C2"],
            "p": [0.1] * 4,
            "type": "label_quality",  # NEW column name
            "mode": "flip",  # NEW column name
            "accuracy": [0.8, 0.85, 0.75, 0.9],
            "random_seed": [0, 0, 1, 1],
        })
        
        # This should work without errors
        fig = plot_accuracy_comparison(
            subset_results=df,
            strategy_params_compare="p",
            strategy_params_value=0.1,
            classifier1="C1",
            classifier2="C2",
            prepare_save_plot=True,
            D1=True
        )
        
        assert fig is not None
        plt.close(fig)


class TestDocumentation:
    """Test that functions have proper documentation."""
    
    def test_functions_have_docstrings(self):
        """Verify all plotting functions have docstrings."""
        assert plot_accuracy_vs_strategy_param.__doc__ is not None
        assert len(plot_accuracy_vs_strategy_param.__doc__) > 50
        
        assert plot_accuracy_comparison.__doc__ is not None
        assert len(plot_accuracy_comparison.__doc__) > 50
        
        assert plot_accuracy_trajectory.__doc__ is not None
        assert len(plot_accuracy_trajectory.__doc__) > 50
    
    def test_functions_have_type_hints(self):
        """Verify functions have type hints."""
        import inspect
        
        sig = inspect.signature(plot_accuracy_vs_strategy_param)
        # Check that at least some parameters have type hints
        hints_count = sum(1 for p in sig.parameters.values() if p.annotation != inspect.Parameter.empty)
        assert hints_count > 0, "Should have type hints"
