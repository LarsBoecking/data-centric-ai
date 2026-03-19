
"""Visualization utilities for analyzing data-centric experiment results.

Provides plotting functions for analyzing experiment results from the data-centric
strategy optimization framework. Supports interactive visualization via ipywidgets
and batch figure export to PDF.
"""

import os

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import math

import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.getcwd(), "configs", "visualisations.mplstyle"))

from src.data_centric.dataCentricStrategy import DataCentricStrategy

def get_strategy_subset(
    summary_df: pd.DataFrame,
    strategy_type: str,
    strategy_mode: str,
) -> pd.DataFrame:
    """Filter results by strategy type and mode.
    
    Args:
        summary_df (pd.DataFrame): Full results dataframe from ResultHandler
        strategy_type (str): Strategy type (e.g., 'feature_quality', 'label_quality')
        strategy_mode (str): Strategy mode (e.g., 'gaussian', 'flip')
    
    Returns:
        pd.DataFrame: Filtered subset of results matching type and mode
    """
    subset = summary_df[
        (summary_df["type"] == strategy_type)
        & (summary_df["mode"] == strategy_mode)
    ]
    return subset.sort_values(
        by=["dataset", "classifier", "type", "mode"]
    )

def plot_data(strategy_config: Dict, X_train_raw: np.ndarray, y_train_raw: np.ndarray, title: str):
    """Plot raw and preprocessed data side by side for each class.
    
    Applies a data-centric strategy and visualizes both raw and preprocessed data
    for comparison. Creates a figure with raw data in the first row and preprocessed
    data in the second row, with subplots for each class (up to 5 columns).
    
    Args:
        strategy_config (Dict): Configuration dict for DataCentricStrategy.
            Example: {"type": "temporal_padding", "mode": "gaussian", "params": {"p": 0.4, "sigma": 1}}
        X_train_raw (np.ndarray): Raw training data of shape (n_samples, n_features, n_timesteps) 
            or (n_samples, n_timesteps)
        y_train_raw (np.ndarray): Raw training labels of shape (n_samples,)
        title (str): Title for the entire figure
    """
    strategy = DataCentricStrategy.from_config(strategy_config)
    X_preprocessed, y_preprocessed = strategy.apply(X_train_raw, y_train_raw)
    
    class_labels = np.unique(y_train_raw)
    n_classes = len(class_labels)
    
    n_cols = min(5, n_classes)
    n_rows = 2  
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 4), sharex=True, sharey=True)
    
    if n_cols == 1 and n_rows > 1:
        axs = axs.reshape(n_rows, 1)
    elif n_rows == 1 and n_cols > 1:
        axs = axs.reshape(1, -1)
    elif n_cols == 1 and n_rows == 1:
        axs = np.array([[axs]])
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    raw_max_len = X_train_raw.shape[-1]  
    preprocessed_max_len = X_preprocessed.shape[-1]
    x_max = max(raw_max_len, preprocessed_max_len)
    
    for col_idx, cls in enumerate(class_labels[:n_cols]):
        ax = axs[0, col_idx]
        class_indices = np.where(y_train_raw == cls)[0][:5]
        samples = X_train_raw[class_indices]
        
        if samples.ndim > 2:
            samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.reshape(samples.shape[0], -1)
        
        for i, sample in enumerate(samples):
            gt_label = y_train_raw[class_indices[i]]
            color = colors[int(gt_label) % len(colors)]
            ax.plot(sample, color=color)
        
        ax.set_title(f"Class {cls}")
        if col_idx == 0:
            ax.set_ylabel("Raw", fontweight='bold')
        ax.tick_params(labelbottom=False)
    
    for col_idx, cls in enumerate(class_labels[:n_cols]):
        ax = axs[1, col_idx]
        class_indices = np.where(y_preprocessed == cls)[0][:5]
        samples = X_preprocessed[class_indices]
        
        if samples.ndim > 2:
            samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.reshape(samples.shape[0], -1)
        
        for i, sample in enumerate(samples):
            gt_label = y_train_raw[class_indices[i]]
            color = colors[int(gt_label) % len(colors)]
            ax.plot(sample, color=color)
        
        if col_idx == 0:
            ax.set_ylabel("Preprocessed", fontweight='bold')

    for col_idx in range(n_classes, n_cols):
        axs[0, col_idx].set_visible(False)
        axs[1, col_idx].set_visible(False)
    
    axs[0, 0].set_xlim(0, x_max)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plot_accuracy_vs_strategy_param(
    summary_df: pd.DataFrame,
    strategy_type: str,
    strategy_mode: str,
    strategy_params_compare: str,
    classifier: str,
    dataset_names: List[str],
    prepare_save_plot: bool = False,
    **datasets_selected,
) -> Optional[plt.Figure]:
    """Plot accuracy vs. strategy parameter across datasets and classifiers.
    
    Creates line plots with error bars showing how accuracy varies with a strategy
    parameter (e.g., degradation intensity 'p') for a given classifier across
    multiple datasets.
    
    Args:
        summary_df (pd.DataFrame): Full results dataframe from ResultHandler
        strategy_type (str): Strategy type (e.g., 'feature_quality')
        strategy_mode (str): Strategy mode (e.g., 'gaussian')
        strategy_params_compare (str): Parameter column to plot on x-axis
        classifier (str): Classifier to visualize
        dataset_names (List[str]): Available dataset names
        prepare_save_plot (bool): If True, return figure; if False, show and return None
        **datasets_selected: Dataset name → bool mapping for filtering
    
    Returns:
        Optional[plt.Figure]: Figure object if prepare_save_plot=True, else None
    """
    subset_results = get_strategy_subset(summary_df, strategy_type, strategy_mode)
    
    fig, ax = plt.subplots()
    flip_ratios = sorted([x for x in subset_results[strategy_params_compare].unique() if x is not None])
    
    if not flip_ratios:
        print(f"No data available for {strategy_type} - {strategy_mode}")
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        if prepare_save_plot:
            return fig
        else:
            plt.show()
            return None

    for dataset in dataset_names:
        if not datasets_selected.get(dataset, True):
            continue
        ds = subset_results[
            (subset_results["dataset"] == dataset)
            & (subset_results["classifier"] == classifier)
        ]
        if ds.empty:
            continue

        means, stds, valid_strategy_params = [], [], []
        for sp in flip_ratios:
            sp_ds = ds[ds[strategy_params_compare] == sp]
            if sp_ds.empty:
                continue
            grouped = sp_ds.groupby("random_seed")["accuracy"].mean()
            means.append(grouped.mean())
            stds.append(grouped.std())
            valid_strategy_params.append(sp)

        if valid_strategy_params:
            ax.errorbar(
                valid_strategy_params,
                means,
                yerr=stds,
                fmt="-o",
                label=dataset,
                capsize=4,
                markersize=6,
            )

    ax.set_xlabel(f"{strategy_params_compare.capitalize()}")
    ax.set_ylabel(f"{classifier} Accuracy")
    
    ax.set_title(f"{classifier} Accuracy vs. {strategy_params_compare.capitalize()} ({strategy_type} - {strategy_mode})")
    
    ax.set_xlim(min(flip_ratios), max(flip_ratios))
    ax.set_ylim(0, 1)
    ax.legend()
    plt.show()

    if prepare_save_plot:
        return fig
    
def plot_accuracy_comparison(
    summary_df: pd.DataFrame,
    strategy_type: str,
    strategy_mode: str,
    strategy_params_compare: str,
    strategy_params_value: float,
    classifier1: str,
    classifier2: str,
    prepare_save_plot: bool = False,
    **datasets_selected,
) -> Optional[plt.Figure]:
    """Compare accuracy between two classifiers at a fixed strategy parameter.
    
    Creates scatter plots comparing the accuracy of two classifiers at a specific
    strategy parameter value, with one classifier on each axis. Points represent
    individual datasets or aggregated results.
    
    Args:
        summary_df (pd.DataFrame): Full results dataframe from ResultHandler
        strategy_type (str): Strategy type (e.g., 'feature_quality')
        strategy_mode (str): Strategy mode (e.g., 'gaussian')
        strategy_params_compare (str): Parameter to filter at a fixed value
        strategy_params_value (float): Fixed value of the parameter
        classifier1 (str): First classifier (x-axis)
        classifier2 (str): Second classifier (y-axis)
        prepare_save_plot (bool): If True, return figure; if False, show and return None
        **datasets_selected: Dataset name → bool mapping for filtering
    
    Returns:
        Optional[plt.Figure]: Figure object if prepare_save_plot=True, else None
    """
    subset_results = get_strategy_subset(summary_df, strategy_type, strategy_mode)

    subset = subset_results[
        subset_results[strategy_params_compare] == strategy_params_value
    ]
    if subset.empty:
        print(f"No results found for {strategy_params_compare}={strategy_params_value} in {strategy_type} - {strategy_mode}.")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        if prepare_save_plot:
            return fig
        else:
            plt.show()
            return None

    subset = subset[subset["classifier"].isin([classifier1, classifier2])]
    if subset.empty or not all(
        c in subset["classifier"].unique() for c in [classifier1, classifier2]
    ):
        print(
            f"Not enough results for classifiers {classifier1} and {classifier2} at {strategy_params_compare}={strategy_params_value}."
        )
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    for dataset in subset["dataset"].unique():
        if not datasets_selected.get(dataset, True):
            continue
        ds = subset[subset["dataset"] == dataset]
        if ds.empty:
            continue

        grouped = ds.groupby(["random_seed", "classifier"])["accuracy"].mean().unstack()
        if classifier1 not in grouped or classifier2 not in grouped:
            continue
        x = grouped[classifier1]
        y = grouped[classifier2]

        if len(grouped) > 1:
            plt.errorbar(
                x.mean(),
                y.mean(),
                xerr=x.std(),
                yerr=y.std(),
                fmt="o",
                label=dataset,
                capsize=5,
                markersize=8,
            )
        else:
            ax.scatter(x, y, label=dataset, s=50)

    ax.set_xlabel(f"{classifier1} Accuracy")
    ax.set_ylabel(f"{classifier2} Accuracy")
    
    ax.set_title(f"Accuracy Comparison: {classifier1} vs {classifier2}\n({strategy_type} - {strategy_mode}, {strategy_params_compare}={strategy_params_value})")
    
    ax.axline((0, 0), slope=1, color="grey", linestyle="--", alpha=0.5, label="")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.show()

    if prepare_save_plot:
        return fig

def plot_accuracy_trajectory(
    summary_df: pd.DataFrame,
    strategy_type: str,
    strategy_mode: str,
    strategy_params_compare: str,
    classifier1: str,
    classifier2: str,
    dataset_names: List[str],
    prepare_save_plot: bool = False,
    **datasets_selected,
) -> Optional[plt.Figure]:
    """Plot accuracy trajectories of two classifiers across strategy parameters.
    
    Creates trajectory plots showing how the accuracy relationship between two
    classifiers changes as a strategy parameter varies. Useful for understanding
    how classifiers respond differently to increasing degradation intensity.
    
    Args:
        summary_df (pd.DataFrame): Full results dataframe from ResultHandler
        strategy_type (str): Strategy type (e.g., 'feature_quality')
        strategy_mode (str): Strategy mode (e.g., 'gaussian')
        strategy_params_compare (str): Parameter to sweep along trajectory
        classifier1 (str): First classifier (x-axis)
        classifier2 (str): Second classifier (y-axis)
        dataset_names (List[str]): Available dataset names
        prepare_save_plot (bool): If True, return figure; if False, show and return None
        **datasets_selected: Dataset name → bool mapping for filtering
    
    Returns:
        Optional[plt.Figure]: Figure object if prepare_save_plot=True, else None
    """
    subset_results = get_strategy_subset(summary_df, strategy_type, strategy_mode)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    strategy_params_available = sorted([x for x in subset_results[strategy_params_compare].unique() if x is not None])
    
    if not strategy_params_available:
        print(f"No data available for {strategy_type} - {strategy_mode}")
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        if prepare_save_plot:
            return fig
        else:
            plt.show()
            return None

    for dataset in dataset_names:
        if not datasets_selected.get(dataset, True):
            continue
        ds = subset_results[
            (subset_results["dataset"] == dataset)
            & (
                subset_results["classifier"].isin(
                    [classifier1, classifier2]
                )
            )
        ]
        if ds.empty:
            continue

        x_means, y_means, x_stds, y_stds = [], [], [], []
        valid_strategy_params = []

        for fr in strategy_params_available:
            fr_ds = ds[ds[strategy_params_compare] == fr]
            grouped = (
                fr_ds.groupby(["random_seed", "classifier"])["accuracy"]
                .mean()
                .unstack()
            )
            if (
                grouped is None
                or classifier1 not in grouped
                or classifier2 not in grouped
            ):
                continue
            x = grouped[classifier1]
            y = grouped[classifier2]
            if len(x) == 0 or len(y) == 0:
                continue
            x_means.append(x.mean())
            y_means.append(y.mean())
            x_stds.append(x.std())
            y_stds.append(y.std())
            valid_strategy_params.append(fr)

        if valid_strategy_params:
            plt.errorbar(
                x_means,
                y_means,
                xerr=x_stds,
                yerr=y_stds,
                fmt="-o",
                label=dataset,
                capsize=3,
                markersize=5,
            )

    ax.set_xlabel(f"{classifier1} Accuracy")
    ax.set_ylabel(f"{classifier2} Accuracy")
    
    title_base = f"Trajectory: {classifier1} vs {classifier2}\n({strategy_type} - {strategy_mode})"
    
    if strategy_params_available:
        ax.set_title(
            f"{title_base}\n{strategy_params_compare} from {min(strategy_params_available):.2f} to {max(strategy_params_available):.2f}"
        )
    else:
        ax.set_title(title_base)
    ax.axline((0, 0), slope=1, color="grey", linestyle="--", alpha=0.5, label="")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.show()

    if prepare_save_plot:
        return fig

def create_dependent_dropdowns(summary_df: pd.DataFrame):
    """Create dependent type and mode dropdowns for interactive widgets.
    
    Returns dropdowns where mode options update based on selected type.
    
    Args:
        summary_df (pd.DataFrame): Full results dataframe
    
    Returns:
        Tuple of (type_dropdown, mode_dropdown)
    """
    import ipywidgets as widgets
    
    available_types = sorted(summary_df['type'].unique())
    available_modes_by_type = {
        t: sorted(summary_df[summary_df['type'] == t]['mode'].unique())
        for t in available_types
    }
    
    type_dd = widgets.Dropdown(
        options=available_types,
        value=available_types[0] if len(available_types) > 0 else None,
        description='Type',
    )
    
    mode_dd = widgets.Dropdown(
        options=available_modes_by_type[available_types[0]] if len(available_types) > 0 else [],
        description='Mode',
    )
    
    def on_type_change(change):
        mode_dd.options = available_modes_by_type[change['new']]
        if mode_dd.options:
            mode_dd.value = mode_dd.options[0]
    
    type_dd.observe(on_type_change, names='value')
    
    return type_dd, mode_dd

def batch_export_accuracy_vs_param(
    summary_df: pd.DataFrame,
    strategy_type: str,
    strategy_mode: str,
    strategy_params_compare: str,
    dataset_names: List[str],
    output_path: str,
    dataset_filter: Dict[str, bool] = None,
) -> None:
    """Batch export accuracy vs parameter plots for all classifiers.
    
    Args:
        summary_df (pd.DataFrame): Full results dataframe
        strategy_type (str): Strategy type to export
        strategy_mode (str): Strategy mode to export
        strategy_params_compare (str): Parameter to compare
        dataset_names (List[str]): Available datasets
        output_path (str): Directory to save PDFs
        dataset_filter (Dict[str, bool]): Filter for which datasets to include
    """
    if dataset_filter is None:
        dataset_filter = {name: True for name in dataset_names}
    
    os.makedirs(output_path, exist_ok=True)
    
    for classifier in sorted(summary_df["classifier"].unique()):
        fig = plot_accuracy_vs_strategy_param(
            summary_df,
            strategy_type=strategy_type,
            strategy_mode=strategy_mode,
            strategy_params_compare=strategy_params_compare,
            classifier=classifier,
            dataset_names=dataset_names,
            prepare_save_plot=True,
            **dataset_filter,
        )
        
        if fig is not None:
            fig.savefig(
                os.path.join(output_path, f"accuracy_vs_{strategy_params_compare}_{classifier}.pdf"),
                dpi=300,
                bbox_inches='tight',
            )
            plt.close(fig)

def batch_export_comparison(
    summary_df: pd.DataFrame,
    strategy_type: str,
    strategy_mode: str,
    strategy_params_compare: str,
    dataset_names: List[str],
    output_path: str,
) -> None:
    """Batch export classifier comparison plots for all parameter values.
    
    Args:
        summary_df (pd.DataFrame): Full results dataframe
        strategy_type (str): Strategy type to export
        strategy_mode (str): Strategy mode to export
        strategy_params_compare (str): Parameter to sweep
        dataset_names (List[str]): Available datasets
        output_path (str): Directory to save PDFs
    """
    os.makedirs(output_path, exist_ok=True)
    
    classifiers = sorted(summary_df["classifier"].unique())
    param_values = sorted([x for x in summary_df[strategy_params_compare].unique() if x is not None])
    
    for classifier1, classifier2 in zip(classifiers[:-1], classifiers[1:]):
        if classifier1 == classifier2:
            continue
        
        for param_value in param_values:
            fig = plot_accuracy_comparison(
                summary_df,
                strategy_type=strategy_type,
                strategy_mode=strategy_mode,
                strategy_params_compare=strategy_params_compare,
                strategy_params_value=param_value,
                classifier1=classifier1,
                classifier2=classifier2,
                dataset_names=dataset_names,
                prepare_save_plot=True,
                **{name: True for name in dataset_names},
            )
            
            if fig is not None:
                fig.savefig(
                    os.path.join(output_path, f"comparison_{classifier1}_vs_{classifier2}_{strategy_params_compare}_{param_value:.2f}.pdf"),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close(fig)

def batch_export_trajectory(
    summary_df: pd.DataFrame,
    strategy_type: str,
    strategy_mode: str,
    strategy_params_compare: str,
    dataset_names: List[str],
    output_path: str,
) -> None:
    """Batch export trajectory plots for all classifier pairs and datasets.
    
    Args:
        summary_df (pd.DataFrame): Full results dataframe
        strategy_type (str): Strategy type to export
        strategy_mode (str): Strategy mode to export
        strategy_params_compare (str): Parameter to sweep
        dataset_names (List[str]): Available datasets
        output_path (str): Directory to save PDFs
    """
    os.makedirs(output_path, exist_ok=True)
    
    classifiers = sorted(summary_df["classifier"].unique())
    
    for classifier1, classifier2 in zip(classifiers[:-1], classifiers[1:]):
        if classifier1 == classifier2:
            continue
        
        for data_set in dataset_names:
            fig = plot_accuracy_trajectory(
                summary_df,
                strategy_type=strategy_type,
                strategy_mode=strategy_mode,
                strategy_params_compare=strategy_params_compare,
                classifier1=classifier1,
                classifier2=classifier2,
                dataset_names=dataset_names,
                prepare_save_plot=True,
                **{name: False if name != data_set else True for name in dataset_names},
            )
            
            if fig is not None:
                fig.savefig(
                    os.path.join(output_path, f"trajectory_{classifier1}_vs_{classifier2}_{data_set}.pdf"),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close(fig)
