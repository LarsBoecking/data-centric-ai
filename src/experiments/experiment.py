import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from ..data_centric.dataCentricStrategy import DataCentricStrategy
from ..data_handling.datasetHandler import UCRDataset
from ..models.classifierHandler import BakeoffClassifier
from ..utils.logger import get_logger


class Experiment:
    """
    Orchestrates data-centric AI experiments for time series classification.

    This class manages the complete experimental pipeline, including dataset loading,
    data-centric strategy application, model training, evaluation, and result persistence.
    It automatically handles experiment deduplication by checking against previous results
    and provides comprehensive logging throughout the process.

    The class integrates three main components:
    - UCRDataset for time series data loading
    - DataCentricStrategy for data modification
    - BakeoffClassifier for model training and prediction

    Args:
        config (Dict[str, Any]): Experiment configuration dictionary containing:
            - dataset (Dict): Dataset configuration with 'name' key
            - classifier (Dict): Classifier configuration with 'name' key  
            - strategy (Dict): Strategy configuration with 'type', 'mode', and 'params'
            - random_seed (int, optional): Random seed for reproducibility
        base_path (str): Base directory path where UCR datasets are stored
        results_root (str): Root directory for storing experiment results
        summary_file (str, optional): Path to CSV file for experiment summaries. 
            Defaults to "summary.csv"

    Methods:
        run() -> None:
            Executes the complete experiment pipeline including training,
            evaluation, and result persistence.

        _check_existing_results() -> bool:
            Checks if experiment with identical configuration already exists.

        _save_results(accuracy: float, f1: float, preds: np.ndarray) -> None:
            Saves experiment results to files and updates summary CSV.

    Raises:
        FileNotFoundError: If dataset files are not found at the specified path.
        ValueError: If configuration parameters are invalid.

    Example:
        ```python
        config = {
            "dataset": {"name": "Beef"},
            "classifier": {"name": "mini-rocket"},
            "strategy": {
                "type": "label_flipping",
                "mode": "random",
                "params": {"flip_ratio": 0.1}
            },
            "random_seed": 42
        }
        
        experiment = Experiment(
            config=config,
            base_path="./datasets",
            results_root="./results"
        )
        experiment.run()
        ```
    """
    def __init__(self, config: Dict[str, Any], base_path: str, results_root: str, summary_file: str = "summary.csv"):
        """
        Initialize the Experiment with configuration and paths.

        Sets up the experiment by loading dataset, initializing classifier and strategy,
        checking for existing results, and creating output directory structure.

        Args:
            config (Dict[str, Any]): Complete experiment configuration
            base_path (str): Path to UCR dataset directory
            results_root (str): Root directory for experiment outputs  
            summary_file (str, optional): Summary CSV filename. Defaults to "summary.csv"

        Raises:
            FileNotFoundError: If dataset cannot be found at base_path
            ValueError: If configuration contains invalid parameters
        """
        self.config = config
        self.base_path = base_path
        self.results_root = results_root
        self.summary_file = summary_file
        self.random_seed = config.get("random_seed", 0)
        self.logger = get_logger(__name__)

        ds_name = config["dataset"]["name"]
        clf_name = config["classifier"]["name"]
        strategy_conf = config["strategy"]

        self.logger.info(
            f"Initializing Experiment with dataset: {ds_name}, classifier: {clf_name}, strategy: {strategy_conf}"
        )

        self.dataset = UCRDataset(ds_name, path=base_path)
        self.classifier = BakeoffClassifier(clf_name, random_state=self.random_seed)
        self.strategy = DataCentricStrategy.from_config(strategy_conf)

        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            match = (
                (summary_df["dataset"] == ds_name)
                & (summary_df["classifier"] == clf_name)
                & (summary_df["random_seed"] == self.random_seed)
                & (summary_df["strategy"] == strategy_conf["type"])
                & (summary_df["strategy_mode"] == strategy_conf.get("mode"))
                & (summary_df["strategy_params"] == json.dumps(strategy_conf["params"]))
            )
            if match.any():
                self.logger.info("Skipping already executed configuration. \n")
                self.skip = True
                return

        self.skip = False
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(results_root, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info(f"Output directory: {self.output_dir}")

        (
            self.X_train_raw,
            self.y_train_raw,
            self.X_test_raw,
            self.y_test_raw,
            self.meta,
        ) = self.dataset.load()
        self.X_train, self.y_train = self.strategy.apply(
            self.X_train_raw, self.y_train_raw
        )
        
        #  Keep test data unchanged
        self.X_test, self.y_test = self.X_test_raw, self.y_test_raw  

    def run(self):
        if self.skip:
            return

        self.classifier.fit(self.X_train, self.y_train)
        preds = self.classifier.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average="weighted")

        np.save(os.path.join(self.output_dir, "y_test.npy"), self.y_test)
        np.save(os.path.join(self.output_dir, "preds.npy"), preds)
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

        metrics = {"accuracy": accuracy, "f1_score": f1}
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        summary_row = {
            "dataset": self.config["dataset"]["name"],
            "classifier": self.config["classifier"]["name"],
            "strategy": self.config["strategy"]["type"],
            "strategy_mode": self.config["strategy"].get("mode"),
            "strategy_params": json.dumps(self.config["strategy"]["params"]),
            "random_seed": self.random_seed,
            "accuracy": accuracy,
            "f1_score": f1,
            "folder": self.output_dir,
        }

        df_summary = pd.DataFrame([summary_row])
        if os.path.exists(self.summary_file):
            df_summary.to_csv(self.summary_file, mode="a", header=False, index=False)
        else:
            df_summary.to_csv(self.summary_file, index=False)

        self.logger.info(
            f"Experiment finished with accuracy: {accuracy:.4f}, f1_score: {f1:.4f} \n"
        )
