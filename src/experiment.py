import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from src.dataCentricStrategy import DataCentricStrategy
from src.utils import logger, SUMMARY_FILE
from src.datasetHandler import UCRDataset
from src.classifierHandler import BakeoffClassifier


### Experiment ###
class Experiment:
    def __init__(self, config: Dict[str, Any], base_path: str, results_root: str):
        self.config = config
        self.base_path = base_path
        self.random_seed = config.get("random_seed", 0)

        ds_name = config["dataset"]["name"]
        clf_name = config["classifier"]["name"]
        strategy_conf = config["strategy"]

        logger.info(
            f"Initializing Experiment with dataset: {ds_name}, classifier: {clf_name}, strategy: {strategy_conf}"
        )

        self.dataset = UCRDataset(ds_name, path=base_path)
        self.classifier = BakeoffClassifier(clf_name, random_state=self.random_seed)
        self.strategy = DataCentricStrategy.from_config(strategy_conf)

        # Check for duplicates
        if os.path.exists(SUMMARY_FILE):
            summary_df = pd.read_csv(SUMMARY_FILE)
            match = (
                (summary_df["dataset"] == ds_name)
                & (summary_df["classifier"] == clf_name)
                & (summary_df["strategy"] == strategy_conf["type"])
                & (summary_df["strategy_mode"] == strategy_conf.get("mode"))
                & (summary_df["strategy_params"] == json.dumps(strategy_conf["params"]))
                & (summary_df["random_seed"] == self.random_seed)
            )
            if match.any():
                logger.info("Skipping already executed configuration.")
                self.skip = True
                return

        self.skip = False
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(results_root, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

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
        self.X_test, self.y_test = self.strategy.apply(self.X_test_raw, self.y_test_raw)

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
        if os.path.exists(SUMMARY_FILE):
            df_summary.to_csv(SUMMARY_FILE, mode="a", header=False, index=False)
        else:
            df_summary.to_csv(SUMMARY_FILE, index=False)

        logger.info(
            f"Experiment finished with accuracy: {accuracy:.4f}, f1_score: {f1:.4f}"
        )
