import os
import logging
from itertools import product
import yaml
import urllib.request
import zipfile
from ..utils.logger import get_logger


class ConfigHandler:
    def __init__(
        self,
        datasets_path="Univariate_ts",
        results_dir="results",
        summary_file="summary.csv",
        config_path="configs/experiment.yaml",
        data_url="http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip",
        data_zip="Univariate2018_ts.zip",
    ):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ConfigHandler")
        
        # Set path configurations from parameters
        self.DATASETS_PATH = datasets_path
        self.RESULTS_DIR = results_dir
        self.SUMMARY_FILE = os.path.join(results_dir, summary_file)
        self.CONFIG_PATH = config_path
        self.DATA_URL = data_url
        self.DATA_ZIP = data_zip
        
        # Create results directory
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        self.logger.debug(f"Loaded configuration - RESULTS_DIR: {self.RESULTS_DIR}, "
                         f"DATASETS_PATH: {self.DATASETS_PATH}, CONFIG_PATH: {self.CONFIG_PATH}")


    def ensure_datasets_exist(self):
        """Download and extract datasets if they don't exist"""
        if not os.path.exists(self.DATASETS_PATH):
            self.logger.info(f"{self.DATASETS_PATH} not found. Downloading dataset...")
            try:
                urllib.request.urlretrieve(self.DATA_URL, self.DATA_ZIP)
                self.logger.info("Extracting dataset...")
                with zipfile.ZipFile(self.DATA_ZIP, "r") as zip_ref:
                    zip_ref.extractall(".")
                os.remove(self.DATA_ZIP)
                self.logger.info("Dataset ready.")
            except Exception as e:
                self.logger.error(f"Error downloading or extracting dataset: {e}")
                raise
        else:
            self.logger.info(f"{self.DATASETS_PATH} already exists. No need to download. \n")

    def load_and_expand_yaml(self, config_path=None):
        """Load and expand YAML configuration into individual experiment configs"""
        if config_path is None:
            config_path = self.CONFIG_PATH
            
        self.logger.info(f"Loading and expanding YAML configuration from: {config_path}")
        
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Experiment configuration file not found at path: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing experiment YAML file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading experiment YAML file: {e}")
            raise

        def ensure_list(val):
            return val if isinstance(val, list) else [val]

        configs = []
        for block in ensure_list(config["experiment"]):
            datasets = ensure_list(block["dataset"])
            classifiers = ensure_list(block["classifier"])
            seeds = ensure_list(block.get("random_seed", 0))

            strategy_blocks = (
                ensure_list(block["strategy"])
                if isinstance(block["strategy"], list)
                else [block["strategy"]]
            )

            for strategy in strategy_blocks:
                strategy_type = strategy["type"]
                strategy_mode = strategy.get("mode")
                strategy_params = strategy.get("params", {})

                param_combinations = (
                    [
                        dict(zip(strategy_params.keys(), v))
                        for v in product(
                            *[ensure_list(val) for val in strategy_params.values()]
                        )
                    ]
                    if strategy_params
                    else [{}]
                )

                for dataset_name, classifier_name, seed, params in product(
                    datasets, classifiers, seeds, param_combinations
                ):
                    new_conf = {
                        "dataset": {"name": dataset_name},
                        "classifier": {"name": classifier_name},
                        "strategy": {
                            "type": strategy_type,
                            "mode": strategy_mode,
                            "params": params,
                        },
                        "random_seed": seed,
                    }
                    configs.append(new_conf)

        self.logger.info(f"YAML configuration expanded into {len(configs)} configurations")
        return configs

    def get_path(self, *keys):
        """Generic getter for any path in the paths yaml, using nested keys"""
        self.logger.debug(f"Retrieving path for keys: {keys}")
        try:
            value = self.paths_config
            for key in keys:
                value = value[key]
            self.logger.debug(f"Successfully retrieved path: {value}")
            return value
        except KeyError as e:
            self.logger.error(f"Key not found in configuration: {e} for keys: {keys}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error retrieving path for keys {keys}: {e}")
            raise