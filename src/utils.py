import os
import logging
from itertools import product
import yaml
import urllib.request
import zipfile

RESULTS_DIR = "results"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.csv")
DATA_URL = "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip"
DATA_ZIP = "Univariate2018_ts.zip"
CONFIG_PATH = "experiment.yaml"
DATASETS_PATH = "Univariate_ts"

os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ensure_datasets_exist():
    if not os.path.exists(DATASETS_PATH):
        logger.info(f"{DATASETS_PATH} not found. Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(DATA_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(DATA_ZIP)
        logger.info("Dataset ready.")
    else:
        logger.info(f"{DATASETS_PATH} already exists. No need to download. \n")


def load_and_expand_yaml(path: str):
    logger.info(f"Loading and expanding YAML configuration from: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

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

    logger.info(f"YAML configuration expanded into {len(configs)} configurations")
    return configs
