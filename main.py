from src.utils import CONFIG_PATH, DATASETS_PATH, ensure_datasets_exist, load_and_expand_yaml, logger, RESULTS_DIR
from src.experiment import Experiment
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


if __name__ == "__main__":
    ensure_datasets_exist()
    configs = load_and_expand_yaml(CONFIG_PATH)

    with logging_redirect_tqdm():
        for config in tqdm(configs, desc="Experiments", unit="config"):
            experiment = Experiment(config, base_path=DATASETS_PATH, results_root=RESULTS_DIR)
            experiment.run()
        logger.info("All experiments completed")