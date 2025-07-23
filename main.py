from src.utils.configHandler import ConfigHandler
from src.experiments.experiment import Experiment
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


if __name__ == "__main__":

    _config_handler = ConfigHandler()

    _config_handler.ensure_datasets_exist()
    configs = _config_handler.load_and_expand_yaml(_config_handler.CONFIG_PATH)

    with logging_redirect_tqdm():
        for config in tqdm(configs, desc="Experiments", unit="config"):
            experiment = Experiment(
                config,
                base_path=_config_handler.DATASETS_PATH,
                results_root=_config_handler.RESULTS_DIR,
                summary_file=_config_handler.SUMMARY_FILE,
            )
            experiment.run()
        _config_handler.logger.info("All experiments completed")
