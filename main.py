import os 
from src.utils import load_and_expand_yaml, logger, RESULTS_DIR
from src.experiment import Experiment

if __name__ == "__main__":
    config_path = "experiment.yaml"
    base_path = "112UCRFolds"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    configs = load_and_expand_yaml(config_path)

    for config in configs:
        experiment = Experiment(config, base_path=base_path, results_root=RESULTS_DIR)
        experiment.run()

    logger.info("All experiments completed")