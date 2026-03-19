import argparse
from src.utils.configHandler import ConfigHandler
from src.experiments.experiment import Experiment
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data-Centric AI Experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to the experiment configuration file"
    )
    parser.add_argument(
        "--datasets-path",
        type=str,
        default="Univariate_ts",
        help="Path to datasets directory"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for experiment results"
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="summary.csv",
        help="Filename for results summary"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun of all experiments, bypassing deduplication check"
    )
    args = parser.parse_args()

    _config_handler = ConfigHandler(
        datasets_path=args.datasets_path,
        results_dir=args.results_dir,
        summary_file=args.summary_file,
        config_path=args.config,
    )
    
    _config_handler.logger.info(f"Using config file: {args.config}")
    configs = _config_handler.load_and_expand_yaml(args.config)

    _config_handler.ensure_datasets_exist()

    with logging_redirect_tqdm():
        for config in tqdm(configs, desc="Experiments", unit="config"):
            try:
                experiment = Experiment(
                    config,
                    base_path=_config_handler.DATASETS_PATH,
                    results_root=_config_handler.RESULTS_DIR,
                    summary_file=_config_handler.SUMMARY_FILE,
                    force_rerun=args.force_rerun,
                )
                experiment.run()
            except Exception as e:
                _config_handler.logger.error(f"Experiment failed with config {config}: {e}", exc_info=True)
                continue
        _config_handler.logger.info("All experiments completed")
