import logging
import os

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger with the specified name and level.
    Logs are output to both the console and a file named 'app.log'.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # File handler
        log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, 'data_centric.log'))
        fh.setLevel(level)
        fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger