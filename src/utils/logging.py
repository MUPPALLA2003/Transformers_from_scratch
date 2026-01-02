import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # IMPORTANT
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
