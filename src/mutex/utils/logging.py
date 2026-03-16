from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path


def setup_logger(
    name: str = __name__,
    level: int = INFO,
    log_file: Path | None = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> Logger:
    logger = getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = Formatter(fmt)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = __name__) -> Logger:
    return getLogger(name)
