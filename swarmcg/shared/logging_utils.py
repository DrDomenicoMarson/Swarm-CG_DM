import logging
import os
import sys

LOG_NAME = "swarmcg"

_LEVEL_COLORS = {
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[31m",
}
_COLOR_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, use_color=False):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        message = super().format(record)
        if self.use_color:
            color = _LEVEL_COLORS.get(record.levelno)
            if color:
                return f"{color}{message}{_COLOR_RESET}"
        return message


def get_logger(name=None):
    logger_name = LOG_NAME if name is None else f"{LOG_NAME}.{name}"
    return logging.getLogger(logger_name)


def setup_logging(module_name=None, log_dir=None, verbose=False):
    logger = get_logger()
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            _ColorFormatter(fmt="%(message)s", use_color=sys.stdout.isatty())
        )
        logger.addHandler(console_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    if log_dir:
        log_name = f"swarmcg-{module_name}.log" if module_name else "swarmcg.log"
        log_path = os.path.join(log_dir, log_name)
        if not any(
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(getattr(handler, "baseFilename", "")) == os.path.abspath(log_path)
            for handler in logger.handlers
        ):
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)

    return logger
