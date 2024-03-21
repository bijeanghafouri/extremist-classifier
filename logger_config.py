import logging


def configure_logger(logger_name):
    logger = logging.getLogger(logger_name)

    # Check if the logger already has handlers configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

