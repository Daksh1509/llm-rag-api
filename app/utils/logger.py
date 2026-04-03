import logging
import sys
from pythonjsonlogger import jsonlogger
from app.utils.config import get_settings

settings = get_settings()


def setup_logger(name: str) -> logging.Logger:
    """
    Returns a JSON-structured logger.
    JSON logs are machine-parseable by tools like Datadog, CloudWatch, Grafana.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Stream handler → stdout (captured by Docker/cloud logging)
    handler = logging.StreamHandler(sys.stdout)

    # JSON formatter — each log line is valid JSON
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate logs)
    logger.propagate = False

    return logger


# Module-level logger for quick import across files
log = setup_logger("rag_api")