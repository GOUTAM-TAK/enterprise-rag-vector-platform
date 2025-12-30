import logging
import logging.handlers
import queue
import sys
from pathlib import Path
from typing import Optional

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
INFO_LOG_FILE = LOG_DIR / "info.log"
ERROR_LOG_FILE = LOG_DIR / "error.log"

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Async Queue ----------
_log_queue: Optional[queue.Queue] = None
_listener: Optional[logging.handlers.QueueListener] = None


def _setup_async_logging():
    global _log_queue, _listener

    if _log_queue is not None:
        return

    _log_queue = queue.Queue(-1)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # -------- INFO & WARNING HANDLER --------
    info_handler = logging.handlers.RotatingFileHandler(
        INFO_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    info_handler.addFilter(lambda record: record.levelno < logging.ERROR)

    # -------- ERROR & EXCEPTION HANDLER --------
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # -------- Console (Optional but Enterprise-Useful) --------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # -------- Queue Listener (Background Thread) --------
    _listener = logging.handlers.QueueListener(
        _log_queue,
        info_handler,
        error_handler,
        console_handler,
        respect_handler_level=True,
    )
    _listener.start()


def get_logger(name: str) -> logging.Logger:
    """
    Enterprise async-safe logger.
    - INFO / WARNING → logs/info.log
    - ERROR / EXCEPTION → logs/error.log
    """
    _setup_async_logging()

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        queue_handler = logging.handlers.QueueHandler(_log_queue)
        logger.addHandler(queue_handler)

        # Prevent duplicate logs
        logger.propagate = False

    return logger
