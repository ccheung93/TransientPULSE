"""
Logging utilities for the multimessenger spectrum analysis.

Provides configurable logging to both console and file with adjustable verbosity.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_file='propagation.log', level='INFO', console_level=None, file_level=None):
    """
    Configure logging for the analysis.

    Args:
        log_file (str): Path to log file
        level (str): Default logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        console_level (str, optional): Override level for console output
        file_level (str, optional): Override level for file output

    Returns:
        logging.Logger: Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger('multimessenger')

    # Clear any existing handlers
    logger.handlers.clear()

    # Set overall level to DEBUG so handlers can filter
    logger.setLevel(logging.DEBUG)

    # Determine levels
    default_level = getattr(logging, level.upper())
    console_log_level = getattr(logging, console_level.upper()) if console_level else default_level
    file_log_level = getattr(logging, file_level.upper()) if file_level else default_level

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger():
    """
    Get the application logger.

    Returns:
        logging.Logger: The multimessenger logger instance
    """
    return logging.getLogger('multimessenger')


# Default logger instance (can be reconfigured with setup_logging)
logger = setup_logging()
