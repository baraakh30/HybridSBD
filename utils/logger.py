"""
Logging utilities.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str, log_dir: str, level: int = logging.INFO, 
                enable_logging: bool = True, file_logging: bool = True, 
                console_logging: bool = True, quiet_mode: bool = False) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        enable_logging: Whether to enable logging at all
        file_logging: Whether to enable file logging
        console_logging: Whether to enable console logging
        quiet_mode: If True, suppress all console output
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # If logging is disabled, set to CRITICAL level to suppress most messages
    if not enable_logging or quiet_mode:
        logger.setLevel(logging.CRITICAL)
        # Remove all existing handlers to ensure no output
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        return logger
    
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create log directory if file logging is enabled
    if file_logging:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    
    # File handler
    if file_logging:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_logging and not quiet_mode:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)