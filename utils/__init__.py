"""
Utility modules initialization.
"""

from .video_utils import VideoProcessor
from .logger import setup_logger, get_logger

__all__ = [
    "VideoProcessor",
    "setup_logger",
    "get_logger",
]