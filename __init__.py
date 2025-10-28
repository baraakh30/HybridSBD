"""
Video Shot Detection System
A multimodal approach for robust video shot boundary detection using 
multiple feature representations and fusion strategies.
"""

__version__ = "1.0.0"
__author__ = "Graduate Research Team"
__email__ = "research@grad.edu"

from .core.config import VideoShotConfig
from .core.coordinator import ProcessingCoordinator
from .extractors import *
from .fusion import *
from .detection import *
from .evaluation import *

__all__ = [
    "VideoShotConfig",
    "ProcessingCoordinator",
]