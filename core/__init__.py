"""
Core module initialization.
"""

from .config import VideoShotConfig, DEFAULT_RESEARCH_CONFIG, DEFAULT_PRODUCTION_CONFIG, DEFAULT_EDGE_CONFIG
from .coordinator import ProcessingCoordinator

__all__ = [
    "VideoShotConfig",
    "DEFAULT_RESEARCH_CONFIG", 
    "DEFAULT_PRODUCTION_CONFIG",
    "DEFAULT_EDGE_CONFIG",
    "ProcessingCoordinator",
]