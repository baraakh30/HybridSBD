"""
Feature extractors module initialization.
"""

from .base import BaseFeatureExtractor, FeatureExtractorRegistry
from .hsv_extractor import HSVHistogramExtractor
from .cnn_extractor import CNNFeatureExtractor
from .clip_extractor import CLIPFeatureExtractor
from .optical_flow_extractor import OpticalFlowExtractor

__all__ = [
    "BaseFeatureExtractor",
    "FeatureExtractorRegistry",
    "HSVHistogramExtractor",
    "CNNFeatureExtractor", 
    "CLIPFeatureExtractor",
    "OpticalFlowExtractor",
]