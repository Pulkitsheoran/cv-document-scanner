"""ScanVision - Smart Document Scanner & OCR.

A fully offline document scanning and OCR pipeline built with classical
computer vision (OpenCV) and Tesseract OCR.

Pipeline stages:
    ingest -> preprocess -> detect -> rectifity -> enhance -> OCR -> export
"""

from scanvision.config import CONFIG, ScanConfig
from scanvision.models import ScanOptions, ScanResult
from scanvision.pipeline import ScanPipeline

__version__ = "1.0.0"
__all__ = ["CONFIG", "ScanConfig", "ScanOptions", "ScanResult", "ScanPipeline", "__version__"]