"""Core data models shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ScanOptions:
    """Runtime options that steer the scanning pipeline for one image.

    ``enhance_mode`` selects the post-rectification enhancement:
      - "gray":     lighting-normalised grayscale page (soft look)
      - "bw":       Otsu binarized page optimised for OCR (default)
      - "otsu":     plain Otsu threshold, no morphological cleanup
    """

    enhance_mode: str = "bw"
    do_ocr: bool = True
    export_pdf: bool = True
    export_text: bool = True
    deskew: bool = True
    fix_aspect: bool = True
    visualize: bool = False
    output_suffix: str = ""


@dataclass
class OCRWord:
    """A single word detected by the OCR engine with its bounding box."""

    word: str
    left: int
    top: int
    width: int
    height: int
    confidence: float


@dataclass
class ScanResult:
    """Outcome of running the pipeline on one input document photo."""

    input_path: Path
    stem: str
    ok: bool = False
    duration: float = 0.0
    stages: dict = field(default_factory=dict)
    ocr_text: Optional[str] = None
    words: list = field(default_factory=list)
    cer: Optional[float] = None
    wer: Optional[float] = None
    char_accuracy: Optional[float] = None
    word_accuracy: Optional[float] = None
    quad: Optional[np.ndarray] = None
    output_paths: list = field(default_factory=list)
    error: Optional[str] = None