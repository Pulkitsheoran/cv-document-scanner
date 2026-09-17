"""Photometric pre-processing prepared for document-boundary detection."""

from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """Transforms a raw BGR photo into artefacts useful for detection.

    Stages used before contour detection:
      1. gray-scale conversion,
      2. bilateral denoising (edge preserving),
      3. illumination compensation by division with a large blurred copy
         (removes shadows/light gradients),
      4. downscaling to ``max_dim`` for faster contour search,
      5. automatic Canny edge map with median-based thresholds.
    """

    def __init__(self) -> None:
        self._cache: dict = {}

    def to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR colour image to grayscale (no-op for 1-channel input)."""
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def denoise(self, gray: np.ndarray, diameter: int = 5) -> np.ndarray:
        """Edge-preserving bilateral denoise to suppress sensor noise."""
        return cv2.bilateralFilter(gray, diameter, sigmaColor=60, sigmaSpace=60)

    def compensate_illumination(self, gray: np.ndarray, kernel: int = 51) -> np.ndarray:
        """Flat-field illumination normalisation via division by blur.

        A document photo often carries a light gradient. Dividing the image
        by a strongly blurred copy removes this gradient while preserving
        local contrast.
        """
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        compensated = cv2.divide(gray, blurred, scale=255.0)
        return compensated.astype(np.uint8)

    def fit_scale(self, image: np.ndarray, max_dim: int = 1600) -> tuple[np.ndarray, float]:
        """Downscale so the longer side equals ``max_dim``; return (image, scale_inv).

        ``scale_inv`` maps co-ordinates back to the original resolution.
        """
        height, width = image.shape[:2]
        longer = max(height, width)
        if longer <= max_dim:
            return image.copy(), 1.0
        scale = max_dim / float(longer)
        resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        return resized, 1.0 / scale

    def edge_map(self, gray: np.ndarray, dilate: bool = True) -> np.ndarray:
        """Canny edge map with thresholds derived automatically from the median.

        ``dilate`` bridges small gaps so quadrilaterals close into one contour.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = float(np.median(blurred))
        low = max(10, int(0.66 * median))
        high = min(255, int(1.33 * median))
        edges = cv2.Canny(blurred, low, high)
        if dilate:
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        return edges

    def prepare_for_detection(self, image: np.ndarray, max_dim: int = 1600) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (gray_small, edges, scale_inv) ready for contour detection."""
        gray = self.to_gray(image)
        gray = self.compensate_illumination(gray)
        gray_small, scale_inv = self.fit_scale(gray, max_dim=max_dim)
        edges = self.edge_map(gray_small)
        self._cache = {"gray_small": gray_small}
        return gray_small, edges, scale_inv