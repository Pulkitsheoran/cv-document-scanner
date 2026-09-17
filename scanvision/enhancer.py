"""Post-rectification enhancement: produce a clean, OCR-ready page image."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from scanvision.exceptions import ScanVisionError

MODES = ("bw", "gray", "otsu")


class DocumentEnhancer:
    """Enhances the rectified page into its final deliverable form.

    Supported modes:
      - "gray": CLAHE contrast-normalised grayscale (natural looking),
      - "bw":   Otsu binarization + light morphological cleanup (default,
                best accuracy for OCR),
      - "otsu": plain Otsu binarization without cleanup.
    """

    LOGGER = logging.getLogger("scanvision.enhancer")

    def enhance(self, page: np.ndarray, mode: str = "bw", deskew: bool = True) -> np.ndarray:
        if mode not in MODES:
            raise ScanVisionError(f"unknown enhance mode '{mode}', expected one of {MODES}")
        gray = self._to_gray(page)
        if mode == "gray":
            result = self._clahe(gray)
        elif mode == "otsu":
            result = self._otsu(gray)
        else:
            result = self._otsu_clean(gray)
            if deskew:
                result = self._deskew(result)
        return result

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _clahe(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _otsu(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    def _otsu_clean(self, gray: np.ndarray) -> np.ndarray:
        binary = self._otsu(gray)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # remove specks
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)  # join strokes
        return binary

    def _deskew(self, binary: np.ndarray) -> np.ndarray:
        """Tiny residual rotation correction for binarized pages.

        Uses the minimum-area rectangle around all foreground (text) pixels.
        Angless outside +/-10 degrees are ignored to avoid turning a correct
        page upside down.
        """
        inverse = cv2.bitwise_not(binary)
        coords = np.column_stack(np.where(inverse > 0))
        if coords.shape[0] < 10:
            return binary
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle  # normalise to [-45, 45)
        if not -10.0 <= angle <= 10.0:
            self.LOGGER.debug("deskew skipped: angle %.2f out of range", angle)
            return binary
        height, width = binary.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            binary, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=255
        )
        self.LOGGER.debug("deskew rotated by %.2f degrees", angle)
        return rotated