"""Document localisation: find the four corners of the page quadrangle."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from scanvision.exceptions import NoDocumentFoundError
from scanvision.geometry import order_points, polygon_area, validate_quad
from scanvision.preprocessor import ImagePreprocessor


class DocumentDetector:
    """Detects the largest convex quadrilateral in the image and orders its corners.

    Approach: Canny edges -> morphology -> external contours ->
    polygon approximation -> select the strongest valid quadrangle.
    """

    LOGGER = logging.getLogger("scanvision.detector")

    def __init__(
        self,
        max_dim: int = 1600,
        min_area_ratio: float = 0.02,
        max_aspect: float = 4.0,
        epsilon_ratio: float = 0.025,
    ) -> None:
        self.max_dim = max_dim
        self.min_area_ratio = min_area_ratio
        self.max_aspect = max_aspect
        self.epsilon_ratio = epsilon_ratio
        self._preprocessor = ImagePreprocessor()

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the ordered page corners (TL,TR,BR,BL) and a quality score.

        The score is the ratio of the quadrangle area to the image area and
        is used for logging and diagnostics. Raises
        :class:`NoDocumentFoundError` when no plausible page is found.
        """
        gray_small, edges, scale_inv = self._preprocessor.prepare_for_detection(image, self.max_dim)
        best = self._best_quad_on(edges, gray_small)
        if best is None:
            # Fallback: locate the page on the Otsu binarized mask. This
            # helps when Canny edges are broken (very low contrast photos).
            _, mask = cv2.threshold(gray_small, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            white = cv2.countNonZero(mask)
            uniform = white >= 0.99 * mask.size or white <= 0.01 * mask.size
            if not uniform:
                best = self._best_quad_on(mask, gray_small)

        if best is None:
            raise NoDocumentFoundError(
                "no document quadrangle detected "
                "(try a higher-resolution photo with the page fully in frame)"
            )

        score, quad = best
        quad = quad * scale_inv
        self.LOGGER.info("document detected with coverage score %.2f", score)
        return quad.astype(np.float32), score

    def _best_quad_on(
        self, edges: np.ndarray, gray_small: np.ndarray, max_area_fraction: Optional[float] = None
    ) -> Optional[tuple[float, np.ndarray]]:
        """Search external contours on a binary/edge image for the best quadrangle.

        Each large contour is first convex-hulled so local boundary notches
        can never destroy the polygon approximation, then simplified to four
        points via ``approxPolyDP``.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(gray_small.shape[0] * gray_small.shape[1])
        best: Optional[tuple[float, np.ndarray]] = None

        for contour in contours:
            if cv2.contourArea(contour) <= self.min_area_ratio * image_area:
                continue
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, self.epsilon_ratio * perimeter, True)
            if len(approx) != 4:
                continue
            quad = order_points(approx[:, 0, :])
            if not validate_quad(quad, image_area, self.min_area_ratio, self.max_aspect):
                continue
            score = polygon_area(quad) / image_area
            if max_area_fraction is not None and score > max_area_fraction:
                continue
            if best is None or score > best[0]:
                best = (score, quad)
        return best