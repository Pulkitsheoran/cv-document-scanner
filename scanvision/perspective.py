"""Perspective correction: warp the detected quadrangle square onto the fronto-parallel plane."""

from __future__ import annotations

import numpy as np
import cv2

from scanvision.geometry import quad_dimensions


class PerspectiveTransformer:
    """Removes the viewing-angle distortion of a photographed page.

    A homography maps the four detected page corners to the corners of an
    output rectangle sized from the mean edge lengths of the quadrangle,
    preserving the natural page aspect ratio.
    """

    def __init__(self, output_aspect: float | None = None) -> None:
        # Optional fixed output ratio (width / height). None keeps the ratio
        # measured from the detected quadrangle.
        self.output_aspect = output_aspect

    def output_size(self, quad: np.ndarray) -> tuple[int, int]:
        """Derive an output (width, height) from the quadrangle's mean edges."""
        width, height = quad_dimensions(quad)
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))
        if self.output_aspect:
            target_width = max(1, int(round(height * self.output_aspect)))
            return (target_width, height)
        return (width, height)

    def transform(self, image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Apply the perspective warp, returning the rectified page image."""
        quad = quad.reshape(4, 2).astype(np.float32)
        width, height = self.output_size(quad)
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
        )
        homography = cv2.getPerspectiveTransform(quad, destination)
        warped = cv2.warpPerspective(
            image, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        return warped