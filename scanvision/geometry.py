"""Low-level geometry helpers: point ordering and quadrangle utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

FLOAT32 = np.float32


def order_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    The classical sum/difference heuristic: the point with the smallest x+y
    sum is the top-left corner and the largest sum is the bottom-right; the
    smallest x-y difference corresponds to the top-right and the largest to
    the bottom-left.
    """
    pts = np.asarray(points, dtype=FLOAT32).reshape(4, 2)
    if not (len(pts) == 4):
        raise ValueError(f"expected 4 points, got {len(pts)}")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=FLOAT32)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace area of a polygon given by its vertices."""
    pts = np.asarray(points, dtype=FLOAT32)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def quad_dimensions(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Average width and height of an ordered quadrangle (TL,TR,BR,BL)."""
    tl, tr, br, bl = (np.asarray(p, dtype=FLOAT32) for p in points)
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    width = 0.5 * (top + bottom)
    height = 0.5 * (left + right)
    return width, height


def validate_quad(
    points: Sequence[Sequence[float]], image_area: float, min_area_ratio: float, max_aspect: float
) -> bool:
    """Heuristic sanity checks that a quadrangle is a plausible document page."""
    area = polygon_area(points)
    if area <= min_area_ratio * image_area:
        return False
    width, height = quad_dimensions(points)
    aspect = (max(width, height) / min(width, height)) if min(width, height) > 0 else float("inf")
    return aspect <= max_aspect