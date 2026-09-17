"""Unit tests for geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scanvision.geometry import order_points, polygon_area, quad_dimensions, validate_quad


def test_order_points_expected_order():
    pts = np.array([[10, 0], [0, 0], [0, 10], [10, 10]], dtype=np.float32)
    ordered = order_points(pts)
    expected = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    np.testing.assert_allclose(ordered, expected)


def test_order_points_rotated_input():
    pts = np.array([[5, 5], [0, 0], [5, 0], [0, 5]], dtype=np.float32)
    ordered = order_points(pts)
    np.testing.assert_allclose(ordered[0], [0, 0])  # top-left
    np.testing.assert_allclose(ordered[2], [5, 5])  # bottom-right


def test_order_points_rejects_wrong_count():
    with pytest.raises(ValueError):
        order_points([[0, 0], [1, 1], [2, 2]])


def test_polygon_area_square():
    assert polygon_area([[0, 0], [4, 0], [4, 4], [0, 4]]) == pytest.approx(16.0)


def test_quad_dimensions():
    tl, tr, br, bl = [0, 0], [10, 0], [10, 6], [0, 6]
    width, height = quad_dimensions([tl, tr, br, bl])
    assert width == pytest.approx(10.0)
    assert height == pytest.approx(6.0)


def test_validate_quad_area_and_aspect():
    good = np.array([[0, 0], [8, 0], [8, 6], [0, 6]], dtype=np.float32)
    assert validate_quad(good, image_area=100.0, min_area_ratio=0.05, max_aspect=4.0)

    tiny = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    assert not validate_quad(tiny, image_area=100.0, min_area_ratio=0.05, max_aspect=4.0)

    thin = np.array([[0, 0], [40, 0], [40, 1], [0, 1]], dtype=np.float32)
    assert not validate_quad(thin, image_area=1000.0, min_area_ratio=0.01, max_aspect=4.0)