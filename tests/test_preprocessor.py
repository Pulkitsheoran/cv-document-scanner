"""Unit tests for the image preprocessor."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scanvision.preprocessor import ImagePreprocessor


def _checkerboard_image(h, w, cell=16):
    """Grayscale checkerboard yields edges everywhere."""
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, cell):
        for j in range(0, w, cell):
            if (i // cell + j // cell) % 2 == 0:
                img[i : i + cell, j : j + cell] = 255
    return img


def test_grayscale_conversion():
    color = np.zeros((20, 30, 3), dtype=np.uint8)
    color[:, :, 0] = 255  # pure blue in BGR
    pre = ImagePreprocessor()
    gray = pre.to_gray(color)
    assert gray.ndim == 2
    assert gray.shape == (20, 30)
    assert gray.dtype == np.uint8


def test_bilateral_keeps_shape_and_type():
    pre = ImagePreprocessor()
    noisy = _checkerboard_image(64, 64).astype(np.uint8)
    denoised = pre.denoise(noisy)
    assert denoised.shape == noisy.shape
    assert denoised.dtype == np.uint8


def test_illumination_compensation_reduces_gradient():
    pre = ImagePreprocessor()
    gradient = np.tile(np.linspace(0, 255, 256), (100, 1)).astype(np.uint8)

    def variance(x: np.ndarray) -> float:
        return float(np.var(x.astype(np.float32)))

    compensated = pre.compensate_illumination(cv2.GaussianBlur(gradient, (1, 1), 0))
    assert variance(compensated) < variance(gradient)
    assert compensated.dtype == np.uint8


def test_edge_map_non_empty_for_structured_image():
    pre = ImagePreprocessor()
    edges = pre.edge_map(_checkerboard_image(128, 128))
    assert cv2.countNonZero(edges) > 100


def test_fit_scale_limits_longer_side():
    pre = ImagePreprocessor()
    img = np.zeros((2000, 3000), dtype=np.uint8)
    resized, scale_inv = pre.fit_scale(img, max_dim=1600)
    assert max(resized.shape) == 1600
    assert scale_inv == pytest.approx(3000 / 1600)