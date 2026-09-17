"""Unit tests for document detection and perspective correction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scanvision.detector import DocumentDetector
from scanvision.exceptions import NoDocumentFoundError
from scanvision.perspective import PerspectiveTransformer
from scanvision.geometry import order_points


def test_detector_finds_page_of_clean_document(synthetic_clean):
    image_path, _ = synthetic_clean
    image = cv2.imread(str(image_path))
    detector = DocumentDetector()
    quad, score = detector.detect(image)
    assert quad.shape == (4, 2)
    assert score > 0.6

    expected = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], dtype=np.float32)
    np.testing.assert_allclose(quad, expected, atol=12)


def test_detector_finds_page_of_warped_photo(synthetic_photo):
    image_path, _ = synthetic_photo
    image = cv2.imread(str(image_path))
    quad, _ = DocumentDetector().detect(image)
    assert quad.shape == (4, 2)
    ordered = order_points(quad)
    assert ordered[0][1] < ordered[2][1]  # top-left above bottom-right


def test_detector_raises_on_blank_image():
    detector = DocumentDetector()
    blank = np.full((600, 800), 200, dtype=np.uint8)
    with pytest.raises(NoDocumentFoundError):
        detector.detect(blank)


def test_perspective_produces_rectified_page(synthetic_photo):
    image_path, _ = synthetic_photo
    image = cv2.imread(str(image_path))
    quad, _ = DocumentDetector().detect(image)
    transformer = PerspectiveTransformer()
    page = transformer.transform(image, quad)
    assert page.ndim == 3
    assert page.shape[0] > 0 and page.shape[1] > 0
    assert page.shape[1] / page.shape[0] == pytest.approx(
        image.shape[1] / image.shape[0], rel=0.25
    )


def test_perspective_identity_is_roughly_lossless():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (60, 80), (540, 320), (0, 200, 0), -1)
    quad = np.array([[0, 0], [600, 0], [600, 400], [0, 400]], dtype=np.float32)
    page = PerspectiveTransformer().transform(image, quad)
    assert page.shape[0] == 400 and page.shape[1] == 600
    assert page.mean() == pytest.approx(image.mean(), rel=0.1)  # content carried across