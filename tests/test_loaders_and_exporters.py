"""Tests for the image loader and exporters."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scanvision.loader import DocumentLoader
from scanvision.exceptions import ImageLoadError, ExportError
from scanvision.exporter import ScanExporter
from scanvision.evaluator import OCREvaluator


@pytest.fixture()
def loader():
    return DocumentLoader()


def test_load_valid_image(loader, tmp_path):
    path = tmp_path / "img.JPG"
    cv2.imwrite(str(path), np.full((50, 60, 3), 200, np.uint8))
    image = loader.load(path)
    assert image.shape[:2] == (50, 60)


def test_load_missing_file_raises(loader, tmp_path):
    with pytest.raises(ImageLoadError):
        loader.load(tmp_path / "nope.jpg")


def test_load_unsupported_extension_raises(loader, tmp_path):
    path = tmp_path / "doc.heic"
    path.write_bytes(b"x")
    with pytest.raises(ImageLoadError):
        loader.load(path)


def test_load_corrupt_file_raises(loader, tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        loader.load(path)


def test_discover_directory(loader, tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hello")
    found = loader.discover(tmp_path)
    assert sorted(p.name for p in found) == ["a.png", "b.jpg"]


def test_discover_recursive(loader, tmp_path):
    deep = tmp_path / "nested" / "deeper"
    deep.mkdir(parents=True)
    (deep / "d.png").write_bytes(b"x")
    found = loader.discover(tmp_path, recursive=True)
    assert len(found) == 1


def test_export_text_atomic(loader, tmp_path):
    target = tmp_path / "out.txt"
    ScanExporter.export_text("some ocr text", target)
    assert target.read_text() == "some ocr text"


def test_export_pdf_writes_real_pdf(tmp_path):
    image = np.full((300, 400, 3), 240, np.uint8)
    out = ScanExporter().export_pdf([(image, "mini scan", "extracted words")], tmp_path / "scan.pdf")
    header = out.read_bytes()[:5]
    assert header == b"%PDF-"


def test_export_pdf_survives_unicode_text(tmp_path):
    image = np.full((80, 80, 3), 255, np.uint8)
    out = ScanExporter().export_pdf(
        [(image, "unicode", "résumé — data — Café")], tmp_path / "utf.pdf"
    )
    assert out.exists() and out.stat().st_size > 0


def test_evaluator_roundtrip_through_loader(synthetic_clean, loader):
    image_path, gt_path = synthetic_clean
    loader.load(image_path)
    assert gt_path.read_text().strip()