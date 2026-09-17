"""End-to-end pipeline tests plus the batch driver and CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from scanvision.pipeline import ScanPipeline
from scanvision.models import ScanOptions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_pipeline_full_run_on_clean_document(synthetic_clean, tmp_path):
    image_path, _ = synthetic_clean
    result = ScanPipeline().process(image_path, tmp_path)
    assert result.ok, result.error
    assert result.ocr_text
    assert any(p.suffix == ".pdf" for p in result.output_paths)
    assert any(p.suffix == ".txt" for p in result.output_paths)
    assert result.quad is not None


def test_pipeline_scores_against_reference(synthetic_clean, tmp_path):
    image_path, gt_path = synthetic_clean
    result = ScanPipeline().process(image_path, tmp_path, reference_text=gt_path.read_text())
    assert result.ok
    assert result.cer is not None and result.cer < 0.1
    assert result.char_accuracy > 0.9


def test_pipeline_honours_disable_exports(synthetic_photo, tmp_path):
    image_path, _ = synthetic_photo
    options = ScanOptions(export_pdf=False, export_text=False, do_ocr=False)
    result = ScanPipeline().process(image_path, tmp_path, options=options)
    assert result.ok
    assert result.output_paths == []
    assert result.ocr_text is None


def test_pipeline_visualisation_artefacts(synthetic_photo, tmp_path):
    image_path, _ = synthetic_photo
    options = ScanOptions(visualize=True, export_pdf=False, export_text=False, do_ocr=False)
    result = ScanPipeline().process(image_path, tmp_path, options=options)
    assert result.ok, result.error
    visuals = list((tmp_path / "_visuals").glob("*.png"))
    assert len(visuals) == 2  # detection overlay + stage grid


def test_batch_processes_multiple_with_isolation(synthetic_photos, tmp_path):
    pairs, gt_dir = synthetic_photos
    references = {p.stem: g.read_text() for p, g in pairs}
    results = ScanPipeline().batch(
        [p for p, _ in pairs], tmp_path, workers=2, reference_texts=references
    )
    assert len(results) == 3
    assert all(r.ok for r in results)
    assert all(r.char_accuracy is not None and r.char_accuracy > 0.85 for r in results)


def test_batch_tolerates_missing_file(synthetic_clean, tmp_path):
    image_path, _ = synthetic_clean
    missing = tmp_path / "does-not-exist.jpg"
    results = ScanPipeline().batch([image_path, missing], tmp_path, workers=2)
    assert len(results) == 2
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    assert len(ok) == 1 and len(failed) == 1
    assert "ImageLoadError" in failed[0].error or "no such file" in failed[0].error