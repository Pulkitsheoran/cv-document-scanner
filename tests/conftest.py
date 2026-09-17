"""Shared fixtures: synthetic document images for pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.synth_dataset import generate_document  # noqa: E402


@pytest.fixture()
def synthetic_clean(tmp_path):
    """Generate a clean (undistorted) synthetic document. Returns (image_path, ground_truth_path)."""
    return generate_document(tmp_path / "clean", seed=101, photo=False)


@pytest.fixture()
def synthetic_photo(tmp_path):
    """Generate a perspective-warped + degraded synthetic document photo."""
    return generate_document(tmp_path / "photo", seed=101, photo=True)


@pytest.fixture()
def synthetic_photos(tmp_path):
    """Generate a batch of synthetic photos and the ground-truth texts."""
    from tools.synth_dataset import generate_documents

    pairs = generate_documents(tmp_path, count=3, start_seed=201, photo=True)
    return pairs, tmp_path / "ground_truth"