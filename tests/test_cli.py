"""CLI integration tests (run in-process for speed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanvision.cli import main  # noqa: E402


def test_cli_version(capsys):
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert "scanvision" in capsys.readouterr().out


def test_cli_scan_single_file(synthetic_clean, tmp_path):
    image_path, _ = synthetic_clean
    code = main(["--config", "config/pipeline.toml", "scan", str(image_path), "--out", str(tmp_path / "out")])
    assert code == 0
    assert (tmp_path / "out" / f"{image_path.stem}.pdf").exists()
    assert (tmp_path / "out" / f"{image_path.stem}.txt").exists()


def test_cli_evaluate_writes_reports(synthetic_clean, tmp_path):
    image_path, gt_path = synthetic_clean
    photos = tmp_path / "photos"
    photos.mkdir()
    (photos / image_path.name).write_bytes(image_path.read_bytes())
    gt = tmp_path / "ground_truth"
    gt.mkdir()
    (gt / f"{image_path.stem}.txt").write_text(gt_path.read_text())
    code = main(
        [
            "evaluate",
            str(photos),
            "--ground-truth", str(gt),
            "--out", str(tmp_path / "out"),
            "--workers", "1",
        ]
    )
    assert code == 0
    report = tmp_path / "out" / "report.json"
    assert report.exists()