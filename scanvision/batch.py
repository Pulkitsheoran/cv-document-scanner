"""Parallel batch processing with per-file failure isolation."""

from __future__ import annotations

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from scanvision.exceptions import ScanVisionError
from scanvision.models import ScanOptions, ScanResult


class BatchProcessor:
    """Thread-parallel driver over :class:`ScanPipeline.process`.

    A failed file never aborts the batch: the corresponding
    :class:`ScanResult` carries ``ok=False`` and an error message, and the
    processing continues with the remaining inputs.
    """

    LOGGER = logging.getLogger("scanvision.batch")

    def __init__(self, pipeline, workers: int = 1) -> None:
        self.pipeline = pipeline
        self.workers = max(1, workers)

    def run(
        self,
        paths: list[Path],
        out_dir: Path,
        options: Optional[ScanOptions] = None,
        reference_texts: Optional[dict[str, str]] = None,
    ) -> list[ScanResult]:
        """Execute the pipeline over ``paths`` and return one result per input."""
        if not paths:
            self.LOGGER.warning("batch called with no input files")
            return []

        options = options or ScanOptions()
        reference_texts = reference_texts or {}
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results: list[ScanResult] = []
        done = itertools.count(1)
        total = len(paths)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self.pipeline.process, path, out_dir, options, reference_texts.get(path.stem)
                ): path
                for path in paths
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - isolate all failures
                    self.LOGGER.exception("unexpected failure for %s", path)
                    result = ScanResult(input_path=path, stem=path.stem, ok=False, error=str(exc))
                results.append(result)
                n = next(done)
                if n % 5 == 0 or n == total:
                    self.LOGGER.info("batch progress %d/%d", n, total)

        ok_count = sum(1 for r in results if r.ok)
        self.LOGGER.info(
            "batch complete: %d succeeded, %d failed of %d",
            ok_count,
            total - ok_count,
            total,
        )
        return sorted(results, key=lambda r: r.stem)