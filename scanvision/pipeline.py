"""ScanPipeline: orchestrates the full ingest -> export workflow for one image."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from scanvision.batch import BatchProcessor
from scanvision.detector import DocumentDetector
from scanvision.enhancer import DocumentEnhancer
from scanvision.evaluator import OCREvaluator, TextNormalizer
from scanvision.exceptions import ScanVisionError
from scanvision.exporter import ScanExporter
from scanvision.loader import DocumentLoader
from scanvision.models import ScanOptions, ScanResult
from scanvision.ocr import OCREngine
from scanvision.perspective import PerspectiveTransformer
from scanvision.visualizer import PipelineVisualizer


class ScanPipeline:
    """End-to-end pipeline.

    The canonical workflow:

        ingest (raw) -> preprocess -> detect quad -> perspective warp
        -> enhance -> OCR -> export PDF/text -> (optional) visualise
    """

    LOGGER = logging.getLogger("scanvision.pipeline")

    def __init__(
        self,
        max_dim: int = 1600,
        min_area_ratio: float = 0.02,
        max_aspect: float = 4.0,
        ocr_engine: Optional[OCREngine] = None,
    ) -> None:
        self.loader = DocumentLoader()
        self.detector = DocumentDetector(max_dim=max_dim, min_area_ratio=min_area_ratio, max_aspect=max_aspect)
        self.transformer = PerspectiveTransformer()
        self.enhancer = DocumentEnhancer()
        self.ocr = ocr_engine or OCREngine()
        self.exporter = ScanExporter()
        self.visualizer = PipelineVisualizer()
        self.evaluator = OCREvaluator()

    def process(
        self,
        input_path: Path,
        out_dir: Path,
        options: Optional[ScanOptions] = None,
        reference_text: Optional[str] = None,
    ) -> ScanResult:
        """Run the full pipeline on one photo and write the outputs to ``out_dir``."""
        options = options or ScanOptions()
        start = time.perf_counter()
        result = ScanResult(input_path=Path(input_path), stem=Path(input_path).stem)

        try:
            image = self.loader.load(input_path)
            result.stages["loaded_shape"] = image.shape

            quad, score = self.detector.detect(image)
            result.quad = quad
            result.stages["document_score"] = score

            page = self.transformer.transform(image, quad)
            result.stages["rectified_shape"] = page.shape

            enhanced = self.enhancer.enhance(page, mode=options.enhance_mode, deskew=options.deskew)
            result.stages["enhanced_shape"] = enhanced.shape

            output_base = out_dir / f"{result.stem}{options.output_suffix}"
            output_base.parent.mkdir(parents=True, exist_ok=True)

            if options.do_ocr:
                result.ocr_text = self.ocr.extract_text(enhanced)
                result.words = self.ocr.extract_data(enhanced)
                result.stages["ocr_char_count"] = len(TextNormalizer.normalize(result.ocr_text))
                result.stages["ocr_word_count"] = len(TextNormalizer.tokens(result.ocr_text))

            if options.export_pdf:
                text_for_pdf = result.ocr_text or ("(OCR disabled)" if not options.do_ocr else "")
                pdf_path = self.exporter.export_pdf([(page, result.stem, text_for_pdf)], output_base.with_suffix(".pdf"))
                result.output_paths.append(pdf_path)

            if options.export_text and options.do_ocr:
                txt_path = self.exporter.export_text(result.ocr_text or "", output_base.with_suffix(".txt"))
                result.output_paths.append(txt_path)

            if reference_text:
                metrics = self.evaluator.metrics(reference_text, result.ocr_text or "")
                result.cer = metrics.cer
                result.wer = metrics.wer
                result.char_accuracy = metrics.char_accuracy
                result.word_accuracy = metrics.word_accuracy

            if options.visualize:
                viz_dir = output_base.parent / "_visuals"
                self.visualizer.overlay_quad(image, quad, viz_dir / f"{result.stem}_detection.png")
                stages = {
                    "1. original": image,
                    "2. detected region": self._crop(image, quad),
                    "3. rectified page": page,
                    f"4. enhanced ({options.enhance_mode})": enhanced,
                }
                if result.words:
                    display = (
                        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                        if enhanced.ndim == 2
                        else enhanced
                    )
                    stages["5. OCR boxes"] = self.ocr.draw_overlay(display, result.words)
                grid = self.visualizer.stage_grid(stages, viz_dir / f"{result.stem}_stages.png")
                result.output_paths.append(grid)

            result.ok = True
        except (ScanVisionError, OSError, ValueError) as exc:
            self.LOGGER.error("pipeline failed for %s: %s", result.stem, exc)
            result.error = str(exc)
            result.ok = False
        finally:
            result.duration = time.perf_counter() - start
        return result

    @staticmethod
    def _crop(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Smallest axis-aligned crop fully containing the detected quadrangle."""
        q = np.asarray(quad, dtype=np.int32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(q)
        return image[y : y + h, x : x + w]

    def batch(
        self,
        paths: list[Path],
        out_dir: Path,
        options: Optional[ScanOptions] = None,
        workers: int = 1,
        reference_texts: Optional[dict[str, str]] = None,
    ) -> list[ScanResult]:
        """Process several photos, optionally in parallel, with failure isolation."""
        return BatchProcessor(self, workers=workers).run(paths, out_dir, options, reference_texts)