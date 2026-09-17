"""Output export: PDF albums of scans and plain-text OCR transcripts."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from scanvision.exceptions import ExportError

try:
    import matplotlib

    _UNICODE_TTF = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    _BOLD_TTF = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
except Exception:  # noqa: BLE001
    _UNICODE_TTF = ""
    _BOLD_TTF = ""


class ScanExporter:
    """Writes scan results to disk.

    ``export_pdf`` produces an A4 album: each scan occupies a full page and,
    when requested, a following page contains the machine-extracted text so
    both the image and the readable transcript ship in one file.
    """

    LOGGER = logging.getLogger("scanvision.exporter")

    def __init__(self) -> None:
        self._tmp: tempfile.TemporaryDirectory | None = None

    def _temp_result_dir(self) -> Path:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="scanvision_")
        return Path(self._tmp.name)

    def export_pdf(self, scans: Iterable[tuple[np.ndarray | None, str, str]], out_path: Path) -> Path:
        """Build a multi-page PDF from ``(image_bgr_or_none, title, ocr_text)`` entries."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pdf = FPDF(orientation="P", unit="mm", format="A4")
            if _UNICODE_TTF:
                pdf.add_font("DejaVu", "", _UNICODE_TTF)
                if _BOLD_TTF:
                    pdf.add_font("DejaVu", "B", _BOLD_TTF)
            font_name = "DejaVu" if _UNICODE_TTF else "helvetica"
            pdf.set_title("ScanVision output")
            pdf.set_author("ScanVision Document Scanner")
            page_w, page_h = pdf.w, pdf.h
            margin = 8.0

            for image, title, text in scans:
                pdf.add_page()
                pdf.set_font(font_name, "B", 14)
                pdf.set_text_color(30, 30, 30)
                pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                if image is not None:
                    tmp_image = self._persist_image(image)
                    width, height = image.shape[1], image.shape[0]
                    scale = min(
                        (page_w - 2 * margin) / width,
                        (page_h - 2 * margin - 14) / height,
                    )
                    img_w = width * scale
                    img_h = height * scale
                    x = (page_w - img_w) / 2
                    y = 14 + (page_h - 14 - img_h) / 2
                    pdf.image(str(tmp_image), x=x, y=y, w=img_w, h=img_h)
                if (text or "").strip():
                    pdf.set_font(font_name, "", 9)
                    pdf.set_text_color(60, 60, 60)
                    pdf.multi_cell(0, 5, f"\nExtracted text:\n{text}")
            pdf.output(str(out_path))
        except Exception as exc:  # noqa: BLE001
            self.LOGGER.exception("PDF export failed")
            raise ExportError(f"failed to write PDF {out_path}: {exc}") from exc
        self.LOGGER.info("wrote PDF %s (%d bytes)", out_path.name, out_path.stat().st_size)
        return out_path

    @staticmethod
    def export_text(text: str, out_path: Path) -> Path:
        """Atomically write the OCR transcript to a UTF-8 text file."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        try:
            tmp.write_text(text or "", encoding="utf-8")
            os.replace(tmp, out_path)
        except OSError as exc:
            raise ExportError(f"failed to write text file {out_path}: {exc}") from exc
        return out_path

    def _persist_image(self, image: np.ndarray) -> Path:
        """Temporarily materialise a BGR numpy array as PNG for the PDF renderer."""
        encoded, buffer = cv2.imencode(".png", image)
        if not encoded:
            raise ExportError("failed to encode scan for PDF export")
        file_path = self._temp_result_dir() / f"{uuid.uuid4().hex}.png"
        file_path.write_bytes(buffer.tobytes())
        return file_path