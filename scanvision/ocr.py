"""OCR engine wrapper around Tesseract via pytesseract."""

from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np

from scanvision.exceptions import OCRError
from scanvision.models import OCRWord

try:  # pragma: no cover - import used for marker only
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None  # type: ignore[assignment]


class OCREngine:
    """Thin, configurable wrapper around the Tesseract OCR engine.

    Exposes both raw text extraction (:meth:`extract_text`) and word-level
    output with bounding boxes (:meth:`extract_data`), the latter being used
    by the visualizer to draw highlights on the scanned page.
    """

    LOGGER = logging.getLogger("scanvision.ocr")

    def __init__(self, lang: str = "eng", psm: int = 6, oem: int = 3, tesseract_cmd: str | None = None) -> None:
        if pytesseract is None:
            raise OCRError("pytesseract is not installed; run 'pip install pytesseract'")
        self.lang = lang
        self.psm = psm
        self.oem = oem
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @property
    def _config(self) -> str:
        return f"--psm {self.psm} --oem {self.oem}"

    def extract_text(self, image: np.ndarray) -> str:
        """Run OCR and return the recognised text (leading/trailing whitespace trimmed)."""
        try:
            text = pytesseract.image_to_string(image, lang=self.lang, config=self._config)
        except Exception as exc:  # noqa: BLE001
            raise OCRError(f"tesseract failed: {exc}") from exc
        if text is None:
            return ""
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()

    def extract_data(self, image: np.ndarray) -> list[OCRWord]:
        """Run OCR and return word-level detections with confidence and boxes."""
        try:
            data = pytesseract.image_to_data(
                image, lang=self.lang, config=self._config, output_type=pytesseract.Output.DICT
            )
        except Exception as exc:  # noqa: BLE001
            raise OCRError(f"tesseract failed: {exc}") from exc

        words: list[OCRWord] = []
        for i, word in enumerate(data.get("text", [])):
            word = str(word).strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = 0.0
            if conf <= 0:
                continue
            words.append(
                OCRWord(
                    word=word,
                    left=int(data["left"][i]),
                    top=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                    confidence=conf,
                )
            )
        return words

    @staticmethod
    def draw_overlay(image: np.ndarray, words: Iterable[OCRWord]) -> np.ndarray:
        """Draw bounding boxes + recognised words on a BGR image."""
        overlay = image.copy()
        for word in words:
            cv2.rectangle(
                overlay,
                (word.left, word.top),
                (word.left + word.width, word.top + word.height),
                (0, 160, 255),
                1,
            )
            cv2.putText(
                overlay,
                word.word,
                (word.left, max(0, word.top - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 160, 255),
                1,
                cv2.LINE_AA,
            )
        return overlay