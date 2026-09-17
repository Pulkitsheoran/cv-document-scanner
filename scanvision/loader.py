"""Image ingestion: discovery, validation and reading of input photographs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

from scanvision.exceptions import ImageLoadError

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pnm", ".pgm"}


class DocumentLoader:
    """Reads and validates document photos from the filesystem.

    Uses ``cv2.imdecode`` on raw bytes so that paths containing non-ASCII
    characters still load correctly (``cv2.imread`` is unreliable on such
    paths on some platforms).
    """

    LOGGER = logging.getLogger("scanvision.loader")

    def load(self, path: Union[str, Path]) -> np.ndarray:
        file_path = Path(path)
        if not file_path.is_file():
            raise ImageLoadError(f"no such file: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_EXTS:
            raise ImageLoadError(
                f"{file_path.name}: unsupported image type '{file_path.suffix}', "
                f"expected one of {sorted(SUPPORTED_EXTS)}"
            )
        raw = file_path.read_bytes()
        if not raw:
            raise ImageLoadError(f"{file_path.name}: file is empty")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ImageLoadError(f"{file_path.name}: could not decode image")
        self.LOGGER.debug("loaded %s (%dx%d)", file_path.name, image.shape[1], image.shape[0])
        return image

    def discover(self, path: Union[str, Path], recursive: bool = False) -> List[Path]:
        """Return a sorted list of supported image files for a file or folder."""
        target = Path(path)
        if target.is_file():
            return [target] if target.suffix.lower() in SUPPORTED_EXTS else []
        if not target.is_dir():
            raise ImageLoadError(f"not a file or directory: {target}")
        pattern = "**/*" if recursive else "*"
        found = [p for p in target.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        return sorted(found)