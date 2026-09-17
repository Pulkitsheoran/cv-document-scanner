"""Configuration loading and merging for the ScanVision pipeline."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

from scanvision.exceptions import ConfigurationError
from scanvision.models import ScanOptions

DEFAULT_CONFIG_PATH = Path("config/pipeline.toml")

DEFAULTS: Dict[str, Any] = {
    "out_dir": "data/outputs",
    "workers": 4,
    "max_dim": 1600,
    "min_area_ratio": 0.02,
    "max_aspect": 4.0,
    "fix_aspect": True,
    "deskew": True,
    "enhance_mode": "bw",
    "do_ocr": True,
    "export_pdf": True,
    "export_text": True,
    "visualize": False,
    "tesseract": {"lang": "eng", "psm": 6, "oem": 3, "cmd": None},
    "resize_to_a4": False,
}


class ScanConfig:
    """Holds resolved configuration and can turn it into :class:`ScanOptions`.

    Values are resolved with increasing precedence:
    built-in defaults < on-disk TOML < explicit keyword arguments.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        merged = dict(DEFAULTS)
        if data:
            merged.update({k: v for k, v in data.items() if v is not None})
        self._data = merged
        self.scan_options = ScanOptions(
            enhance_mode=str(self._data["enhance_mode"]),
            do_ocr=bool(self._data["do_ocr"]),
            export_pdf=bool(self._data["export_pdf"]),
            export_text=bool(self._data["export_text"]),
            deskew=bool(self._data["deskew"]),
            fix_aspect=bool(self._data["fix_aspect"]),
            visualize=bool(self._data["visualize"]),
        )

    @classmethod
    def from_file(cls, path: Path) -> "ScanConfig":
        if not Path(path).is_file():
            raise ConfigurationError(f"config file not found: {path}")
        try:
            with open(path, "rb") as fh:
                raw = tomllib.load(fh)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ConfigurationError(f"failed to parse config {path}: {exc}") from exc
        return cls(raw)

    @property
    def out_dir(self) -> Path:
        return Path(self._data["out_dir"])

    @property
    def workers(self) -> int:
        return max(1, int(self._data["workers"]) or 1)

    @property
    def max_dim(self) -> int:
        return int(self._data["max_dim"])

    @property
    def min_area_ratio(self) -> float:
        return float(self._data["min_area_ratio"])

    @property
    def max_aspect(self) -> float:
        return float(self._data["max_aspect"])

    @property
    def ocr(self) -> Dict[str, Any]:
        return dict(self._data["tesseract"] or {})

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


CONFIG = ScanConfig()  # module-level default-config singleton