"""Typed exception hierarchy for ScanVision."""

from __future__ import annotations


class ScanVisionError(Exception):
    """Base class for all domain errors raised by ScanVision."""


class ImageLoadError(ScanVisionError):
    """Raised when an input image cannot be read or validated."""


class NoDocumentFoundError(ScanVisionError):
    """Raised when no document quadrangle can be detected in an image."""


class OCRError(ScanVisionError):
    """Raised when the OCR engine fails to produce output."""


class ExportError(ScanVisionError):
    """Raised when an output artefact (PDF, text, report) cannot be written."""


class ConfigurationError(ScanVisionError):
    """Raised when the configuration file or CLI overrides are invalid."""