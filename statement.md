# Project Statement

## Problem statement

Hand-held photos of printed documents suffer from perspective distortion,
uneven illumination, noise and blur. Converting such photos into clean,
machine-readable documents normally requires desktop scanning hardware or
paid "scan-to-PDF" applications. The problem is to build a fully offline,
classical-computer-vision pipeline that: (1) detects the page boundary in a
photographed document, (2) removes the viewing-angle distortion, (3)
binarizes and cleans the page, (4) extracts the text with OCR, and (5)
exports a PDF plus a text transcript — while (6) reporting quantitative OCR
quality metrics (CER/WER) against ground truth.

## Scope

- **In scope**

  - Single-image and folder-level (batch) document scanning on plain photos
    (`.jpg/.png/.bmp/.tiff/.webp`).
  - Automatic page-localization, perspective rectification, and
    enhancement in three export modes (`bw`, `gray`, `otsu`).
  - OCR text extraction and word-level bounding-box information.
  - PDF and `.txt` export for every scan.
  - Batch reports in JSON / Markdown / CSV form.
  - Synthetic "phone-photo" dataset generator that produces images together
    with ground-truth text, used both for demos and for automated OCR
    evaluation.
  - Unit and end-to-end tests (pytest).

- **Out of scope**

  - Handwriting recognition, multi-page photo-album stitching, cloud
    services, multi-language OCR (engine is configured for English), and
    live webcam scanning.

## Target users

- Students and professionals who need to digitise loose paper documents with
  a smartphone on an unlimited-offline budget.
- Academic users working with photographed lab notes, invoices, receipts and
  printed reports.
- CV/ML developers who want a small, dependency-light reference for
  OpenCV-based geometric image processing and OCR quality evaluation.

## High-level features

1. **Document localisation** — largest-quadrangle detection with
   convex-hull simplification and an Otsu-mask fallback.
2. **Perspective rectification** — 4-point homography that maps the page
   back to a fronto-parallel rectangle, preserving the natural aspect ratio.
3. **Page enhancement** — illumination compensation, Otsu binarization,
   speckle removal and conservative residual deskewing.
4. **OCR integration** — Tesseract via pytesseract; raw text and
   word-box data.
5. **Export** — multi-page PDF albums (image + extracted text) and atomic
   UTF-8 text transcripts.
6. **Evaluation** — character- and word-error-rate scoring (CER/WER) against
   ground truth.
7. **Batch processing** — threaded parallel scans with per-file failure
   isolation and progress reporting.
8. **Reporting & visualisation** — JSON/Markdown/CSV reports and stage-by-stage
   diagnostic montages.
9. **Configurable CLI** — `scan`, `batch` and `evaluate` sub-commands driven by
   a TOML configuration file with CLI overrides.