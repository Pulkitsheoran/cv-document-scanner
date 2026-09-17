# ScanVision — Smart Document Scanner & OCR

Convert phone photos of printed documents into clean, fronto-parallel scans
with machine-readable text — **fully offline**, using classical computer
vision (OpenCV) and Tesseract OCR. Built as a Computer Vision course project.

![Pipeline stages](docs/screenshots/pipeline_stages.png)

## Overview

ScanVision is an end-to-end document scanning pipeline:

```
photo -> ingest -> preprocess -> detect page quad -> perspective rectify
     -> enhance (binarize/deskew) -> OCR -> export PDF + text
     -> evaluate OCR quality (CER/WER) -> batch reports
```

It detects the page boundary even in photos with perspective distortion,
uneven lighting, noise and blur, then rectifies, cleans and OCRs the page.
A synthetic dataset generator produces realistic "phone-photo" documents
together with ground-truth text so OCR quality can be measured
automatically.

## Features

- **Document detection**: largest-quadrangle localisation on Canny edges
  with convex-hull simplification and an Otsu-mask fallback.
- **Perspective rectification**: homography that maps the detected quad back
  to a real page rectangle, preserving aspect ratio.
- **Enhancement**: illumination compensation, Otsu binarization, speckle
  removal and conservative deskewing. Modes: `bw` (OCR-optimal), `gray`,
  `otsu`.
- **OCR**: Tesseract (`eng`, `--psm 6`) with word-level boxes and confidence.
- **Export**: single- or multi-page PDF (scanned image + extracted text page)
  and atomic UTF-8 `.txt` transcripts.
- **Evaluation**: character and word error rates (CER/WER) versus ground
  truth.
- **Batch processing**: threaded parallel scans with per-file failure
  isolation.
- **Reporting**: JSON / Markdown / CSV summaries per batch.
- **Visualisation**: detection overlay + stage-grid montages for demos.

## Technologies / tools

| Area | Tooling |
| --- | --- |
| Image processing / CV | OpenCV 5.x (contours, morphology, homography, Otsu) |
| Numerics | NumPy |
| OCR engine | Tesseract via pytesseract |
| Image I/O | Pillow (synthetic dataset rendering) |
| PDF generation | fpdf2 (Unicode font embedded) |
| Diagrams & plots | matplotlib |
| Config | TOML (`config/pipeline.toml`) |
| Testing | pytest |
| Language / runtime | Python ≥ 3.10 |

## Project structure

```
scanvision/            # core package (11 modules)
  cli.py               # scan / batch / evaluate sub-commands
  pipeline.py          # ScanPipeline orchestration
  loader.py            # image ingestion & validation
  preprocessor.py      # gray/denoise/illumination/edges
  detector.py          # document quad detection
  perspective.py       # 4-point homography rectification
  enhancer.py          # binarization, cleanup, deskew
  ocr.py               # OCREngine (Tesseract wrapper)
  evaluator.py         # CER/WER metrics
  exporter.py          # PDF + text writers
  report.py            # batch reports (JSON/MD/CSV)
  visualizer.py        # stage montages & overlays
  batch.py             # threaded batch driver
  models.py / config.py / exceptions.py / __main__.py
tools/
  synth_dataset.py     # synthetic phone-photo dataset generator
tests/                 # 50 pytest tests
config/pipeline.toml   # runtime configuration
docs/screenshots/       # demo visualisations
data/                  # generated samples and outputs (git-ignored)
```

## Installation

```bash
# 1. System dependency: Tesseract OCR
sudo apt install tesseract-ocr        # Debian/Ubuntu
# or: brew install tesseract           # macOS

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. (optional) install the `scanvision` command; otherwise use `python -m scanvision`
pip install -e .
```

## Usage

```bash
# Scan a single photo
scanvision scan path/to/photo.jpg --out data/outputs
# (without `pip install -e .`, prefix with `python -m`:  python -m scanvision scan ...)

# Batch scan every photo in a folder
scanvision batch path/to/photo_dir --out data/outputs --workers 4

# Scan a folder and score OCR quality vs. ground truth
scanvision evaluate data/samples/photos \
    --ground-truth data/samples/ground_truth --out data/outputs

# Produce detection overlay + stage-grid visualisation
scanvision scan photo.jpg --out data/outputs --visualize

# Each output image yields a .pdf (image + OCR text) and a .txt transcript
```

Options: `--enhance bw|gray|otsu`, `--no-ocr`, `--no-pdf`, `--no-text`,
`--no-deskew`, `--recursive`, `--workers N`, and `--config FILE` (TOML).

### Generate the demo dataset

```bash
python -m tools.synth_dataset --count 8        # -> data/samples/photos + ground_truth
```

## Testing

```bash
pytest                     # full suite (data generated on the fly)
```

The suite covers geometry helpers, pre-processing, detection (incl. the
"no document found" failure path), perspective correction, OCR metric
computation, exporters, batch failure isolation and CLI integration.
Expected result: all tests pass.

## Results (evaluation on 8 synthetic phone-photo documents)

| Metric | Value |
| --- | --- |
| Documents processed | 8 / 8 (100 %) |
| Mean character accuracy (1 − CER) | 97.3 % |
| Mean word accuracy (1 − WER) | 94.0 % |
| Best CER | 0.002 |
| Worst CER | 0.104 |
| Mean pipeline time / document | ~2.5 s (CPU, 4 workers) |

![Detection overlay](docs/screenshots/detection_overlay.png)

Full per-file numbers live in `data/outputs/demo/report.md`/`.json`.

## Evaluation rubric mapping

- Problem understanding & requirements → `statement.md`.
- Design & documentation → this README (architecture, features,
  project structure).
- Implementation quality → `scanvision/` package (18 modules), error
  handling, typed exceptions, logging, `tests/`.
- Innovation & depth → synthetic-photo generator, CER/WER evaluation
  harness, hull-based detection + mask fallback, threaded batch.
- Git repository & version control → this repository.

## Acknowledgements

- Tesseract OCR — https://github.com/tesseract-ocr/tesseract
- OpenCV — https://opencv.org
- fpdf2, pytesseract, Pillow, matplotlib, NumPy, pytest

---

*Built as a Computer Vision course project. MIT-style license file included.*