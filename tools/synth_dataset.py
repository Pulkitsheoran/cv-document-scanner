"""Synthetic document-photo dataset generator.

Renders clean pages of text and then simulates a handheld phone camera:

    - perspective distortion (random homography),
    - small whole-frame rotation,
    - Gaussian sensor noise and blur,
    - illumination gradient + vignetting,
    - JPEG compression artefacts.

The generator simultaneously emits the ground-truth text for every page so
OCR quality can be scored with character/word error rates. It is used for
the demo data, the unit tests and the evaluation section of the report.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAGE_WIDTH, PAGE_HEIGHT = 1200, 1697  # A4 at ~145 dpi
MARGIN = 90
BODY_FONT_SIZE = 34
TITLE_FONT_SIZE = 54

_CONTENT_TEMPLATES = [
    {
        "title": "Project Meeting Minutes",
        "body": [
            "The computer vision team met on Thursday to review the scanner prototype demo.",
            "The final output should batch-process a whole folder of documents automatically.",
            "Key decisions recorded by the secretary:",
            "- use classical image processing for edge detection and rectification",
            "- integrate OCR to export searchable text along with every scanned page",
            "- store evaluation metrics to measure recognition accuracy over time",
            "The next meeting is scheduled for the following Friday at ten o'clock.",
        ],
    },
    {
        "title": "Research Abstract - Optical Character Recognition",
        "body": [
            "Optical character recognition converts images of text into machine readable strings.",
            "Modern pipelines first localize the document region and remove perspective distortion.",
            "A binarized page performs better than a grayscale one for tesseract as the engine.",
            "This study compares clean scans with synthetic camera photos of the same pages.",
            "Perspective warp, blur, and uneven lighting are the dominant sources of error.",
        ],
    },
    {
        "title": "Invoice - GreenLeaf Supplies",
        "body": [
            "Billing address: North Bridge Road, Apartment 14, New Delhi, India.",
            "Items purchased today at the hardware store:",
            "- mechanical pencils, two boxes, quantity five, total four hundred rupees",
            "- A4 paper reams, three units, quantity two, total nine hundred rupees",
            "- whiteboard markers, one set, total two hundred and fifty rupees",
            "Payment due within thirty days. Please retain this receipt for records.",
        ],
    },
    {
        "title": "Employee Handbook Cheat Sheet",
        "body": [
            "New employees must complete the orientation before the first working day.",
            "The office operates from nine in the morning until six in the evening.",
            "Remember to badge in at the main entrance on every visit without fail.",
            "Remote work requires manager approval more than one week in advance.",
            "Forget your badge and the security desk will issue a visitor pass.",
        ],
    },
    {
        "title": "Lab Notes - Image Preprocessing",
        "body": [
            "Bilateral filtering preserves edges while smoothing flat background regions.",
            "Dividing by a strong blur compensates for uneven illumination on the page.",
            "Canny edge detection with auto thresholds performed well in bright rooms.",
            "Largest contour selection reliably recovered the page in all test photos.",
            "The perspective transform then mapped the quadrangle back to a rectangle.",
            "Final binarization with the Otsu threshold gave clean black on white text.",
        ],
    },
]


def _wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines, current, current_width = [], [], 0.0
    for word in words:
        width = draw.textlength(word + " ", font=font)
        if current and current_width + width > max_width:
            lines.append(" ".join(current))
            current, current_width = [], 0.0
        current.append(word)
        current_width += width
    if current:
        lines.append(" ".join(current))
    return lines


def render_page(template: dict, font_path: str = "") -> tuple[np.ndarray, str]:
    """Render a clean binary (white-bg/black-text) page. Returns (BGR image, ground-truth text)."""
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(font_path, TITLE_FONT_SIZE)
    body_font = ImageFont.truetype(font_path, BODY_FONT_SIZE)
    max_width = PAGE_WIDTH - 2 * MARGIN

    ground_truth_lines: list[str] = []
    cursor_y = MARGIN

    draw.text((MARGIN, cursor_y), template["title"], font=title_font, fill="black")
    ground_truth_lines.append(template["title"])
    cursor_y += TITLE_FONT_SIZE * 2

    draw.line((MARGIN, cursor_y, PAGE_WIDTH - MARGIN, cursor_y), fill="black", width=6)
    cursor_y += TITLE_FONT_SIZE

    for paragraph in template["body"]:
        if paragraph.startswith("- "):
            wrapped = [f"- {line}" if idx == 0 else f"  {line}" for idx, line in enumerate(_wrap_text(paragraph[2:], draw, body_font, max_width - 20))]
        else:
            wrapped = _wrap_text(paragraph, draw, body_font, max_width)
        for line in wrapped:
            draw.text((MARGIN, cursor_y), line, font=body_font, fill="black")
            ground_truth_lines.append(line)
            cursor_y += int(BODY_FONT_SIZE * 1.35)
        cursor_y += BODY_FONT_SIZE

    page = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(page, cv2.COLOR_RGB2BGR), "\n".join(ground_truth_lines)


def _canvas_for(quad: np.ndarray, pad: int = 120) -> tuple[int, int, int, int]:
    xs, ys = quad[:, 0], quad[:, 1]
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    offset_x = max(0, pad - min_x)
    offset_y = max(0, pad - min_y)
    width = max_x + offset_x + pad
    height = max_y + offset_y + pad
    return width, height, offset_x, offset_y


def warp_into_photo(
    page: np.ndarray, rng: np.random.Generator, warp_strength: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random perspective + rotation, return (photo, quad-in-canvas).

    The page is always kept fully in-frame so the demo dataset exercises the
    detector without deliberately introducing clipped corners.
    """
    height, width = page.shape[:2]
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Perturb each corner to simulate a non-orthogonal viewing angle.
    max_shift = (min(width, height)) * 0.14 * warp_strength
    shifts = rng.uniform(-max_shift, max_shift, size=(4, 2))
    dst = src + shifts.astype(np.float32)

    pad = int(0.15 * min(width, height))
    canvas_width, canvas_height, offset_x, offset_y = _canvas_for(dst, pad)
    dst = dst + np.array([offset_x, offset_y], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(page, homography, (canvas_width, canvas_height))

    # Small whole-frame rotation; map the quad through the same transform so
    # corner co-ordinates stay consistent with the produced image.
    angle = rng.uniform(-5.0, 5.0)
    if abs(angle) > 0.3:
        center = (canvas_width / 2.0, canvas_height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_dst = np.hstack([dst, np.ones((4, 1), np.float32)]) @ matrix.T
        x2, y2 = rotated_dst[:, 0], rotated_dst[:, 1]
        shift_x = int(pad - x2.min())
        shift_y = int(pad - y2.min())
        matrix[:, 2] += [shift_x, shift_y]
        width2 = int(max(1, x2.max() - x2.min() + 2 * pad))
        height2 = int(max(1, y2.max() - y2.min() + 2 * pad))
        photo = cv2.warpAffine(photo, matrix, (width2, height2))
        quad = rotated_dst + np.array([shift_x, shift_y], np.float32)
    else:
        quad = dst.copy()
    return photo, quad


def simulate_camera_artefacts(photo: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blur, noise, lighting gradient, vignette and JPEG compression."""
    result = photo.astype(np.float32) / 255.0
    result = cv2.GaussianBlur(result, (5, 5), rng.uniform(0.3, 1.1))
    noise = rng.normal(0.0, rng.uniform(0.012, 0.03), size=result.shape)
    result = result + noise

    height, width = result.shape[:2]
    top = rng.uniform(0.5, 1.0)
    bottom = rng.uniform(0.5, 1.0)
    gradient = np.linspace(top, bottom, height, dtype=np.float32).reshape(-1, 1, 1)
    result = result * gradient

    yy, xx = np.mgrid[0:height, 0:width]
    dist_x = (xx - width / 2) / (width / 2)
    dist_y = (yy - height / 2) / (height / 2)
    vignette = 1.0 - rng.uniform(0.05, 0.2) * (dist_x**2 + dist_y**2)
    result = result * vignette[..., np.newaxis]

    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    encode_ok, buffer = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not encode_ok:
        return result
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def default_font_path() -> str:
    """Locate the DejaVu Sans font bundled with matplotlib."""
    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")


def generate_document(
    out_dir: Path,
    seed: int,
    photo: bool = True,
    polygon_marker: bool = False,
    font_path: str = "",
) -> tuple[Path, Path]:
    """Generate one rendered document. Returns (image path, ground-truth path).

    The document image is saved as ``<stem>.jpg`` (or ``.png`` when ``photo``
    is disabled) and the ground truth as ``ground_truth/<stem>.txt``.
    A ``_quad.json`` sidecar is written when ``polygon_marker`` is set and is
    useful for detector unit tests.
    """
    rng = np.random.default_rng(seed)
    template = _CONTENT_TEMPLATES[seed % len(_CONTENT_TEMPLATES)]
    page, text = render_page(template, font_path=font_path or default_font_path())

    out_dir = Path(out_dir)
    image_dir = out_dir / ("photos" if photo else "clean")
    truth_dir = out_dir / "ground_truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    stem = _slug(template["title"], seed)
    if photo:
        photo_image, quad = warp_into_photo(page, rng)
        photo_image = simulate_camera_artefacts(photo_image, rng)
        image_path = image_dir / f"{stem}.jpg"
        cv2.imwrite(str(image_path), photo_image)
        if polygon_marker:
            sidecar = out_dir / f"{stem}_quad.npy"
            np.save(sidecar, quad)
    else:
        image_path = image_dir / f"{stem}.png"
        cv2.imwrite(str(image_path), page)
        if polygon_marker:
            sidecar = out_dir / f"{stem}_quad.png"
            cv2.imwrite(str(sidecar), page.copy())

    truth_path = truth_dir / f"{stem}.txt"
    truth_path.write_text(text, encoding="utf-8")
    return image_path, truth_path


def _slug(title: str, seed: int) -> str:
    words = [w for w in title.lower().replace("-", " ").split() if w.isalnum()]
    base = "-".join(words[:4]) or "document"
    return f"{base}-{seed:03d}"


def generate_documents(
    out_dir: Path, count: int = 8, start_seed: int = 1, photo: bool = True
) -> list[tuple[Path, Path]]:
    font_path = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    pairs = []
    for seed in range(start_seed, start_seed + count):
        pairs.append(generate_document(out_dir, seed, photo=photo, font_path=font_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="generate the ScanVision sample dataset")
    parser.add_argument("--out", default="data/samples", help="output directory")
    parser.add_argument("--count", type=int, default=8, help="number of documents")
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--clean", action="store_true", help="generate undistorted reference pages too")
    args = parser.parse_args()

    pairs = generate_documents(Path(args.out), count=args.count, start_seed=args.start_seed)
    print(f"generated {len(pairs)} document photos in {args.out}/")
    if args.clean:
        clean = generate_documents(Path(args.out) / "clean", count=args.count, start_seed=args.start_seed, photo=False)
        print(f"generated {len(clean)} clean reference pages in {args.out}/clean/")


if __name__ == "__main__":
    main()