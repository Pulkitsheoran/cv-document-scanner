"""Generate the design diagrams for the ScanVision report.

Produces PNG figures in docs/diagrams/:
    architecture.png, workflow.png, usecase.png, class_diagram.png,
    sequence.png, er.png

Rendered with matplotlib's object-oriented API.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

OUT = Path(__file__).resolve().parent / "diagrams"


def _render(draw: callable, name: str, size=(10.0, 6.5)) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=size, dpi=160)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    draw(ax)
    figure.savefig(OUT / name, facecolor="white")
    print(f"wrote {OUT / name}")


def _box(ax, x, y, w, h, text, fc="#eef3fb", ec="#2f5d8f", lw=1.4, fs=8.5, bold=False):
    rect = matplotlib.patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight,
        wrap=True, linespacing=1.25,
    )


def _arrow(ax, x1, y1, x2, y2, text=None, fs=7.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1),
        zorder=1,
    )
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=fs, color="#444444")


def _arrow_lr(ax, x1, y1, x2, y2, text=None, fs=7.5):
    _arrow(ax, x1, y1, x2, y2, text, fs)


def architecture(ax):
    ax.text(5, 9.6, "ScanVision - System Architecture (layered)", ha="center", fontsize=12, fontweight="bold")
    layers = [
        (8.4, 1.9, "Presentation / CLI", "scanvision.cli  •  config/pipeline.toml", "#f9efd9"),
        (6.3, 1.9, "Application services", "ScanPipeline  •  BatchProcessor  •  ReportGenerator  •  PipelineVisualizer", "#e7f2df"),
        (3.4, 2.9, "Core computer-vision services",
         "DocumentLoader\nImagePreprocessor\nDocumentDetector\nPerspectiveTransformer\nDocumentEnhancer\nOCREngine\nScanExporter", "#ddeaf7"),
        (0.4, 1.6, "Infrastructure",
         "OpenCV  •  NumPy  •  Pillow  •  pytesseract / Tesseract  •  fpdf2", "#efefef"),
    ]
    for y, h, title, body, color in layers:
        ax.add_patch(matplotlib.patches.Rectangle((0.2, y), 9.6, h, facecolor=color, edgecolor="#5b6f84", linewidth=1.2, zorder=2))
        ax.text(0.6, y + h - 0.32, title, fontsize=9.5, fontweight="bold", va="center")
        ax.text(0.6, y + 0.5, body, fontsize=8, va="center", linespacing=1.35)
    _arrow(ax, 5, 8.4, 5, 6.6, "calls")
    _arrow(ax, 5, 6.3, 5, 4.3)


def workflow(ax):
    ax.text(5, 9.6, "Document scan workflow (process flow)", ha="center", fontsize=12, fontweight="bold")
    steps = [
        (0.3, "1. Ingest", "photo file"),
        (2.1, "2. Preprocess", "gray • denoise •\nillumination • edges"),
        (3.9, "3. Detect", "largest quad\ncontour + hull"),
        (5.7, "4. Detect", "page found?"),
        (7.5, "5. Rectify", "perspective warp"),
    ]
    box_w, h = 1.6, 1.0
    for x, title, body in steps:
        fc = "#fbf5e6" if title.startswith("4.") else "#eaf2fb"
        _box(ax, x, 6.3, box_w, h, f"{title}\n{body}", fc=fc)
    ax.add_patch(matplotlib.patches.Ellipse((7.2, 6.3 + h / 2), 1.7, 1.1, facecolor="#fdf3e3", edgecolor="#8a6d1a", lw=1.4, zorder=2))
    ax.text(7.2, 6.3 + h / 2, "found?", ha="center", va="center", fontsize=9)
    _arrow(ax, 1.9, 6.8, 2.2, 6.8)
    _arrow(ax, 3.7, 6.8, 4.0, 6.8)
    _arrow(ax, 5.5, 6.8, 6.4, 6.8)
    # no -> fallback
    ax.plot([7.2, 9.0], [6.8, 6.8], color="#333333", lw=1.1, linestyle="--", zorder=1)
    ax.text(8.0, 7.0, "no", fontsize=8)
    ax.plot([9.0, 9.0], [6.8, 3.3], color="#333333", lw=1.1, linestyle="--", zorder=1)
    _box(ax, 8.2, 2.3, 1.6, 1.0, "fallback\nOtsu mask\n(whole frame)", fc="#f6f0dc")
    ax.plot([8.2, 7.5], [2.8, 2.8], color="#333333", lw=1.1, linestyle="--", zorder=1)
    # yes -> down
    _arrow(ax, 7.2, 6.3, 7.2, 4.4)
    # post-process steps
    post = [
        (0.5, "6. Enhance", "Otsu bw • deskew"),
        (2.3, "7. OCR", "Tesseract\npsm=6 eng"),
        (4.1, "8. Export", "PDF + text"),
        (5.9, "9. Evaluate", "CER / WER\nvs ground truth"),
    ]
    for x, title, body in post:
        _box(ax, x, 1.0, 1.7, 1.0, f"{title}\n{body}")
    _arrow(ax, 1.3, 2.3, 1.35, 2.2)
    for i in range(1, len(post)):
        _arrow(ax, post[i - 1][0] + 1.7, 1.5, post[i][0], 1.5)
    _arrow(ax, 7.2, 4.4, 6.0, 2.3, "yes")


def usecase(ax):
    ax.text(5, 9.6, "Use case diagram", ha="center", fontsize=12, fontweight="bold")
    _box(ax, 0.3, 3.2, 1.6, 1.4, "User\n(primary)\n\n\n>>> depends scan", fc="#eaf2d9")
    ax.text(0.6, 7.3, "<<actor>>\nUser", ha="center", va="center", fontsize=9)
    ax.add_patch(matplotlib.patches.Rectangle((2.4, 1.1), 5.2, 8.0, facecolor="#fbfaf7", edgecolor="#555555", lw=1.4, zorder=1))
    ax.text(5.0, 8.7, "<<system boundary>> ScanVision", ha="center", fontsize=9.5, fontweight="bold")
    cases = [
        ("Scan single document", 7.2),
        ("Batch scan a folder", 6.4),
        ("Evaluate OCR accuracy", 5.6),
        ("Export PDF / text", 4.8),
        ("Generate batch report", 4.0),
        ("Visualise pipeline stages", 3.2),
    ]
    for label, y in cases:
        ax.add_patch(matplotlib.patches.Ellipse((5.0, y), 3.1, 0.62, facecolor="#ffffff", edgecolor="#2f5d8f", lw=1.3, zorder=3))
        ax.text(5.0, y, label, ha="center", va="center", fontsize=8.5)
        _arrow(ax, 1.9, y + 0.15, 3.5, y)
    # tesseract external actor
    _box(ax, 8.2, 4.3, 1.6, 1.6, "<<external>>\nTesseract\nOCR", fc="#f1ede6")
    _arrow_lr(ax, 6.8, 5.0, 8.1, 5.0)
    ax.text(7.45, 5.15, "invokes", fontsize=7.5)


def class_diagram(ax):
    ax.text(5, 9.7, "Component / class diagram (simplified)", ha="center", fontsize=12, fontweight="bold")
    def klass(x, y, w, h, name, members, methods, fc="#eef3fb"):
        _box(ax, x, y, w, h, "", fc=fc)
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=fc, edgecolor="#2f5d8f", lw=1.3, zorder=2))
        ax.text(x + w/2, y + h - 0.3, name, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.plot([x, x + w], [y + h - 0.55, y + h - 0.55], color="#2f5d8f", lw=0.8)
        ax.text(x + 0.15, y + h - 0.85, members, fontsize=6.6, va="center", linespacing=1.25)
        ax.plot([x, x + w], [y + h - 0.55 - 0.5, y + h - 0.55 - 0.5], color="#2f5d8f", lw=0.8, visible=False)
        ax.text(x + 0.15, y + 0.12, methods, fontsize=6.6, va="bottom", linespacing=1.3)
    klass(0.2, 5.0, 2.9, 3.2, "DocumentDetector", "max_dim : int\nmin_area_ratio: float\nmax_aspect : float", "detect(image) ->\n  (quad, score)")
    klass(3.4, 5.0, 2.9, 3.2, "ImagePreprocessor", "filters (CV ops)", "to_gray()\ndenoise()\ncompensate_illumination()\nedge_map()\nprepare_for_detection()")
    klass(6.6, 5.0, 3.2, 3.2, "PerspectiveTransformer", "output_aspect: float|None", "output_size(quad)\ntransform(image, quad)-> page")
    klass(0.2, 1.3, 2.9, 2.6, "DocumentEnhancer", "modes: bw | gray | otsu", "enhance(page, mode, deskew)")
    klass(3.4, 1.3, 2.9, 2.6, "OCREngine", "lang, psm, oem", "extract_text(image)\nextract_data(image)\ndraw_overlay()")
    klass(6.6, 1.3, 3.2, 2.6, "ScanExporter", "outputs", "export_pdf(scans)\nexport_text(text)")
    klass(0.2, 0.1, 9.6, 1.0, "ScanPipeline", "", "process(path, out_dir) -> ScanResult   |   batch(...) -> [ScanResult]", fc="#fdf6e3")
    ax.text(5.0, 0.55, "Note: ScanResult, ScanOptions, BatchProcessor, ReportGenerator, cli form the remaining application layer.", ha="center", fontsize=7.5, style="italic")


def sequence(ax):
    ax.text(5, 9.7, "Sequence diagram - scan a single document", ha="center", fontsize=12, fontweight="bold")
    parts = ["User/CLI", "ScanPipeline", "Loader", "Detector", "Enhancer", "OCREngine", "Exporter"]
    xs = [0.6, 2.0, 3.5, 5.0, 6.5, 8.0, 9.4]
    for x, name in zip(xs, parts):
        ax.plot([x, x], [0.8, 9.0], color="#999999", lw=1, linestyle=":", zorder=1)
        ax.text(x, 9.2, name, ha="center", va="center", fontsize=8)
    def msg(x1, y, x2, label, dashed=False):
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="-|>" if not dashed else "->", color="#222", lw=1.0, linestyle="--" if dashed else "-"), zorder=2)
        ax.text(min(x1, x2) + 0.05, y + 0.1, label, fontsize=7)
    y = 8.2
    msg(xs[0], y, xs[1], "scan(path, out_dir)")
    msg(xs[1], y - 0.55, xs[2], "load(path) -> image")
    msg(xs[2], y - 1.1, xs[1], "BGR image")
    msg(xs[1], y - 1.65, xs[3], "detect(image) -> quad")
    msg(xs[3], y - 2.2, xs[1], "(ordered corners, score)")
    msg(xs[1], y - 2.75, xs[1], "perspective warp")
    msg(xs[1], y - 3.4, xs[4], "enhance(page) -> binarized")
    msg(xs[4], y - 3.95, xs[1], "clean page")
    msg(xs[1], y - 4.5, xs[5], "extract_text(page)")
    msg(xs[5], y - 5.05, xs[1], "transcript + word boxes")
    msg(xs[1], y - 5.6, xs[6], "export_pdf / export_text")
    msg(xs[6], y - 6.15, xs[1], "output paths")


def er(ax):
    ax.text(5, 9.6, "Storage design - report entities (no relational DB)", ha="center", fontsize=12, fontweight="bold")
    def ent(x, y, w, h, name, rows):
        _box(ax, x, y, w, h, "", fc="#f6efe2")
        ax.add_patch(matplotlib.patches.Rectangle((x, y), w, h, facecolor="#f6efe2", edgecolor="#7a5c21", lw=1.3, zorder=2))
        ax.add_patch(matplotlib.patches.Rectangle((x, y + h - 0.5), w, 0.5, facecolor="#e4d6b8", edgecolor="#7a5c21", lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h - 0.25, name, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + 0.18, y + 0.25, rows, fontsize=7, va="bottom", linespacing=1.4)
    ent(0.3, 4.0, 3.4, 4.4, "ScanReport", "PK report_id : string\nsubmited_at : datetime\ntotal_files : int\nok_files : int\nfailed_files: int\nmean_cer : float\nmean_wer : float")
    ent(4.6, 4.0, 3.4, 4.4, "ScanResult", "PK stem : string\nOK : bool\nduration_s : float\ncer / wer : float|null\ninput_path : string\noutput_paths : list")
    ent(8.7, 6.1, 1.0, 2.3, "1", "#f6efe2")
    ent(0.3, 0.4, 3.4, 2.6, "StageTiming", "PK stage : string\nseconds : float\nstage_order : int")
    ent(4.6, 0.4, 3.4, 2.6, "OCRWord", "PK word_id : string\nword : string\nbbox : (l,t,w,h)\nconfidence: float")
    ax.plot([3.7, 4.5], [7.0, 7.0], color="#7a5c21", lw=1.2, zorder=1)
    ax.text(4.1, 7.15, "1..n", fontsize=7.5)
    ax.plot([3.7, 4.5], [2.0, 2.0], color="#7a5c21", lw=1.2)
    ax.text(4.1, 2.15, "1..n", fontsize=7.5)
    ax.text(0.4, 3.6, "report.json : nested records      report.md / .csv : flattened tables", fontsize=7.5, style="italic", color="#555555")


RENDERERS = [architecture, workflow, usecase, class_diagram, sequence, er]


def main() -> None:
    for renderer in RENDERERS:
        _render(renderer, f"{renderer.__name__}.png")


if __name__ == "__main__":
    main()