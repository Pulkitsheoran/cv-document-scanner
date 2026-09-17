"""Command-line interface for ScanVision.

Sub-commands
------------
scan       process a single document photo,
batch      process every photo in a folder (with parallel workers),
evaluate   process a folder and score OCR output against ground-truth text.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scanvision import __version__
from scanvision.config import ScanConfig
from scanvision.loader import DocumentLoader
from scanvision.models import ScanOptions
from scanvision.pipeline import ScanPipeline
from scanvision.report import ReportGenerator

LOG_FORMAT = "%(levelname)-7s [%(name)s] %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanvision",
        description="ScanVision - smart document scanner and OCR pipeline (OpenCV + Tesseract).",
    )
    parser.add_argument("--version", action="version", version=f"scanvision {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--config", default="config/pipeline.toml", help="TOML configuration file")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("input", help="image file or directory")
    common.add_argument("--out", default=None, help="output directory (default: config out_dir)")
    common.add_argument("--enhance", choices=["bw", "gray", "otsu"], default=None)
    common.add_argument("--no-ocr", action="store_true", help="skip OCR")
    common.add_argument("--no-pdf", action="store_true", help="skip PDF export")
    common.add_argument("--no-text", action="store_true", help="skip .txt export")
    common.add_argument("--no-deskew", action="store_true", help="disable text deskewing")
    common.add_argument("--visualize", action="store_true", help="save detection/stage visualisations")
    subcommands = parser.add_subparsers(dest="command", required=False)

    parser_scan = subcommands.add_parser("scan", help="scan a single photo", parents=[common])
    parser_scan.add_argument("--reference", default=None, help="ground-truth .txt file to score against")

    parser_batch = subcommands.add_parser("batch", help="scan every photo in a directory", parents=[common])
    parser_batch.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    parser_batch.add_argument("--workers", type=int, default=None, help="parallel workers (default: config)")

    parser_eval = subcommands.add_parser(
        "evaluate", help="scan a directory and score OCR quality vs ground truth", parents=[common]
    )
    parser_eval.add_argument("--recursive", action="store_true")
    parser_eval.add_argument("--workers", type=int, default=None)
    parser_eval.add_argument(
        "--ground-truth", default="data/samples/ground_truth", help="directory holding <stem>.txt references"
    )

    return parser


def _options_from(args: argparse.Namespace, config: ScanConfig, ocr_disabled_hint: str = "") -> ScanOptions:
    options = config.scan_options
    return ScanOptions(
        enhance_mode=args.enhance or options.enhance_mode,
        do_ocr=not args.no_ocr and options.do_ocr,
        export_pdf=not args.no_pdf and options.export_pdf,
        export_text=not args.no_text and options.export_text,
        deskew=not args.no_deskew and options.deskew,
        fix_aspect=options.fix_aspect,
        visualize=args.visualize or options.visualize,
    )


def _load_references(directory: Path) -> dict[str, str]:
    references: dict[str, str] = {}
    directory = Path(directory)
    if not directory.is_dir():
        return references
    for file_path in sorted(directory.glob("*.txt")):
        references[file_path.stem] = file_path.read_text(encoding="utf-8")
    return references


def run_scan(args: argparse.Namespace, config: ScanConfig, texts: dict[str, str] | None) -> int:
    pipeline = ScanPipeline(max_dim=config.max_dim, min_area_ratio=config.min_area_ratio, max_aspect=config.max_aspect)
    options = _options_from(args, config)
    out_dir = Path(args.out) if args.out else config.out_dir
    paths = DocumentLoader().discover(args.input)
    if not paths:
        logging.getLogger("scanvision").error("no supported images found for %s", args.input)
        return 2
    if args.command == "scan":
        results = [pipeline.process(paths[0], out_dir, options, _reference_for(paths[0], texts))]
        print(_format_result(results[0]))
    else:
        args.workers = args.workers or config.workers
        references = _load_references(Path(args.ground_truth)) if args.command == "evaluate" else {}
        results = pipeline.batch(paths, out_dir, options, workers=args.workers, reference_texts=references)
        report = ReportGenerator().write_markdown(results, out_dir / "report.md")
        ReportGenerator().write_csv(results, out_dir / "report.csv")
        ReportGenerator().write_json(results, out_dir / "report.json")
        print(_format_report(results))
        print(f"\nReports written: {report}")
    return 0 if all(r.ok for r in results) else 1


def _reference_for(path: Path, texts: dict[str, str] | None) -> str | None:
    """Pick a reference by file stem (without extension) from provided map,
    or look for a sibling ``<name>_gt.txt`` file."""
    if texts:
        return texts.get(path.stem)
    sibling = path.with_name(f"{path.stem}_gt.txt")
    if sibling.is_file():
        return sibling.read_text(encoding="utf-8")
    return None


def _format_result(result) -> str:
    lines = [f"{'file':<16} {result.stem}"]
    lines.append(f"{'status':<16} {'ok' if result.ok else 'FAILED: ' + (result.error or '')}")
    if result.ok:
        lines.append(f"{'time':<16} {result.duration:.2f}s")
        lines.append(f"{'CER / acc':<16} {result.cer if result.cer is not None else '-'} / {result.char_accuracy if result.char_accuracy is not None else '-'}")
        lines.append(f"{'WER / acc':<16} {result.wer if result.wer is not None else '-'} / {result.word_accuracy if result.word_accuracy is not None else '-'}")
        if result.output_paths:
            lines.append(f"{'outputs':<16} {', '.join(str(p) for p in result.output_paths)}")
    return "\n".join(lines)


def _format_report(results) -> str:
    summary = ReportGenerator.summary(results)
    lines = [f"Processed {summary['total']} file(s): {summary['ok']} ok, {summary['failed']} failed."]
    if summary["evaluated"]:
        lines.append(
            f"OCR quality: mean CER {summary['mean_cer']:.4f}, mean WER {summary['mean_wer']:.4f} "
            f"(across {summary['evaluated']} scored documents)"
        )
    else:
        lines.append("No ground truth provided - CER/WER not evaluated.")
    if summary["avg_duration_s"]:
        lines.append(f"Average pipeline time: {summary['avg_duration_s']:.3f}s per document")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)

    try:
        config = ScanConfig.from_file(Path(args.config)) if Path(args.config).exists() else ScanConfig()
    except Exception as exc:
        logging.getLogger("scanvision").error("configuration error: %s", exc)
        return 2

    args.command = args.command or "scan"
    return run_scan(args, config, None)


if __name__ == "__main__":
    sys.exit(main())