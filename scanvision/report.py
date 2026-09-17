"""Batch report generation (JSON / Markdown / CSV)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from scanvision.exceptions import ExportError
from scanvision.models import ScanResult


class ReportGenerator:
    """Aggregates :class:`ScanResult` objects into human- and machine-readable reports."""

    @staticmethod
    def summary(results: Iterable[ScanResult]) -> dict:
        rows = [r for r in results if r.ok]
        failures = [r for r in results if not r.ok]
        cers = [r.cer for r in rows if r.cer is not None]
        wers = [r.wer for r in rows if r.wer is not None]
        return {
            "total": len(list(results)),
            "ok": len(rows),
            "failed": len(failures),
            "failure_detail": {r.stem: r.error for r in failures},
            "avg_duration_s": round(sum(r.duration for r in rows) / len(rows), 3) if rows else None,
            "evaluated": len(cers),
            "mean_cer": round(sum(cers) / len(cers), 4) if cers else None,
            "mean_wer": round(sum(wers) / len(wers), 4) if wers else None,
        }

    @staticmethod
    def write_json(results: Iterable[ScanResult], out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": ReportGenerator.summary(results),
            "results": [
                {
                    k: (v.tolist() if isinstance(v, list) and hasattr(v, "tolist") else v)
                    for k, v in _result_dict(r).items()
                }
                for r in results
            ],
        }
        try:
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"failed to write report {out_path}: {exc}") from exc
        return out_path

    @staticmethod
    def write_markdown(results: Iterable[ScanResult], out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(results, key=lambda r: r.stem)
        audio = "OK" if not any(not r.ok for r in rows) else "HAS FAILURES"
        lines = [
            "# ScanVision batch report\n",
            f"- Generated run summary: **{audio}**",
            f"- Inputs processed: **{len(rows)}**",
        ]
        table = ["| file | ok | time (s) | CER | WER | outputs | error |",
                 "| --- | :---: | ---: | ---: | ---: | --- | --- |"]
        for r in rows:
            time_s = f"{r.duration:.2f}"
            cer = f"{r.cer:.3f}" if r.cer is not None else "-"
            wer = f"{r.wer:.3f}" if r.wer is not None else "-"
            outs = ", ".join(p.name for p in r.output_paths) or "-"
            err = (r.error or "-").replace("|", "/")
            table.append(
                f"| {r.stem} | {'yes' if r.ok else 'no'} | {time_s} | {cer} | {wer} | {outs} | {err} |"
            )
        lines += table + [""]
        try:
            out_path.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            raise ExportError(f"failed to write report {out_path}: {exc}") from exc
        return out_path

    @staticmethod
    def write_csv(results: Iterable[ScanResult], out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(results, key=lambda r: r.stem)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "ok", "duration_s", "cer", "wer", "char_acc", "word_acc", "error"])
            for r in rows:
                writer.writerow(
                    [
                        r.stem,
                        r.ok,
                        round(r.duration, 3),
                        "" if r.cer is None else round(r.cer, 4),
                        "" if r.wer is None else round(r.wer, 4),
                        "" if r.char_accuracy is None else round(r.char_accuracy, 4),
                        "" if r.word_accuracy is None else round(r.word_accuracy, 4),
                        r.error or "",
                    ]
                )
        return out_path


def _result_dict(result: ScanResult) -> dict:
    self_dict = result.__dict__.copy()
    self_dict["input_path"] = str(result.input_path)
    self_dict["words"] = [w.__dict__ for w in result.words]
    if result.quad is not None:
        self_dict["quad"] = [[float(v) for v in point] for point in result.quad.tolist()]
    self_dict["output_paths"] = [str(p) for p in result.output_paths]
    return self_dict