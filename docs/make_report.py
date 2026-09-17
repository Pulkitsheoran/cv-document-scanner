"""Render docs/report.md into docs/report.pdf.

The parser understands a constrained markdown subset produced by hand for the
report:

    # Heading          -> new PDF page, H1 section title
    ## Sub-heading     -> H2 sub-title
    - item             -> bullet paragraph
    | a | b |          -> table rows (`|---|` separator skipped,
                          continuation lines merged into the last cell)
    ![caption](path)   -> centred image with caption (paths relative to docs/)
    blank lines        -> paragraph breaks

HTML-ish bold (**x**) inside body text is honoured; `code` markers are
rendered as regular text. The cover page is generated programmatically.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib
from fpdf import FPDF
from fpdf.enums import XPos, YPos, TableCellFillMode

DOCS = Path(__file__).resolve().parent
REPORT_MD = DOCS / "report.md"
OUT_PDF = DOCS / "report.pdf"

_FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
_FONT_BOLD = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")
_FONT_MONO = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf")

STUDENT_NAME = "[Student Name]"
ROLL_NO = "[Roll Number]"
DATE = "17 September 2026"
COURSE = "Computer Vision - VITyarthi: Build Your Own Project"
TITLE = "ScanVision"
SUBTITLE = "Smart Document Scanner & OCR using Classical Computer Vision"


class ReportPdf(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.add_font("DejaVu", "", _FONT)
        self.add_font("DejaVu", "B", _FONT_BOLD)
        self.add_font("DejaVuMono", "", _FONT_MONO)
        self.set_auto_page_break(auto=True, margin=14)
        self.set_margins(14, 14, 14)


def _no_markup(text: str) -> str:
    return text.replace("`", "").replace("&amp;", "&").replace("*", "").strip()


def _runs(text: str) -> list[tuple[str, bool]]:
    """Split text into (chunk, is_bold) runs honouring **bold** markers."""
    text = text.replace("&amp;", "&").replace("*", "") if "*" in text and "**" not in text else text
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    runs: list[tuple[str, bool]] = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            runs.append((part[2:-2], True))
        elif part:
            runs.append((part.replace("`", ""), False))
    return runs or [("", False)]


def _paragraph(pdf: ReportPdf, text: str, size: float = 9.5) -> None:
    pdf.set_font("DejaVu", "", size)
    runs = _runs(text)
    for chunk, bold in runs:
        pdf.set_font("DejaVu", "B" if bold else "", size)
        pdf.write(size * 0.55, chunk)
    pdf.ln(size * 0.85)


def _bullets(pdf: ReportPdf, items: Iterable[str]) -> None:
    for item in items:
        pdf.set_font("DejaVu", "", 9.5)
        x = pdf.get_x()
        pdf.cell(5, 5.5, "- ")
        _paragraph(pdf, item)
        pdf.set_x(x)


def _heading1(pdf: ReportPdf, text: str) -> None:
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 15)
    pdf.set_text_color(30, 60, 110)
    pdf.multi_cell(0, 8, _no_markup(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _heading2(pdf: ReportPdf, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("DejaVu", "B", 11.5)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6.5, _no_markup(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)


def _table(pdf: ReportPdf, raw_rows: list[list[str]]) -> None:
    rows = [row for row in raw_rows if row and not (len(row) >= 1 and re.match(r"^:?-{2,}:?$", row[0].strip()))]
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    col_w = (pdf.epw) / ncols
    with pdf.table(
        col_widths=[col_w] * ncols,
        borders_layout="ALL",
        first_row_as_headings=True,
        line_height=5.0,
        text_align="LEFT",
    ) as table:
        for row in rows:
            table_row = table.row()
            for cell_text in row:
                chunk = _no_markup(cell_text).strip()
                table_row.cell(chunk)


def _image(pdf: ReportPdf, caption: str, path: Path) -> None:
    from PIL import Image

    image_path = DOCS / path
    target_w = min(pdf.epw, 170)
    with Image.open(image_path) as im:
        width, height = im.size
    scale = target_w / width
    height_px = height * scale
    usable = pdf.h - pdf.b_margin - pdf.get_y()
    if height_px > usable - 20:
        pdf.add_page()
    pdf.image(str(image_path), w=target_w)
    pdf.ln(1)
    pdf.set_font("DejaVu", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, caption, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)


def _cover(pdf: ReportPdf) -> None:
    pdf.add_page()
    pdf.ln(52)
    pdf.set_font("DejaVu", "B", 40)
    pdf.set_text_color(30, 60, 110)
    pdf.cell(0, 16, TITLE, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, SUBTITLE, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(38)
    pdf.set_font("DejaVu", "", 11.5)
    pdf.set_text_color(0, 0, 0)
    meta = [
        f"Course: {COURSE}",
        "Student: " + STUDENT_NAME,
        "Roll No: " + ROLL_NO,
        f"Date: {DATE}",
        "Subject: Computer Vision",
    ]
    for line in meta:
        pdf.cell(0, 8, line, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(16)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Project Report (PDF for portal submission)", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def build() -> Path:
    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    pdf = ReportPdf()
    _cover(pdf)

    first_h1_seen = False
    table_buffer: list[list[str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # table continuation lines
        if table_buffer and stripped and not stripped.startswith("|"):
            if stripped.startswith("#") or stripped.startswith("![") or stripped.startswith("- "):
                _table(pdf, table_buffer)
                table_buffer = []
            else:
                table_buffer[-1][-1] += " " + _no_markup(stripped)
                i += 1
                continue

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            table_buffer.append(cells)
            i += 1
            continue
        if table_buffer:
            _table(pdf, table_buffer)
            table_buffer = []

        if stripped.startswith("# "):
            heading = stripped[2:]
            if not first_h1_seen:
                first_h1_seen = True  # the markdown title line duplicates the cover
            else:
                _heading1(pdf, heading)
        elif stripped.startswith("## "):
            _heading2(pdf, stripped[3:])
        elif stripped.startswith("- "):
            items = [stripped[2:]]
            # absorb a following bold-only or short bullet if it is also a list item
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("- "):
                i += 1
                items.append(lines[i].strip()[2:])
            _bullets(pdf, items)
        elif stripped.startswith("![") and "](" in stripped:
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                _image(pdf, match.group(1), Path(match.group(2)))
        elif stripped == "---":
            pass  # visual separators are meaningless in PDF
        elif stripped:
            _paragraph(pdf, stripped)
        i += 1
    if table_buffer:
        _table(pdf, table_buffer)

    pdf.set_creator("ScanVision report builder (fpdf2)")
    pdf.output(str(OUT_PDF))
    return OUT_PDF


if __name__ == "__main__":
    out = build()
    print(f"wrote {out} ({out.stat().st_size} bytes)")