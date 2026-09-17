"""Diagnostic visualisation: stage grids and quadrangle overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402


class PipelineVisualizer:
    """Composes annotated images that show the pipeline working end to end.

    These visuals are used for the demo screenshots, tests and the project
    report (bright green quadrangle overlay + a stage-grid montage).
    """

    @staticmethod
    def overlay_quad(image: np.ndarray, quad: np.ndarray, out_path: Path, color=(0, 255, 140), thickness: int = 4) -> Path:
        """Draw the detected document quadrangle (BGR img) and save PNG."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = image.copy()
        quad = np.asarray(quad, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(overlay, [quad], isClosed=True, color=color, thickness=thickness)
        for point in quad:
            cv2.circle(overlay, tuple(point), 8, color, thickness=-1)
        cv2.imwrite(str(out_path), overlay)
        return out_path

    @staticmethod
    def stage_grid(stages: dict[str, np.ndarray], out_path: Path, title: str = "ScanVision pipeline") -> Path:
        """Save a labelled grid of pipeline stage outputs as a PNG montage.

        Uses the object-oriented matplotlib API (no global pyplot state) so the
        visualizer is safe to call from multiple worker threads.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        items = list(stages.items())
        cols = 3
        rows = int(np.ceil(len(items) / cols))
        figure = Figure(figsize=(cols * 4.2, rows * 3.4), dpi=140)
        axes = [figure.add_subplot(rows, cols, index) for index in range(1, len(items) + 1)]
        for axis, (name, image) in zip(axes, items):
            display = image[..., ::-1] if image.ndim == 3 else image
            axis.imshow(display, cmap="gray")
            axis.set_title(name, fontsize=11)
            axis.axis("off")
        figure.suptitle(title, fontsize=14, y=0.995)
        figure.tight_layout(rect=[0, 0, 1, 0.97])
        figure.savefig(out_path, bbox_inches="tight", facecolor="white")
        return out_path