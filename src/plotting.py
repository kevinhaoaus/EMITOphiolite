"""Headless matplotlib savers. Agg only; these functions never call plt.show."""

from __future__ import annotations

from pathlib import Path

import matplotlib

try:
    matplotlib.use("Agg", force=True)
except TypeError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

_DARK = (0.04, 0.02, 0.06, 1.0)


def _prepare(path) -> Path:
    out = Path(path)
    if out.parent.parts:
        out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _finish(fig: plt.Figure, path) -> None:
    out = _prepare(path)
    fig.tight_layout()
    try:
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def save_rgb(path, rgb, title) -> None:
    """Save an RGB image. A 2–98 stretch is already expected inside ``rgb``."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("rgb must have shape (rows, cols, 3) or (rows, cols, 4)")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.imshow(image, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    _finish(fig, path)


def save_score(path, image2d, title, cmap="magma") -> None:
    """Save a 2-D score image. Non-finite samples are a dark masked color."""
    values = np.asarray(image2d, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("image2d must be 2-D")
    masked = np.ma.masked_invalid(values)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(_DARK)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor(_DARK)
    image = ax.imshow(masked, cmap=colormap, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")
    _finish(fig, path)


def save_labels(path, labels2d, names, title) -> None:
    """Save an integer label map. ``names`` aligns with ids ``0 .. len(names)-1``.

    Negative and non-finite labels are drawn in the same dark masked color
    used for non-finite scores.
    """
    labels = [str(name) for name in names]
    classes = np.asarray(labels2d)
    if classes.ndim != 2:
        raise ValueError("labels2d must be 2-D")
    n_classes = len(labels)
    invalid = ~np.isfinite(classes.astype(np.float64)) | (classes < 0)
    shown = np.ma.array(classes.astype(np.float64), mask=invalid)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor(_DARK)
    if n_classes == 0:
        ax.set_title(title)
        ax.axis("off")
        _finish(fig, path)
        return

    base = plt.get_cmap("tab20")
    colormap = ListedColormap([base(i % base.N) for i in range(n_classes)])
    colormap.set_bad(_DARK)
    colormap.set_under(_DARK)
    colormap.set_over(_DARK)
    norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5), n_classes)
    image = ax.imshow(shown, cmap=colormap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(n_classes), fraction=0.046, pad=0.04)
    colorbar.ax.set_yticklabels(labels)
    _finish(fig, path)
