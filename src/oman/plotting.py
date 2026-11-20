"""Matplotlib helpers that save PNG figures and close them.

These functions never call ``plt.show``. Each one creates the parent directory
of ``path`` when it is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def _prepare_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _save(fig: plt.Figure, path: str | Path, dpi: int) -> None:
    out = _prepare_path(path)
    try:
        fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")
    finally:
        plt.close(fig)


def _finite_percentile_limits(
    image: np.ndarray,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """Fill missing limits with the 2nd and 98th percentiles of finite values."""
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        low, high = 0.0, 1.0
    else:
        low_value, high_value = np.percentile(finite, (2.0, 98.0))
        low, high = float(low_value), float(high_value)

    if vmin is None:
        vmin = low
    if vmax is None:
        vmax = high
    low_limit = float(vmin)
    high_limit = float(vmax)
    if low_limit == high_limit:
        high_limit = low_limit + 1.0
    return low_limit, high_limit


def save_scalar_map(
    path: str | Path,
    image: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save a single-band map. NaNs are transparent; stretch uses finite percentiles."""
    values = np.asarray(image, dtype=float)
    masked = np.ma.masked_invalid(values)
    low, high = _finite_percentile_limits(values, vmin, vmax)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad((0.0, 0.0, 0.0, 0.0))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.imshow(
        masked,
        cmap=colormap,
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")
    _save(fig, path, dpi=160)


def save_rgb(path: str | Path, rgb: np.ndarray, title: str) -> None:
    """Save an RGB image with samples in ``0..1`` and shape ``(rows, cols, 3)``."""
    image = np.asarray(rgb, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (rows, cols, 3)")

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.imshow(np.clip(image, 0.0, 1.0), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    _save(fig, path, dpi=160)


def save_class_map(
    path: str | Path,
    class_map: np.ndarray,
    names: Sequence[str],
    title: str,
) -> None:
    """Save an integer class map. ``-1`` is nodata; colors cycle through tab20."""
    labels = [str(name) for name in names]
    n_classes = len(labels)
    classes = np.asarray(class_map)
    numeric = classes.astype(float)
    nodata = ~np.isfinite(numeric) | (numeric == -1)
    masked = np.ma.masked_where(nodata, numeric)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    if n_classes == 0:
        ax.set_title(title)
        ax.axis("off")
        _save(fig, path, dpi=160)
        return

    base = plt.get_cmap("tab20")
    colors = [base(i % base.N) for i in range(n_classes)]
    colormap = ListedColormap(colors)
    colormap.set_bad((0.0, 0.0, 0.0, 0.0))
    norm = BoundaryNorm(np.arange(-0.5, n_classes + 0.5), n_classes)
    image = ax.imshow(
        masked,
        cmap=colormap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(n_classes))
    colorbar.ax.set_yticklabels(labels)
    _save(fig, path, dpi=160)


def save_spectra(
    path: str | Path,
    wavelengths: np.ndarray,
    spectra: dict[str, np.ndarray],
    title: str,
    ylabel: str,
) -> None:
    """Save one line per spectrum against the shared wavelength axis."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    x = np.asarray(wavelengths, dtype=float)
    for name, spectrum in spectra.items():
        ax.plot(x, np.asarray(spectrum, dtype=float), label=str(name))
    ax.set_title(title)
    ax.set_xlabel("Wavelength")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if spectra:
        ax.legend()
    _save(fig, path, dpi=140)


def save_variance(path: str | Path, ratio: np.ndarray, title: str) -> None:
    """Save a bar chart of explained variance ratio, labeled PC1, PC2, ..."""
    values = np.asarray(ratio, dtype=float).ravel()
    labels = [f"PC{index}" for index in range(1, values.size + 1)]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    if values.size > 12:
        ax.tick_params(axis="x", labelrotation=45)
    _save(fig, path, dpi=160)
