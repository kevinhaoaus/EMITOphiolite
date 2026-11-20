"""USGS splib07 AVIRIS-14 text spectra, aligned to a wavelength grid."""

from __future__ import annotations

import os
from glob import glob

import numpy as np

_NODATA = -1e30


def _read_column(path: str) -> np.ndarray:
    """One numeric value per line. Header text is skipped. Sentinel fill becomes NaN."""
    values: list[float] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            token = line.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
    vector = np.asarray(values, dtype=np.float64)
    vector[vector < _NODATA] = np.nan
    return vector


def _mineral_name(filename: str) -> str | None:
    """Token after ``AV14`` (Calcite, Olivine, ...). Wavelength files return None."""
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    try:
        index = next(i for i, part in enumerate(parts) if part.upper() == "AV14")
    except StopIteration:
        return None
    if index + 1 >= len(parts):
        return None
    name = parts[index + 1]
    if name.lower().startswith("wavelength"):
        return None
    return name


def _as_nm(wavelengths: np.ndarray) -> np.ndarray:
    """Micrometre grids (max < 100) are converted to nanometres."""
    finite = wavelengths[np.isfinite(wavelengths)]
    if finite.size and float(np.max(finite)) < 100.0:
        return wavelengths * 1000.0
    return wavelengths


def _linear_resample(
    src_wvl: np.ndarray,
    src: np.ndarray,
    dst_wvl: np.ndarray,
) -> np.ndarray:
    """Linear interpolation. Samples outside the finite source range are NaN."""
    destination = np.asarray(dst_wvl, dtype=np.float64)
    n = min(src_wvl.size, src.size)
    src_wvl = src_wvl[:n]
    src = src[:n]
    valid = np.isfinite(src_wvl) & np.isfinite(src)
    if int(np.count_nonzero(valid)) < 2:
        return np.full(destination.shape, np.nan, dtype=np.float64)

    x = src_wvl[valid]
    y = src[valid]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    keep = np.empty(x.size, dtype=bool)
    keep[0] = True
    keep[1:] = x[1:] > x[:-1]
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return np.full(destination.shape, np.nan, dtype=np.float64)
    return np.interp(destination, x, y, left=np.nan, right=np.nan)


def _preprocess():
    try:
        from .preprocess import continuum_remove, smooth_spectra
    except ImportError:
        try:
            from preprocess import continuum_remove, smooth_spectra
        except ImportError:
            from src.preprocess import continuum_remove, smooth_spectra
    return continuum_remove, smooth_spectra


def load_library(
    usgs_dir: str,
    wavelengths_nm: np.ndarray,
    *,
    continuum: bool = False,
    smooth: bool = False,
) -> dict:
    """Return {mineral_name: spectrum aligned to wavelengths_nm}.

    Linear interpolation. Bands outside the library wavelength range are NaN.
    ``continuum`` and ``smooth`` call ``continuum_remove`` and ``smooth_spectra``
    on a ``(1, B)`` array. L2 normalization is left to ``l2_normalize``.
    """
    target = np.asarray(wavelengths_nm, dtype=np.float64)
    text_files = [
        path
        for path in glob(os.path.join(os.fspath(usgs_dir), "*.txt"))
        if not os.path.basename(path).startswith("._")
    ]
    wavelength_files = [
        path for path in text_files if "wavelength" in os.path.basename(path).lower()
    ]
    if not wavelength_files:
        raise FileNotFoundError(f"No USGS wavelength file in {usgs_dir}")
    library_wvl = _as_nm(_read_column(sorted(wavelength_files)[0]))

    continuum_remove = smooth_spectra = None
    if continuum or smooth:
        continuum_remove, smooth_spectra = _preprocess()

    library: dict[str, np.ndarray] = {}
    for path in sorted(text_files):
        name = _mineral_name(path)
        if name is None:
            continue
        spectrum = _linear_resample(library_wvl, _read_column(path), target)
        if smooth or continuum:
            row = spectrum.reshape(1, -1)
            if smooth:
                row = np.asarray(smooth_spectra(row), dtype=np.float64)
            if continuum:
                row = np.asarray(continuum_remove(row, target), dtype=np.float64)
            spectrum = row.reshape(target.shape)
        library[name] = spectrum
    return library


def l2_normalize(spec: np.ndarray) -> np.ndarray:
    """Divide finite samples by their L2 norm. NaNs stay NaN."""
    out = np.array(np.asarray(spec, dtype=np.float64), copy=True)
    finite = np.isfinite(out)
    if not finite.any():
        return out
    norm = float(np.linalg.norm(out[finite]))
    if norm > 0.0:
        out[finite] /= norm
    return out
