"""USGS splib07 AVIRIS-14 text spectra, resampled onto a destination wavelength grid.

Each spectrum file is one numeric value per line. Values below -1e30 are nodata.
The wavelength file name contains ``Wavelengths``. Mineral names come from
``split("_")[2]`` (Olivine, Calcite, and the rest of that library). Continuum
removal and L2 normalization are separate from loading.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from glob import glob

import numpy as np
from scipy.interpolate import interp1d

_NODATA = -1e30
_MIN_FINITE_FOR_NORM = 5


def load_usgs_vector(path: str | os.PathLike[str]) -> np.ndarray:
    """Load one USGS text spectrum or wavelength file as float64.

    Non-numeric lines are skipped. Values below -1e30 become NaN.
    """
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


def mineral_name(filename: str) -> str:
    """Mineral name from a splib07 filename, via ``split("_")[2]``."""
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[2]
    return base


def resample_spectrum(
    src_wvl: np.ndarray,
    src: np.ndarray,
    dst_wvl: np.ndarray,
) -> np.ndarray:
    """Linear resample onto ``dst_wvl`` using finite wavelength/value pairs only.

    Outside the source range the result is NaN (``bounds_error=False``).
    """
    src_wvl = np.asarray(src_wvl, dtype=np.float64).reshape(-1)
    src = np.asarray(src, dtype=np.float64).reshape(-1)
    destination = np.asarray(dst_wvl, dtype=np.float64)
    n = min(src_wvl.size, src.size)
    src_wvl = src_wvl[:n]
    src = src[:n]
    valid = np.isfinite(src_wvl) & np.isfinite(src)
    empty = np.full(destination.shape, np.nan, dtype=np.float64)
    if int(np.count_nonzero(valid)) < 2:
        return empty

    x = src_wvl[valid]
    y = src[valid]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    keep = np.empty(x.size, dtype=bool)
    keep[0] = True
    keep[1:] = x[1:] != x[:-1]
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return empty

    interpolator = interp1d(x, y, bounds_error=False, fill_value=np.nan)
    return np.asarray(interpolator(destination), dtype=np.float64)


def _wavelengths_nm(wavelengths: np.ndarray) -> np.ndarray:
    """Convert micrometre wavelengths to nanometres when the grid is below 100."""
    finite = wavelengths[np.isfinite(wavelengths)]
    if finite.size and float(np.max(finite)) < 100.0:
        return wavelengths * 1000.0
    return wavelengths


def load_library(
    folder: str | os.PathLike[str],
    dst_wavelengths_nm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load mineral spectra aligned to ``dst_wavelengths_nm``.

    AppleDouble files (``._*``) are skipped. Spectra are not continuum-removed
    and not L2-normalized.
    """
    folder = os.fspath(folder)
    text_files = [
        path
        for path in glob(os.path.join(folder, "*.txt"))
        if not os.path.basename(path).startswith("._")
    ]
    wavelength_files = [
        path for path in text_files if "Wavelengths" in os.path.basename(path)
    ]
    if not wavelength_files:
        raise FileNotFoundError(
            f"No USGS wavelength file (*Wavelengths*.txt) in {folder}"
        )

    wavelengths = _wavelengths_nm(load_usgs_vector(sorted(wavelength_files)[0]))
    destination = np.asarray(dst_wavelengths_nm, dtype=np.float64)
    library: dict[str, np.ndarray] = {}
    for path in sorted(text_files):
        if "Wavelengths" in os.path.basename(path):
            continue
        spectrum = load_usgs_vector(path)
        library[mineral_name(path)] = resample_spectrum(
            wavelengths, spectrum, destination
        )
    return library


def _l2_normalize(spectrum: np.ndarray) -> np.ndarray:
    """L2-normalize one spectrum. NaNs count as 0 in the norm only.

    Fewer than 5 finite samples, or a zero norm, yields an all-NaN spectrum.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    finite = np.isfinite(spectrum)
    if int(np.count_nonzero(finite)) < _MIN_FINITE_FOR_NORM:
        return np.full(spectrum.shape, np.nan, dtype=np.float64)
    filled = np.where(finite, spectrum, 0.0)
    norm = float(np.linalg.norm(filled))
    if norm == 0.0:
        return np.full(spectrum.shape, np.nan, dtype=np.float64)
    return spectrum / norm


def apply_library_continuum(
    library: dict[str, np.ndarray],
    wavelengths: np.ndarray,
    continuum_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    """Continuum-remove each spectrum, then L2-normalize.

    ``continuum_fn(spectra_2d, wavelengths)`` receives one spectrum stacked as
    a 1-row matrix and returns a 2-D array of the same layout.
    """
    removed: dict[str, np.ndarray] = {}
    for name, spectrum in library.items():
        row = np.asarray(spectrum, dtype=np.float64).reshape(1, -1).copy()
        continuum = np.asarray(continuum_fn(row, wavelengths), dtype=np.float64)
        sample = continuum[0] if continuum.ndim >= 2 else continuum
        removed[name] = _l2_normalize(sample)
    return removed
