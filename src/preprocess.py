"""EMIT reflectance preprocessing.

Selects VNIR/SWIR atmospheric windows, drops shadowed, vegetated, and
unusable pixels, then optionally smooths and removes the upper continuum.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

_SAVGOL_WINDOW = 11
_SAVGOL_POLY = 2
_MIN_HULL_SAMPLES = 5
_MIN_STD = 1e-6


def good_band_mask(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Boolean mask of wavelengths inside the EMIT atmospheric windows.

    Keeps the open intervals (450, 1320), (1460, 1780), and (1970, 2450) nm.
    Those windows sit inside the VNIR and SWIR clear regions and are slightly
    tighter than the older (450, 1340) / (1450, 1800) / (1950, 2450) cuts.
    Endpoints and every other wavelength are excluded. Non-finite wavelengths
    are excluded.

    Parameters
    ----------
    wavelengths_nm : array
        Band centers in nanometers, shape ``(B,)``.

    Returns
    -------
    ndarray of bool, shape ``(B,)``
    """
    w = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    return (
        ((w > 450.0) & (w < 1320.0))
        | ((w > 1460.0) & (w < 1780.0))
        | ((w > 1970.0) & (w < 2450.0))
    )


def apply_good_bands(cube, wavelengths_nm):
    """Drop bands outside :func:`good_band_mask`.

    Parameters
    ----------
    cube : array, shape ``(H, W, B)``
        Reflectance cube whose last axis lines up with ``wavelengths_nm``.
    wavelengths_nm : array, shape ``(B,)``
        Band centers in nanometers.

    Returns
    -------
    cube_kept : array, shape ``(H, W, B_kept)``
    wavelengths_kept : array, shape ``(B_kept,)``
    """
    wave = np.asarray(wavelengths_nm).reshape(-1)
    mask = good_band_mask(wave)
    return cube[:, :, mask], wave[mask]


def smooth_spectra(spectra: np.ndarray) -> np.ndarray:
    """Savitzky–Golay smooth along the last axis (window 11, polynomial 2).

    Eligible rows (at least 11 finite samples) are smoothed together with
    ``scipy.signal.savgol_filter(..., axis=-1)``. If that call raises, each
    eligible row is filtered on its own. A row shorter than the window, or with
    fewer than 11 finite samples, is left unchanged. A row the filter cannot
    smooth without writing non-finite values over finite samples is left
    unchanged as well.

    Parameters
    ----------
    spectra : array, shape ``(..., B)``
        Reflectance spectra. The band axis is last.

    Returns
    -------
    ndarray
        Smoothed spectra, same shape as ``spectra``. Floating inputs keep their
        dtype; other inputs are returned as float64.
    """
    original = np.asarray(spectra)
    if np.issubdtype(original.dtype, np.floating):
        out_dtype = original.dtype
    else:
        out_dtype = np.dtype(np.float64)

    if original.shape[-1] < _SAVGOL_WINDOW or original.size == 0:
        return np.array(original, dtype=out_dtype, copy=True)

    work = np.array(original, dtype=np.float64, copy=True)
    flat = work.reshape(-1, work.shape[-1])
    eligible = np.isfinite(flat).sum(axis=1) >= _SAVGOL_WINDOW
    if not np.any(eligible):
        return work.astype(out_dtype, copy=False)

    rows = np.flatnonzero(eligible)

    def _restore_smeared(before: np.ndarray) -> None:
        smeared = np.isfinite(before) & ~np.isfinite(flat[rows])
        bad = np.any(smeared, axis=1)
        if np.any(bad):
            flat[rows[bad]] = before[bad]

    before = flat[rows].copy()
    try:
        flat[rows] = savgol_filter(
            before, _SAVGOL_WINDOW, _SAVGOL_POLY, axis=-1, mode="interp"
        )
        _restore_smeared(before)
    except (ValueError, FloatingPointError):
        flat[rows] = before
        for i in rows:
            try:
                smoothed = savgol_filter(
                    flat[i], _SAVGOL_WINDOW, _SAVGOL_POLY, mode="interp"
                )
            except (ValueError, FloatingPointError):
                continue
            if np.any(np.isfinite(flat[i]) & ~np.isfinite(smoothed)):
                continue
            flat[i] = smoothed
    return work.astype(out_dtype, copy=False)


def _turn(x: np.ndarray, y: np.ndarray, origin: int, left: int, right: int) -> float:
    """Cross product of (left - origin) with (right - origin).

    Positive when ``right`` is a left turn from the directed segment
    ``origin -> left``.
    """
    return (x[left] - x[origin]) * (y[right] - y[origin]) - (
        y[left] - y[origin]
    ) * (x[right] - x[origin])


def _upper_hull(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper convex hull of strictly increasing ``x``.

    Andrew's monotone chain, walked from the right edge toward the left.
    A step that is not a left turn is popped, so the chain that remains is
    the upper envelope. Collinear points are dropped.
    """
    chain: list[int] = []
    for i in range(x.shape[0] - 1, -1, -1):
        while len(chain) >= 2 and _turn(x, y, chain[-2], chain[-1], i) <= 0.0:
            chain.pop()
        chain.append(i)
    chain.reverse()
    idx = np.asarray(chain, dtype=np.int64)
    return x[idx], y[idx]


def _continuum_one(spec: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Continuum-remove one spectrum. Same length as ``spec``."""
    out = np.full(spec.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(spec) & np.isfinite(wavelengths_nm)
    if int(np.count_nonzero(usable)) < _MIN_HULL_SAMPLES:
        return out

    x = wavelengths_nm[usable]
    y = spec[usable]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    fresh = np.empty(x.shape[0], dtype=bool)
    fresh[0] = True
    fresh[1:] = x[1:] != x[:-1]
    starts = np.flatnonzero(fresh)
    x = x[starts]
    y = np.maximum.reduceat(y, starts)
    if x.shape[0] < 2 or np.any(np.diff(x) <= 0.0):
        return out

    hull_x, hull_y = _upper_hull(x, y)
    if hull_x.shape[0] < 2 or np.any(np.diff(hull_x) <= 0.0):
        return out

    query = wavelengths_nm[usable]
    continuum = np.interp(query, hull_x, hull_y)
    good = np.isfinite(continuum) & (continuum > 0.0)
    removed = np.full(query.shape, np.nan, dtype=np.float64)
    np.divide(spec[usable], continuum, out=removed, where=good)
    out[usable] = removed
    return out


def continuum_remove(spectra: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Divide each spectrum by its upper convex-hull continuum.

    The continuum is the upper envelope: a monotone chain walked from long
    wavelength to short wavelength, keeping left turns, then linearly
    interpolated with ``numpy.interp``. Samples where the continuum is not
    finite or is ``<= 0`` become NaN. Non-finite samples are ignored while the
    hull is built and stay NaN in the output. A spectrum with fewer than five
    finite samples is returned as all NaN.

    Parameters
    ----------
    spectra : array, shape ``(..., B)``
        Reflectance. One spectrum per leading index, bands on the last axis.
    wavelengths_nm : array, shape ``(B,)``
        Band centers in nanometers, aligned with the last axis.

    Returns
    -------
    ndarray
        Continuum-removed spectra, same shape as ``spectra``. Floating inputs
        keep their dtype; other inputs are returned as float64.
    """
    original = np.asarray(spectra)
    if np.issubdtype(original.dtype, np.floating):
        out_dtype = original.dtype
    else:
        out_dtype = np.dtype(np.float64)

    wave = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    spec = np.array(original, dtype=np.float64, copy=True)
    if spec.shape[-1] != wave.shape[0]:
        raise ValueError("wavelengths_nm must match the last axis of spectra")

    flat = spec.reshape(-1, spec.shape[-1])
    removed = np.empty_like(flat)
    for i in range(flat.shape[0]):
        removed[i] = _continuum_one(flat[i], wave)
    return removed.reshape(spec.shape).astype(out_dtype, copy=False)


def prepare_spectra(
    chunk,
    wavelengths_nm,
    *,
    continuum=True,
    smooth=True,
    shadow_percentile=8.0,
    ndvi_max=0.25,
    shadow_threshold=None,
):
    """Turn a reflectance chunk into spectra ready for mineral mapping.

    Parameters
    ----------
    chunk : array, shape ``(H, W, B)``
        Reflectance. ``wavelengths_nm`` aligns with the band axis. The input
        array is not modified.
    wavelengths_nm : array, shape ``(B,)``
        Band centers in nanometers.
    continuum : bool, optional
        Remove the upper convex-hull continuum after masking. Default True.
    smooth : bool, optional
        Savitzky–Golay smooth after masking and before continuum removal.
        Default True.
    shadow_percentile : float, optional
        Pixels whose mean reflectance past 2000 nm is below this percentile of
        the finite pixels in the chunk are dropped. Default 8.
    ndvi_max : float, optional
        Pixels with NDVI above this value are dropped. NDVI uses the bands
        nearest 850 nm and 660 nm. Default 0.25.
    shadow_threshold : float, optional
        Precomputed SWIR-brightness cutoff. When given, it replaces the
        per-chunk ``shadow_percentile`` so that a scene processed in row chunks
        uses one scene-wide threshold (see :func:`swir_brightness`).

    Returns
    -------
    spectra : float32 array, shape ``(N, B)``
        Kept pixels, one spectrum per row.
    pixel_index : int32 array, shape ``(N, 2)``
        Row and column of each kept pixel inside ``chunk``.

    A pixel is dropped when any band is non-finite, when the spectrum standard
    deviation is below ``1e-6``, when it falls in the SWIR shadow tail, or when
    its NDVI exceeds ``ndvi_max``. Smoothing and continuum removal run only on
    the pixels that remain.
    """
    cube = np.asarray(chunk)
    if cube.ndim != 3:
        raise ValueError("chunk must have shape (h, w, bands)")
    height, width, bands = cube.shape
    wave = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    if wave.shape[0] != bands:
        raise ValueError("wavelengths_nm must match chunk.shape[-1]")

    empty_spectra = np.empty((0, bands), dtype=np.float32)
    empty_index = np.empty((0, 2), dtype=np.int32)
    if height == 0 or width == 0 or bands == 0:
        return empty_spectra, empty_index

    finite = np.isfinite(cube).all(axis=-1)
    if not np.any(finite):
        return empty_spectra, empty_index

    deviation = np.std(cube, axis=-1)
    swir = wave > 2000.0
    shadow = np.zeros((height, width), dtype=bool)
    if np.any(swir) and np.any(finite):
        brightness = np.mean(cube[:, :, swir], axis=-1)
        if shadow_threshold is None:
            threshold = np.percentile(brightness[finite], shadow_percentile)
        else:
            threshold = float(shadow_threshold)
        shadow = np.isfinite(brightness) & (brightness < threshold)

    red = cube[:, :, int(np.argmin(np.abs(wave - 660.0)))]
    nir = cube[:, :, int(np.argmin(np.abs(wave - 850.0)))]
    denom = nir + red
    ndvi = np.full((height, width), np.nan, dtype=np.float64)
    ok = np.isfinite(red) & np.isfinite(nir) & (np.abs(denom) > 1e-10)
    np.divide(nir - red, denom, out=ndvi, where=ok)
    vegetation = ndvi > ndvi_max

    keep = finite & (deviation >= _MIN_STD) & ~shadow & ~vegetation
    rows, cols = np.nonzero(keep)
    if rows.size == 0:
        return empty_spectra, empty_index

    pixel_index = np.empty((rows.shape[0], 2), dtype=np.int32)
    pixel_index[:, 0] = rows
    pixel_index[:, 1] = cols

    spectra = np.asarray(cube[rows, cols, :], dtype=np.float64)
    if smooth:
        spectra = smooth_spectra(spectra)
    if continuum:
        spectra = continuum_remove(spectra, wave)
    return np.asarray(spectra, dtype=np.float32), pixel_index


def swir_brightness(chunk, wavelengths_nm) -> np.ndarray:
    """Mean reflectance past 2000 nm per pixel, NaN where any band is non-finite.

    This is the brightness :func:`prepare_spectra` thresholds for shadow. A
    streaming caller collects it over the whole scene, takes the
    ``shadow_percentile`` once, and passes it back as ``shadow_threshold``.
    """
    cube = np.asarray(chunk)
    wave = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    swir = wave > 2000.0
    out = np.full(cube.shape[:2], np.nan, dtype=np.float64)
    if not np.any(swir):
        return out
    finite = np.isfinite(cube).all(axis=-1)
    brightness = np.mean(cube[:, :, swir], axis=-1)
    out[finite] = brightness[finite]
    return out
