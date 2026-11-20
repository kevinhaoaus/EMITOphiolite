"""Vectorized EMIT reflectance cleaning for the Oman scene.

The legacy scripts repeated the same masks, Savitzky–Golay smooth, and
continuum removal. This module keeps those choices: atmospheric windows from
the 7-feature-window script, the hand-drawn scene rectangles, a 10th-percentile
SWIR shadow cut, and NDVI above 0.2. Continuum removal divides by the upper
convex hull. It does not divide by the spectrum maximum.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

# Half-open rectangles (r0:r1, c0:c1) copied from the legacy Oman EMIT scripts.
SCENE_EXCLUDE_SLICES: list[tuple[int, int, int, int]] = [
    (291, 384, 1047, 1165),
    (500, 602, 74, 170),
    (287, 330, 151, 218),
    (254, 314, 211, 308),
    (714, 819, 88, 236),
    (1152, 1192, 944, 1021),
    (837, 899, 1, 68),
    (347, 461, 42, 184),
    (414, 470, 175, 300),
    (467, 530, 307, 406),
    (559, 623, 419, 537),
    (506, 547, 545, 618),
    (801, 850, 350, 410),
    (1035, 1078, 560, 611),
    (90, 124, 1029, 1097),
    (0, 61, 1065, 1137),
    (599, 651, 176, 237),
    (703, 731, 331, 382),
    (656, 747, 252, 323),
    (1159, 1194, 462, 510),
    (1155, 1197, 773, 902),
    (779, 865, 630, 666),
]

# Diagnostic windows in nanometers, inclusive when a caller subsets bands.
FEATURE_WINDOWS: dict[str, tuple[int, int]] = {
    "hematite": (550, 950),
    "chlorite": (2180, 2265),
    "carbonate": (2280, 2380),
    "alteration": (2100, 2380),
}

_CONTINUUM_CHUNK = 4096
_MIN_FINITE_SAMPLES = 5
_FLAT_STD = 1e-6
_NDVI_EPS = 1e-10

_NUMBA_KERNEL: Any = None
_NUMBA_TRIED = False


def good_band_mask(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Return the atmospheric windows used by the 7-feature-window script.

    A band is kept when its center is in ``(450, 1340)``, ``(1450, 1800)``,
    or ``(1950, 2450)`` nm. The inequalities are strict, so the endpoints and
    the water intervals between the windows are dropped. Non-finite wavelengths
    are dropped.

    Parameters
    ----------
    wavelengths_nm:
        Band centers in nanometers, shape ``(bands,)``.

    Returns
    -------
    Boolean mask, shape ``(bands,)``.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    return (
        ((wavelengths > 450.0) & (wavelengths < 1340.0))
        | ((wavelengths > 1450.0) & (wavelengths < 1800.0))
        | ((wavelengths > 1950.0) & (wavelengths < 2450.0))
    )


def apply_scene_mask(
    valid_hw: np.ndarray,
    shape: tuple[int, ...],
    use_scene_mask: bool = True,
) -> np.ndarray:
    """Copy ``valid_hw`` and optionally clear the legacy scene rectangles.

    Rectangles are clipped to ``shape[0]`` and ``shape[1]``, so a cube smaller
    than the original EMIT scene does not raise. ``use_scene_mask=False``
    returns the copied mask unchanged.

    Parameters
    ----------
    valid_hw:
        Boolean image, true where a pixel may still be kept.
    shape:
        Array shape. The first two entries are rows and columns.
    use_scene_mask:
        When true, set :data:`SCENE_EXCLUDE_SLICES` to false.

    Returns
    -------
    Boolean array of shape ``(rows, cols)``. The input mask is not modified.
    """
    rows = int(shape[0])
    cols = int(shape[1])
    valid = np.array(valid_hw, dtype=bool, copy=True)
    if valid.shape != (rows, cols):
        raise ValueError(
            f"valid_hw shape {valid.shape} does not match spatial shape {(rows, cols)}"
        )
    if not use_scene_mask:
        return valid

    for r0, r1, c0, c1 in SCENE_EXCLUDE_SLICES:
        rr0 = max(0, min(rows, int(r0)))
        rr1 = max(0, min(rows, int(r1)))
        cc0 = max(0, min(cols, int(c0)))
        cc1 = max(0, min(cols, int(c1)))
        if rr0 < rr1 and cc0 < cc1:
            valid[rr0:rr1, cc0:cc1] = False
    return valid


def pixel_screen(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    shadow_percentile: float = 10,
    ndvi_max: float = 0.2,
) -> np.ndarray:
    """Keep illuminated, non-vegetated pixels with usable spectra.

    Shadowed pixels are those whose mean reflectance in bands longward of
    2000 nm (or in every band, when no such band exists) is below
    ``shadow_percentile`` of the finite brightness values. Vegetation is
    ``NDVI > ndvi_max``, using the bands nearest 660 nm and 850 nm and the
    legacy denominator offset of ``1e-10``. A pixel is also dropped when fewer
    than 95% of its bands are finite, or when ``nanstd <= 1e-6``.

    Parameters
    ----------
    cube:
        Reflectance, shape ``(rows, cols, bands)``. Not modified.
    wavelengths_nm:
        Band centers in nanometers, shape ``(bands,)``.
    shadow_percentile:
        Percentile of finite brightness. Pixels strictly below it are shadows.
    ndvi_max:
        NDVI values strictly above this are treated as vegetation.

    Returns
    -------
    Boolean array, shape ``(rows, cols)``. True means keep.
    """
    data = np.asarray(cube)
    if data.ndim != 3:
        raise ValueError("cube must have shape (rows, cols, bands)")
    rows, cols, bands = data.shape
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    if wavelengths.shape[0] != bands:
        raise ValueError("wavelengths_nm must match cube.shape[-1]")
    if rows == 0 or cols == 0 or bands == 0:
        return np.zeros((rows, cols), dtype=bool)

    swir = wavelengths > 2000.0
    # Excluded rectangles are all-NaN. nanmean/nanstd warn on those pixels.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if np.any(swir):
            brightness = np.nanmean(data[:, :, swir], axis=2)
        else:
            brightness = np.nanmean(data, axis=2)
        deviation = np.nanstd(data, axis=2)
    finite_brightness = np.isfinite(brightness)
    shadow = np.zeros((rows, cols), dtype=bool)
    if np.any(finite_brightness):
        threshold = np.percentile(brightness[finite_brightness], shadow_percentile)
        shadow = finite_brightness & (brightness < threshold)

    red_index = int(np.argmin(np.abs(wavelengths - 660.0)))
    nir_index = int(np.argmin(np.abs(wavelengths - 850.0)))
    red = data[:, :, red_index]
    nir = data[:, :, nir_index]
    ndvi = (nir - red) / (nir + red + _NDVI_EPS)
    vegetation = ndvi > ndvi_max

    finite_fraction = np.isfinite(data).sum(axis=2) / float(bands)
    enough = finite_fraction >= 0.95
    nonflat = deviation > _FLAT_STD
    return (~shadow) & (~vegetation) & enough & nonflat


def subset_bands(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop bands outside :func:`good_band_mask`.

    Parameters
    ----------
    cube:
        Reflectance, shape ``(rows, cols, bands)``.
    wavelengths_nm:
        Band centers in nanometers, shape ``(bands,)``.

    Returns
    -------
    cube_kept:
        Reflectance on the kept bands, shape ``(rows, cols, n_good)``.
    wavelengths_kept:
        Kept band centers, shape ``(n_good,)``.
    """
    data = np.asarray(cube)
    if data.ndim != 3:
        raise ValueError("cube must have shape (rows, cols, bands)")
    wavelengths = np.asarray(wavelengths_nm).reshape(-1)
    if wavelengths.shape[0] != data.shape[-1]:
        raise ValueError("wavelengths_nm must match cube.shape[-1]")
    mask = good_band_mask(wavelengths)
    return data[:, :, mask], wavelengths[mask]


def _upper_hull_indices(wavelengths: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Indices of the upper hull of ``wavelengths``, which must be increasing.

    The chain pops a vertex when the slope increases. That is the same test as
    the legacy scripts, and it keeps collinear samples. A left-to-right walk
    therefore traces the upper envelope rather than the lower hull.
    """
    count = int(wavelengths.shape[0])
    stack = np.empty(count, dtype=np.int64)
    size = 0
    for index in range(count):
        while size >= 2:
            origin = int(stack[size - 2])
            middle = int(stack[size - 1])
            cross = (wavelengths[middle] - wavelengths[origin]) * (
                values[index] - values[middle]
            ) - (values[middle] - values[origin]) * (
                wavelengths[index] - wavelengths[middle]
            )
            if cross > 0.0:
                size -= 1
            else:
                break
        stack[size] = index
        size += 1
    return stack[:size]


def _continuum_python(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill ``out`` with reflectance divided by the upper hull.

    Rows with fewer than five finite samples stay NaN. Hull values that are
    not finite or are ``<= 0`` become NaN. ``out`` must already be NaN-filled.
    """
    finite_wave = np.isfinite(wavelengths)
    for index in range(spectra.shape[0]):
        spectrum = spectra[index]
        usable = np.isfinite(spectrum) & finite_wave
        if int(usable.sum()) < _MIN_FINITE_SAMPLES:
            continue
        sample_w = wavelengths[usable]
        sample_y = spectrum[usable]
        positions = np.flatnonzero(usable)
        order = np.argsort(sample_w, kind="mergesort")
        sample_w = sample_w[order]
        sample_y = sample_y[order]
        positions = positions[order]

        unique = np.empty(sample_w.shape[0], dtype=bool)
        unique[0] = True
        unique[1:] = sample_w[1:] != sample_w[:-1]
        starts = np.flatnonzero(unique)
        hull_w = sample_w[starts]
        hull_y = np.maximum.reduceat(sample_y, starts)
        if hull_w.shape[0] < 2 or np.any(np.diff(hull_w) <= 0.0):
            continue
        vertices = _upper_hull_indices(hull_w, hull_y)
        if vertices.shape[0] < 2:
            continue
        continuum = np.interp(sample_w, hull_w[vertices], hull_y[vertices])
        ratio = np.full(sample_w.shape, np.nan, dtype=np.float64)
        good = np.isfinite(continuum) & (continuum > 0.0)
        ratio[good] = sample_y[good] / continuum[good]
        ratio[~np.isfinite(ratio)] = np.nan
        out[index, positions] = ratio


def _compile_numba() -> Any:
    """Compile the chunk kernel when numba is already installed."""
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=False)
    def _continuum_numba(
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        out: np.ndarray,
    ) -> None:
        n_spectra, n_bands = spectra.shape
        for index in range(n_spectra):
            count = 0
            for band in range(n_bands):
                if np.isfinite(spectra[index, band]) and np.isfinite(wavelengths[band]):
                    count += 1
            if count < 5:
                continue

            sample_w = np.empty(count, dtype=np.float64)
            sample_y = np.empty(count, dtype=np.float64)
            positions = np.empty(count, dtype=np.int64)
            cursor = 0
            for band in range(n_bands):
                value = spectra[index, band]
                wave = wavelengths[band]
                if np.isfinite(value) and np.isfinite(wave):
                    sample_w[cursor] = wave
                    sample_y[cursor] = value
                    positions[cursor] = band
                    cursor += 1

            order = np.argsort(sample_w)
            sample_w = sample_w[order]
            sample_y = sample_y[order]
            positions = positions[order]

            n_unique = 1
            for cursor in range(1, count):
                if sample_w[cursor] != sample_w[cursor - 1]:
                    n_unique += 1
            if n_unique < 2:
                continue

            hull_w = np.empty(n_unique, dtype=np.float64)
            hull_y = np.empty(n_unique, dtype=np.float64)
            unique_index = 0
            hull_w[0] = sample_w[0]
            hull_y[0] = sample_y[0]
            for cursor in range(1, count):
                if sample_w[cursor] == hull_w[unique_index]:
                    if sample_y[cursor] > hull_y[unique_index]:
                        hull_y[unique_index] = sample_y[cursor]
                else:
                    unique_index += 1
                    hull_w[unique_index] = sample_w[cursor]
                    hull_y[unique_index] = sample_y[cursor]

            stack = np.empty(n_unique, dtype=np.int64)
            size = 0
            for vertex in range(n_unique):
                while size >= 2:
                    origin = stack[size - 2]
                    middle = stack[size - 1]
                    cross = (hull_w[middle] - hull_w[origin]) * (
                        hull_y[vertex] - hull_y[middle]
                    ) - (hull_y[middle] - hull_y[origin]) * (
                        hull_w[vertex] - hull_w[middle]
                    )
                    if cross > 0.0:
                        size -= 1
                    else:
                        break
                stack[size] = vertex
                size += 1
            if size < 2:
                continue

            for cursor in range(count):
                x = sample_w[cursor]
                if x <= hull_w[stack[0]]:
                    continuum = hull_y[stack[0]]
                elif x >= hull_w[stack[size - 1]]:
                    continuum = hull_y[stack[size - 1]]
                else:
                    right = 1
                    while right < size - 1 and hull_w[stack[right]] < x:
                        right += 1
                    left = right - 1
                    x0 = hull_w[stack[left]]
                    x1 = hull_w[stack[right]]
                    y0 = hull_y[stack[left]]
                    y1 = hull_y[stack[right]]
                    if x1 == x0:
                        continuum = y0
                    else:
                        continuum = y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
                if continuum > 0.0 and np.isfinite(continuum):
                    ratio = sample_y[cursor] / continuum
                    if np.isfinite(ratio):
                        out[index, positions[cursor]] = ratio

    return _continuum_numba


def _numba_kernel() -> Any:
    """Return the compiled kernel, or None when numba is absent or unusable."""
    global _NUMBA_KERNEL, _NUMBA_TRIED
    if not _NUMBA_TRIED:
        _NUMBA_TRIED = True
        try:
            _NUMBA_KERNEL = _compile_numba()
        except Exception:
            _NUMBA_KERNEL = None
    return _NUMBA_KERNEL


def _disable_numba() -> None:
    global _NUMBA_KERNEL, _NUMBA_TRIED
    _NUMBA_KERNEL = None
    _NUMBA_TRIED = True


def continuum_remove(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Divide each spectrum by its upper convex-hull continuum.

    ``spectra`` has shape ``(n, bands)``. The hull is built in wavelength
    order from the finite samples. Reflectance is divided by the interpolated
    hull. Hull values that are not finite or are ``<= 0`` become NaN, and
    non-finite input samples stay NaN. A row with fewer than five finite
    samples is returned as an all-NaN row.

    Spectra are processed in chunks of 4096. When numba is already installed
    the chunk kernel is compiled; otherwise the same hull is computed in
    NumPy. Numba is not required.

    Parameters
    ----------
    spectra:
        Reflectance spectra, shape ``(n, bands)``.
    wavelengths:
        Band centers aligned with the last axis, shape ``(bands,)``.

    Returns
    -------
    Continuum-removed spectra, same shape as ``spectra``, dtype float64.
    """
    spec = np.asarray(spectra, dtype=np.float64)
    if spec.ndim != 2:
        raise ValueError("spectra must have shape (n, bands)")
    wave = np.ascontiguousarray(np.asarray(wavelengths, dtype=np.float64).reshape(-1))
    if wave.shape[0] != spec.shape[1]:
        raise ValueError("wavelengths must match spectra.shape[1]")

    spec = np.ascontiguousarray(spec)
    out = np.full(spec.shape, np.nan, dtype=np.float64)
    kernel = _numba_kernel()
    n_spectra = spec.shape[0]
    for start in range(0, n_spectra, _CONTINUUM_CHUNK):
        stop = min(start + _CONTINUUM_CHUNK, n_spectra)
        block = spec[start:stop]
        dest = out[start:stop]
        if kernel is not None:
            try:
                kernel(block, wave, dest)
                continue
            except Exception:
                _disable_numba()
                kernel = None
        _continuum_python(block, wave, dest)
    return out


def smooth_spectra(
    spectra: np.ndarray,
    window: int = 11,
    poly: int = 2,
) -> np.ndarray:
    """Savitzky–Golay smooth along the band axis.

    ``scipy.signal.savgol_filter`` is applied on axis 1 with ``mode='interp'``.
    A row with fewer than ``window`` finite samples is left unchanged. Rows
    that contain gaps are filled only for the filter, and the original gaps
    are restored. The input array is not modified.

    Parameters
    ----------
    spectra:
        Reflectance, shape ``(n, bands)``.
    window:
        Odd filter length. The legacy SAM script used 11.
    poly:
        Polynomial order. The legacy scripts used 2.

    Returns
    -------
    Smoothed copy, dtype float64, same shape as ``spectra``.
    """
    out = np.array(np.asarray(spectra), dtype=np.float64, copy=True)
    if out.ndim != 2:
        raise ValueError("spectra must have shape (n, bands)")
    n_spectra, n_bands = out.shape
    # A band axis shorter than the window cannot be filtered. An invalid
    # window or polynomial is left for savgol_filter to reject.
    if n_spectra == 0 or n_bands == 0 or n_bands < window:
        return out

    finite = np.isfinite(out)
    n_finite = finite.sum(axis=1)
    eligible = n_finite >= window
    if not np.any(eligible):
        return out

    complete = eligible & (n_finite == n_bands)
    if np.any(complete):
        out[complete] = savgol_filter(
            out[complete],
            window_length=window,
            polyorder=poly,
            axis=1,
            mode="interp",
        )

    partial = np.flatnonzero(eligible & ~complete)
    if partial.size:
        block = out[partial].copy()
        finite_block = finite[partial]
        band_index = np.arange(n_bands, dtype=np.float64)
        for row in range(block.shape[0]):
            good = finite_block[row]
            block[row] = np.interp(band_index, band_index[good], block[row, good])
        block = savgol_filter(
            block,
            window_length=window,
            polyorder=poly,
            axis=1,
            mode="interp",
        )
        block[~finite_block] = np.nan
        out[partial] = block
    return out


def prepare_matrix(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    use_scene_mask: bool = False,
) -> dict[str, np.ndarray]:
    """Build the continuum-removed matrix used by the Oman classifiers.

    Good bands are selected first. When ``use_scene_mask`` is true, the legacy
    rectangles are set to NaN before the shadow percentile so they do not move
    that threshold, and they are marked invalid. Kept spectra are smoothed and
    then continuum-removed. One row of ``X`` is stored for every true pixel in
    ``valid``, in row-major order.

    Parameters
    ----------
    cube:
        Reflectance, shape ``(rows, cols, bands)``. Not modified.
    wavelengths_nm:
        Band centers in nanometers, shape ``(bands,)``.
    use_scene_mask:
        Drop :data:`SCENE_EXCLUDE_SLICES`. Default false so a subset or a
        synthetic cube is not cleared by rectangles from the full Oman scene.

    Returns
    -------
    dict
        ``wavelengths`` holds the kept band centers. ``valid`` is a boolean
        image. ``X`` is float32 with shape ``(n_valid, n_bands)``. ``rows``
        and ``cols`` are the image indices of each row of ``X``.
    """
    subset, wavelengths = subset_bands(cube, wavelengths_nm)
    if np.may_share_memory(subset, np.asarray(cube)):
        subset = subset.copy()
    if use_scene_mask:
        allowed = apply_scene_mask(
            np.ones(subset.shape[:2], dtype=bool),
            subset.shape,
            True,
        )
        subset[~allowed] = np.nan

    valid = pixel_screen(subset, wavelengths)
    valid = apply_scene_mask(valid, subset.shape, use_scene_mask=use_scene_mask)
    row_index, col_index = np.nonzero(valid)
    n_bands = int(wavelengths.shape[0])
    if row_index.size == 0:
        matrix = np.empty((0, n_bands), dtype=np.float32)
    else:
        spectra = np.asarray(subset[row_index, col_index, :], dtype=np.float64)
        matrix = np.asarray(
            continuum_remove(smooth_spectra(spectra), wavelengths),
            dtype=np.float32,
        )
    return {
        "wavelengths": np.asarray(wavelengths),
        "valid": valid,
        "X": matrix,
        "rows": row_index,
        "cols": col_index,
    }


__all__ = [
    "FEATURE_WINDOWS",
    "SCENE_EXCLUDE_SLICES",
    "apply_scene_mask",
    "continuum_remove",
    "good_band_mask",
    "pixel_screen",
    "prepare_matrix",
    "smooth_spectra",
    "subset_bands",
]
