"""Minimum Noise Fraction for a hyperspectral image cube.

This is the Green et al. (1988) MNF transform, not PCA. Shift-difference
noise is whitened first; a subsequent PCA then orders components by
signal-to-noise rather than by raw variance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

MNF_SUBSAMPLE_ROWS: int = 200_000
_RIDGE_SCALE: float = 1e-6
_EIGH_REL_FLOOR: float = 1e-8
_DIFF_PIXELS_PER_CHUNK: int = 40_000

FloatArray = NDArray[np.floating]


def estimate_noise(cube: np.ndarray) -> FloatArray:
    """Noise covariance from one-pixel shifts to the right and below.

    Residuals are kept only where both spectra are fully finite. Their
    covariance is divided by 2, then a ridge of ``1e-6`` times the mean
    diagonal is added so the result is symmetric positive definite.
    """
    data = np.asarray(cube)
    if data.ndim != 3:
        raise ValueError(f"cube must have shape (rows, cols, bands), got {data.shape}")
    bands = int(data.shape[-1])
    count, scatter = _residual_scatter(data)
    if count < 2 or bands == 0:
        cov = np.zeros((bands, bands), dtype=np.float64)
    else:
        cov = scatter / float(count - 1)
        cov *= 0.5
        cov = 0.5 * (cov + cov.T)

    mean_diag = float(np.mean(np.diag(cov))) if bands else 0.0
    if not np.isfinite(mean_diag):
        mean_diag = 0.0
    ridge = _RIDGE_SCALE * mean_diag
    if ridge != 0.0:
        cov = cov + ridge * np.eye(bands, dtype=np.float64)
    return np.asarray(0.5 * (cov + cov.T), dtype=np.float64)


def fit_mnf(cube: np.ndarray, n_components: int = 10) -> dict[str, FloatArray]:
    """Fit MNF on an image cube and return scores on the same grid.

    Noise is whitened by the inverse square root of :func:`estimate_noise`
    (eigenvalues below ``1e-8`` of the largest are clipped). sklearn PCA is
    then fit on the whitened finite pixels, using at most 200,000 random
    rows when the image has more. ``k`` is capped at
    ``min(n_components, bands - 1)``.

    Scores are float32 and NaN wherever the input spectrum was not fully
    finite. Eigenvalues are the noise-whitened variances (descending SNR).
    ``explained`` is each eigenvalue divided by the total noise-adjusted
    variance when that fraction is finite and sums to at most one.
    ``vectors`` are the MNF basis in original band coordinates, shape
    ``(bands, k)``.
    """
    data = np.asarray(cube)
    if data.ndim != 3:
        raise ValueError(f"cube must have shape (rows, cols, bands), got {data.shape}")
    if int(n_components) < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    rows, cols, bands = (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
    k_cap = min(int(n_components), bands - 1)
    finite = np.isfinite(data).all(axis=-1) if bands else np.zeros((rows, cols), dtype=bool)
    n_finite = int(finite.sum())
    if k_cap < 1 or n_finite < 2:
        k_empty = max(k_cap, 0)
        return _empty_mnf(rows, cols, bands, k_empty)

    whitener = _noise_whitener(estimate_noise(data))
    ys, xs = np.nonzero(finite)
    fit_index = _fit_index(n_finite)
    fit_pixels = np.asarray(data[ys[fit_index], xs[fit_index]], dtype=np.float64)
    whitened_fit = fit_pixels @ whitener

    k = min(k_cap, int(whitened_fit.shape[0]), bands)
    if k < 1:
        return _empty_mnf(rows, cols, bands, 0)

    pca = PCA(n_components=k, svd_solver="full", random_state=42)
    pca.fit(whitened_fit)

    rotation = np.asarray(pca.components_, dtype=np.float64).T
    vectors = np.asarray(whitener @ rotation, dtype=np.float64)
    eigenvalues = np.asarray(pca.explained_variance_, dtype=np.float64)
    explained = _explained_fraction(whitened_fit, np.asarray(pca.mean_, dtype=np.float64), eigenvalues)
    scores = _score_image(data, finite, whitener, pca, k)
    return {
        "scores": scores,
        "eigenvalues": eigenvalues,
        "explained": explained,
        "vectors": vectors,
    }


def mnf_rgb(scores: np.ndarray) -> NDArray[np.float32]:
    """2–98 percentile stretch of the first three MNF components.

    The result is float32 in ``[0, 1]``. NaN and other non-finite scores
    become 0. Missing components stay 0.
    """
    data = np.asarray(scores)
    if data.ndim != 3:
        raise ValueError(
            f"scores must have shape (rows, cols, components), got {data.shape}"
        )
    rows, cols, n_comp = data.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    for channel in range(min(3, int(n_comp))):
        band = np.asarray(data[:, :, channel], dtype=np.float64)
        finite = np.isfinite(band)
        if not np.any(finite):
            continue
        low, high = np.percentile(band[finite], (2.0, 98.0))
        span = float(high - low)
        if not np.isfinite(span) or span <= 0.0:
            continue
        scaled = (band - low) / span
        np.clip(scaled, 0.0, 1.0, out=scaled)
        rgb[:, :, channel] = np.where(finite, scaled, 0.0)
    return rgb


def _empty_mnf(rows: int, cols: int, bands: int, k: int) -> dict[str, FloatArray]:
    return {
        "scores": np.full((rows, cols, k), np.nan, dtype=np.float32),
        "eigenvalues": np.full(k, np.nan, dtype=np.float64),
        "explained": np.full(k, np.nan, dtype=np.float64),
        "vectors": np.full((bands, k), np.nan, dtype=np.float64),
    }


def _residual_scatter(cube: np.ndarray) -> tuple[int, FloatArray]:
    """Centered scatter of pooled right- and below-neighbor residuals.

    Returns the residual count and ``sum((x - mean)(x - mean)^T)``.
    Dividing by ``count - 1`` matches ``numpy.cov``.
    """
    bands = int(cube.shape[-1])
    count = 0
    total = np.zeros(bands, dtype=np.float64)
    for diff in _iter_shift_diffs(cube):
        count += int(diff.shape[0])
        total += diff.sum(axis=0)
    scatter = np.zeros((bands, bands), dtype=np.float64)
    if count < 2:
        return count, scatter
    mean = total / float(count)
    for diff in _iter_shift_diffs(cube):
        centered = diff - mean
        scatter += centered.T @ centered
    return count, scatter


def _iter_shift_diffs(cube: np.ndarray):
    rows, cols, _bands = cube.shape
    if cols >= 2:
        yield from _iter_pair_diffs(cube[:, :-1, :], cube[:, 1:, :])
    if rows >= 2:
        yield from _iter_pair_diffs(cube[:-1, :, :], cube[1:, :, :])


def _iter_pair_diffs(left: np.ndarray, right: np.ndarray):
    n_rows, n_cols, _bands = left.shape
    rows_per = max(1, _DIFF_PIXELS_PER_CHUNK // max(n_cols, 1))
    for r0 in range(0, n_rows, rows_per):
        a = left[r0 : r0 + rows_per]
        b = right[r0 : r0 + rows_per]
        valid = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
        if not np.any(valid):
            continue
        diff = np.asarray(a[valid], dtype=np.float64)
        diff -= np.asarray(b[valid], dtype=np.float64)
        if diff.ndim == 1:
            diff = diff.reshape(1, -1)
        if diff.shape[0] == 0:
            continue
        yield diff


def _noise_whitener(noise: np.ndarray) -> FloatArray:
    """Inverse square root of a symmetric noise covariance."""
    bands = int(noise.shape[0])
    symmetric = 0.5 * (np.asarray(noise, dtype=np.float64) + np.asarray(noise, dtype=np.float64).T)
    evals, evecs = np.linalg.eigh(symmetric)
    max_eval = float(evals[-1]) if evals.size else 0.0
    if not np.isfinite(max_eval) or max_eval <= 0.0:
        return np.eye(bands, dtype=np.float64)
    floor = _EIGH_REL_FLOOR * max_eval
    clipped = np.maximum(evals, floor)
    inv_sqrt = 1.0 / np.sqrt(clipped)
    whitener = (evecs * inv_sqrt) @ evecs.T
    return np.asarray(0.5 * (whitener + whitener.T), dtype=np.float64)


def _fit_index(n_finite: int) -> np.ndarray:
    if n_finite <= MNF_SUBSAMPLE_ROWS:
        return np.arange(n_finite)
    rng = np.random.default_rng(42)
    picked = rng.choice(n_finite, size=MNF_SUBSAMPLE_ROWS, replace=False)
    return np.sort(picked)


def _explained_fraction(
    whitened_fit: np.ndarray,
    mean: np.ndarray,
    eigenvalues: np.ndarray,
) -> FloatArray:
    """Share of total noise-whitened variance, or NaN if it is unstable."""
    n_fit = int(whitened_fit.shape[0])
    k = int(eigenvalues.shape[0])
    unstable = np.full(k, np.nan, dtype=np.float64)
    if n_fit < 2 or k == 0 or not np.all(np.isfinite(eigenvalues)):
        return unstable
    centered = whitened_fit - mean
    total = float(np.sum(centered * centered) / float(n_fit - 1))
    if not np.isfinite(total) or total <= 0.0:
        return unstable
    fraction = np.asarray(eigenvalues / total, dtype=np.float64)
    if not np.all(np.isfinite(fraction)) or float(np.sum(fraction)) > 1.0 + 1e-4:
        return unstable
    return fraction


def _score_image(
    cube: np.ndarray,
    finite: np.ndarray,
    whitener: np.ndarray,
    pca: PCA,
    k: int,
) -> NDArray[np.float32]:
    rows, cols, _bands = cube.shape
    scores = np.full((rows, cols, k), np.nan, dtype=np.float32)
    rows_per = max(1, _DIFF_PIXELS_PER_CHUNK // max(cols, 1))
    for r0 in range(0, rows, rows_per):
        mask = finite[r0 : r0 + rows_per]
        if not np.any(mask):
            continue
        block = cube[r0 : r0 + rows_per]
        pixels = np.asarray(block[mask], dtype=np.float64)
        transformed = pca.transform(pixels @ whitener)
        local_y, local_x = np.nonzero(mask)
        scores[r0 + local_y, local_x, :] = transformed
    return scores
