"""Hyperspectral matched filter (MF) for mineral mapping.

Scores are dimensionless detection scores. A higher score means the spectrum
is more like the target after the background mean is removed and the
background covariance is whitened. A pixel equal to the background mean
scores 0; a pixel equal to the target scores 1. Scores are not abundances
and are not bounded to ``[0, 1]``.

The detector is the classical normalized matched filter::

    s(x) = ((t - m)^T Sigma^{-1} (x - m))
           / ((t - m)^T Sigma^{-1} (t - m))

No plotting and no file I/O live in this module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Background sample larger than this is thinned before mean/covariance.
_MAX_BACKGROUND_PIXELS = 200_000
_BACKGROUND_RANDOM_STATE = 42
# Shrink the sample covariance toward its diagonal, then ridge the result.
_COV_SHRINK_ALPHA = 0.05
_COV_RIDGE = 1e-6
# Relative floor for eigenvalues of a covariance before inversion.
_EIGH_FLOOR_RATIO = 1e-8


def background_stats(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate the background mean and a stabilized covariance.

    Parameters
    ----------
    X
        Spectra with shape ``(n, bands)``. Values are treated as finite.

    Returns
    -------
    mean
        Background mean, shape ``(bands,)``.
    cov
        Shrunk and ridged covariance, shape ``(bands, bands)``.

    If ``n > 200_000``, both statistics use a subsample of 200_000 rows drawn
    without replacement from ``numpy.random.RandomState(42)``. The sample
    covariance uses the unbiased divisor ``n - 1``. It is then shrunk toward
    its diagonal and ridged::

        cov = (1 - alpha) * sample_cov + alpha * diag(sample_cov)
        cov = cov + (1e-6 * mean(diag(sample_cov))) * I

    with ``alpha = 0.05``. Shrinkage leaves the diagonal unchanged, so the
    ridge uses that same mean variance.
    """
    spectra = np.asarray(X)
    if spectra.ndim != 2:
        raise ValueError(f"X must have shape (n, bands), got {spectra.shape}")
    n_pixels, n_bands = spectra.shape
    if n_pixels == 0 or n_bands == 0:
        raise ValueError("X must contain at least one spectrum and one band")

    if n_pixels > _MAX_BACKGROUND_PIXELS:
        rng = np.random.RandomState(_BACKGROUND_RANDOM_STATE)
        chosen = rng.choice(n_pixels, size=_MAX_BACKGROUND_PIXELS, replace=False)
        sample = np.asarray(spectra[chosen], dtype=np.float64)
    else:
        sample = np.asarray(spectra, dtype=np.float64)

    mean = sample.mean(axis=0)
    n_sample = sample.shape[0]
    if n_sample < 2:
        sample_cov = np.zeros((n_bands, n_bands), dtype=np.float64)
    else:
        sample_cov = np.cov(sample, rowvar=False, ddof=1)
        sample_cov = np.atleast_2d(np.asarray(sample_cov, dtype=np.float64))

    variances = np.diag(sample_cov).astype(np.float64, copy=True)
    cov = (1.0 - _COV_SHRINK_ALPHA) * sample_cov + _COV_SHRINK_ALPHA * np.diag(variances)
    cov = np.asarray(cov, dtype=np.float64)
    ridge = _COV_RIDGE * float(np.mean(variances))
    diagonal = np.arange(n_bands)
    cov[diagonal, diagonal] += ridge
    cov = 0.5 * (cov + cov.T)
    return mean.astype(np.float64, copy=False), cov


def invert_cov(cov: ArrayLike) -> NDArray[np.float64]:
    """Stable inverse of a covariance via a clipped eigendecomposition.

    Eigenvalues from ``numpy.linalg.eigh`` are raised to at least
    ``1e-8 * max(eigenvalue)`` before inversion, which keeps a near-singular
    background covariance invertible. The input is symmetrized first.

    Returns
    -------
    cov_inv
        Symmetric inverse, shape ``(bands, bands)``, float64.
    """
    matrix = np.asarray(cov, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"cov must be square, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("cov must be non-empty")

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    max_eigenvalue = float(np.max(eigenvalues))
    floor = _EIGH_FLOOR_RATIO * max_eigenvalue
    # A non-positive floor would leave a zero or negative mode uninverted.
    if not np.isfinite(floor) or floor <= 0.0:
        floor = _EIGH_FLOOR_RATIO
    clipped = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors * (1.0 / clipped)) @ eigenvectors.T
    return 0.5 * (inverse + inverse.T)


def matched_filter_scores(
    X: ArrayLike,
    target: ArrayLike,
    mean: ArrayLike,
    cov_inv: ArrayLike,
) -> NDArray[np.float32]:
    """Score each spectrum with the normalized matched filter.

    Parameters
    ----------
    X
        Spectra, shape ``(n, bands)``.
    target
        Target spectrum, shape ``(bands,)``. Must have the same band count
        as ``X``.
    mean
        Background mean, shape ``(bands,)``.
    cov_inv
        Inverse background covariance, shape ``(bands, bands)``, typically
        from :func:`invert_cov`.

    Returns
    -------
    scores
        Shape ``(n,)``, float32. Dimensionless detection scores: higher means
        more like the target. Rows of ``X`` that contain a non-finite value
        are ``NaN``. A spectrum equal to ``target`` scores 1 and a spectrum
        equal to ``mean`` scores 0 when the denominator is finite.

    The score is::

        s = ((t - m)^T Sigma^{-1} (x - m))
            / ((t - m)^T Sigma^{-1} (t - m))
    """
    spectra = np.asarray(X, dtype=np.float64)
    if spectra.ndim != 2:
        raise ValueError(f"X must have shape (n, bands), got {getattr(X, 'shape', None)}")
    n_pixels, n_bands = spectra.shape

    target_spectrum = np.asarray(target, dtype=np.float64).reshape(-1)
    background_mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    inverse = np.asarray(cov_inv, dtype=np.float64)
    if target_spectrum.shape != (n_bands,) or background_mean.shape != (n_bands,):
        raise ValueError(
            "target and X must share bands: "
            f"X has {n_bands}, target has {target_spectrum.size}, "
            f"mean has {background_mean.size}"
        )
    if inverse.shape != (n_bands, n_bands):
        raise ValueError(
            f"cov_inv must have shape {(n_bands, n_bands)}, got {inverse.shape}"
        )
    if not np.isfinite(target_spectrum).all():
        raise ValueError("target must be finite")
    if not np.isfinite(background_mean).all():
        raise ValueError("mean must be finite")

    contrast = target_spectrum - background_mean
    weights = inverse @ contrast
    denominator = float(np.dot(contrast, weights))

    scores = np.full(n_pixels, np.nan, dtype=np.float32)
    finite_rows = np.isfinite(spectra).all(axis=1)
    if n_pixels == 0 or not np.any(finite_rows) or not np.isfinite(denominator):
        return scores

    centered = spectra[finite_rows] - background_mean
    scores[finite_rows] = (centered @ weights) / denominator
    return scores


def run_matched_filters(
    X: ArrayLike,
    targets: dict[str, np.ndarray],
) -> dict[str, NDArray[np.float32]]:
    """Run one matched filter per target against a shared background.

    The background mean and inverse covariance are fit once from ``X`` and
    reused for every target.

    Parameters
    ----------
    X
        Spectra, shape ``(n, bands)``.
    targets
        Map of target name to spectrum, each shape ``(bands,)``.

    Returns
    -------
    dict
        Map of target name to a float32 score vector of shape ``(n,)``.
        Scores are dimensionless detection scores; higher means more like
        that target.
    """
    mean, cov = background_stats(X)
    cov_inv = invert_cov(cov)
    return {
        name: matched_filter_scores(X, spectrum, mean, cov_inv)
        for name, spectrum in targets.items()
    }


def scores_to_map(
    scores: ArrayLike,
    valid_flat_index: ArrayLike,
    shape: tuple[int, int],
) -> NDArray[np.float32]:
    """Scatter score samples onto a 2-D map.

    Parameters
    ----------
    scores
        Detection scores for the valid pixels only. Length must match the
        number of valid locations.
    valid_flat_index
        Either integer indices into the C-order ravel of a ``(rows, cols)``
        image (``index = row * cols + col``), or a boolean mask of shape
        ``(rows, cols)``. For a mask, ``scores`` aligns with the ``True``
        cells in C-order and ``len(scores) == mask.sum()``.
    shape
        ``(rows, cols)`` of the output map.

    Returns
    -------
    score_map
        Float32 array of shape ``(rows, cols)``. Valid pixels hold the
        scores; every other pixel is ``NaN``.
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be (rows, cols), got {shape}")
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 0 or cols < 0:
        raise ValueError(f"shape must be non-negative, got {(rows, cols)}")

    score_vector = np.asarray(scores, dtype=np.float32).reshape(-1)
    index = np.asarray(valid_flat_index)
    score_map = np.full((rows, cols), np.nan, dtype=np.float32)

    if index.dtype == np.bool_:
        if index.shape != (rows, cols):
            raise ValueError(
                f"boolean mask shape {index.shape} != map shape {(rows, cols)}"
            )
        n_valid = int(index.sum())
        if score_vector.size != n_valid:
            raise ValueError(
                f"scores length {score_vector.size} != mask.sum() {n_valid}"
            )
        positions = np.flatnonzero(index.reshape(-1, order="C"))
        score_map.ravel()[positions] = score_vector
        return score_map

    if not np.issubdtype(index.dtype, np.integer):
        raise TypeError(
            "valid_flat_index must be integer indices or a boolean mask "
            f"of shape {(rows, cols)}, got dtype {index.dtype}"
        )
    flat = index.astype(np.int64, copy=False).reshape(-1)
    if flat.size != score_vector.size:
        raise ValueError(
            f"scores length {score_vector.size} != number of indices {flat.size}"
        )
    score_map.ravel()[flat] = score_vector
    return score_map


__all__ = [
    "background_stats",
    "invert_cov",
    "matched_filter_scores",
    "run_matched_filters",
    "scores_to_map",
]
