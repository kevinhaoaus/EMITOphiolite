"""Minimum Noise Fraction (Green et al., 1988).

Noise is the covariance of a one-pixel shift along axis 1. Components are the
leading generalized eigenvectors of signal_cov v = lambda noise_cov v.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def estimate_noise_covariance(cube: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """cube (H,W,B), valid bool (H,W). Noise samples are cube[:,1:]-cube[:,:-1] where BOTH pixels are valid. Return (B,B) covariance. If fewer than B+5 noise samples, fall back to a small diagonal (1e-6 * I). Add 1e-6 ridge * trace/B to the diagonal for stability. Use float64."""
    cube = np.asarray(cube, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    bands = int(cube.shape[-1])
    eye = np.eye(bands, dtype=np.float64)

    if cube.ndim != 3 or cube.shape[1] < 2 or bands == 0:
        return 1e-6 * eye

    diff = cube[:, 1:, :] - cube[:, :-1, :]
    pair = valid[:, 1:] & valid[:, :-1]
    samples = diff[pair]
    if samples.shape[0] < bands + 5:
        return 1e-6 * eye

    cov = np.atleast_2d(np.asarray(np.cov(samples, rowvar=False), dtype=np.float64))
    ridge = 1e-6 * float(np.trace(cov)) / float(bands)
    if ridge <= 0.0:
        return 1e-6 * eye
    return cov + ridge * eye


def fit_mnf(cube: np.ndarray, valid: np.ndarray, n_components: int = 8) -> dict:
    """Signal covariance of valid pixels (rowvar=False), noise covariance as above.
    Solve generalized eigenproblem with scipy.linalg.eigh(signal_cov, noise_cov).
    Sort by eigenvalue descending. Keep k = min(n_components, B).
    Return eigenvalues (k,), eigenvectors (B,k) columns, noise_cov, signal_mean (B,).
    Project as (X - mean) @ eigenvectors. Do that inside transform.
    """
    cube = np.asarray(cube, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    bands = int(cube.shape[-1])

    pixels = cube[valid]
    if pixels.ndim == 1:
        pixels = pixels.reshape(-1, bands)

    if pixels.shape[0] == 0:
        signal_mean = np.zeros(bands, dtype=np.float64)
        signal_cov = np.zeros((bands, bands), dtype=np.float64)
    else:
        signal_mean = np.asarray(pixels.mean(axis=0), dtype=np.float64)
        if pixels.shape[0] < 2:
            signal_cov = np.zeros((bands, bands), dtype=np.float64)
        else:
            signal_cov = np.atleast_2d(
                np.asarray(np.cov(pixels, rowvar=False), dtype=np.float64)
            )

    noise_cov = estimate_noise_covariance(cube, valid)
    eigenvalues, eigenvectors = linalg.eigh(signal_cov, noise_cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)[order]
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)[:, order]

    k = min(int(n_components), bands)
    if k < 0:
        k = 0

    return {
        "eigenvalues": eigenvalues[:k].copy(),
        "eigenvectors": eigenvectors[:, :k].copy(),
        "noise_cov": noise_cov,
        "signal_mean": signal_mean,
    }


def transform_mnf(cube, valid, mnf: dict) -> np.ndarray:
    """Return scores (H,W,k) float32, nan where ~valid."""
    cube = np.asarray(cube, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    height, width, bands = cube.shape
    eigenvectors = np.asarray(mnf["eigenvectors"], dtype=np.float64)
    mean = np.asarray(mnf["signal_mean"], dtype=np.float64)
    k = int(eigenvectors.shape[1])

    scores = np.full((height, width, k), np.nan, dtype=np.float32)
    if k == 0 or not np.any(valid):
        return scores

    pixels = cube[valid]
    if pixels.ndim == 1:
        pixels = pixels.reshape(-1, bands)
    scores[valid] = ((pixels - mean) @ eigenvectors).astype(np.float32, copy=False)
    return scores


def mnf_rgb(scores):
    """Same 2-98 stretch on first 3 components to (H,W,3). NaN->0."""
    scores = np.asarray(scores, dtype=np.float64)
    height, width, k = scores.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(min(3, k)):
        band = scores[:, :, i]
        finite = np.isfinite(band)
        if not np.any(finite):
            continue
        p2, p98 = np.nanpercentile(band, (2, 98))
        stretched = (band - p2) / (p98 - p2 + 1e-10)
        stretched = np.clip(stretched, 0.0, 1.0)
        stretched = np.where(finite, stretched, 0.0)
        rgb[:, :, i] = stretched.astype(np.float32, copy=False)
    return rgb


def noise_covariance_from_differences(diff_samples, bands: int) -> np.ndarray:
    """Noise covariance from pre-sampled one-pixel difference spectra (N, B).

    Same rule as estimate_noise_covariance (fewer than B+5 samples -> 1e-6 * I,
    otherwise np.cov plus a 1e-6 * trace/B ridge). Used by the streaming
    pipeline, which samples difference pairs chunk by chunk.
    """
    samples = np.asarray(diff_samples, dtype=np.float64)
    eye = np.eye(bands, dtype=np.float64)
    if bands == 0 or samples.ndim != 2 or samples.shape[0] < bands + 5:
        return 1e-6 * eye
    cov = np.atleast_2d(np.asarray(np.cov(samples, rowvar=False), dtype=np.float64))
    ridge = 1e-6 * float(np.trace(cov)) / float(bands)
    if ridge <= 0.0:
        return 1e-6 * eye
    return cov + ridge * eye


def fit_mnf_from_samples(signal_pixels, noise_cov, n_components: int = 8) -> dict:
    """fit_mnf on a sampled (N, B) signal matrix and a precomputed noise covariance.

    Same math as fit_mnf: np.cov signal covariance, scipy.linalg.eigh(signal_cov,
    noise_cov), eigenvalues descending, k = min(n_components, B).
    """
    pixels = np.asarray(signal_pixels, dtype=np.float64)
    noise_cov = np.asarray(noise_cov, dtype=np.float64)
    bands = int(noise_cov.shape[0])
    if pixels.shape[0] == 0:
        signal_mean = np.zeros(bands, dtype=np.float64)
        signal_cov = np.zeros((bands, bands), dtype=np.float64)
    else:
        signal_mean = np.asarray(pixels.mean(axis=0), dtype=np.float64)
        if pixels.shape[0] < 2:
            signal_cov = np.zeros((bands, bands), dtype=np.float64)
        else:
            signal_cov = np.atleast_2d(np.asarray(np.cov(pixels, rowvar=False), dtype=np.float64))
    eigenvalues, eigenvectors = linalg.eigh(signal_cov, noise_cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)[order]
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)[:, order]
    k = max(min(int(n_components), bands), 0)
    return {
        "eigenvalues": eigenvalues[:k].copy(),
        "eigenvectors": eigenvectors[:, :k].copy(),
        "noise_cov": noise_cov,
        "signal_mean": signal_mean,
    }


def transform_mnf_pixels(pixels, mnf: dict) -> np.ndarray:
    """Project (N, B) pixels as (X - mean) @ eigenvectors; float32 (N, k)."""
    x = np.asarray(pixels, dtype=np.float64)
    eigenvectors = np.asarray(mnf["eigenvectors"], dtype=np.float64)
    if x.shape[0] == 0:
        return np.empty((0, eigenvectors.shape[1]), dtype=np.float32)
    mean = np.asarray(mnf["signal_mean"], dtype=np.float64)
    return ((x - mean) @ eigenvectors).astype(np.float32, copy=False)
