"""Hyperspectral matched filter for mineral mapping.

score(x) = (t - m)^T Sigma^{-1} (x - m) / (t - m)^T Sigma^{-1} (t - m)
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def background_stats(spectra: np.ndarray) -> tuple:
    """spectra (N,B) finite. Return mean (B,), cov (B,B) float64.
    Covariance: Ledoit-Wolf shrinkage via sklearn.covariance.LedoitWolf if N>B else np.cov with a ridge of 1e-3 * mean eigenvalue on the diagonal.
    """
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("spectra must have shape (N, B)")
    n_samples, n_bands = x.shape
    mean = np.mean(x, axis=0)
    if n_samples > n_bands:
        cov = np.array(LedoitWolf().fit(x).covariance_, dtype=np.float64, copy=True)
    else:
        cov = np.array(np.cov(x, rowvar=False), dtype=np.float64, copy=True, ndmin=2)
        ridge = 1e-3 * float(np.mean(np.linalg.eigvalsh(cov)))
        cov = cov + ridge * np.eye(n_bands, dtype=np.float64)
    return mean, cov


def matched_filter(
    spectra: np.ndarray, target: np.ndarray, background=None
) -> np.ndarray:
    """spectra (N,B), target (B,). background is optional (mean, cov). If None, estimate from spectra.
    Target bands that are nan: drop those bands from spectra, target, mean, and cov together.
    If fewer than 3 bands remain, return zeros.
    Use np.linalg.solve, not explicit inverse. Clip nothing. Return float32 (N,). Infeasible rows (non-finite spectra) -> nan.
    """
    x = np.asarray(spectra, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("spectra must have shape (N, B)")
    n_samples, n_bands = x.shape
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if target.shape[0] != n_bands:
        raise ValueError("target length must equal the number of bands")

    keep = np.isfinite(target)
    if int(np.count_nonzero(keep)) < 3:
        return np.zeros(n_samples, dtype=np.float32)

    x_keep = x[:, keep]
    target_keep = target[keep]
    feasible = np.isfinite(x_keep).all(axis=1)

    if background is None:
        if not np.any(feasible):
            return np.full(n_samples, np.nan, dtype=np.float32)
        mean, cov = background_stats(x_keep[feasible])
    else:
        mean, cov = background
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)[keep]
        cov = np.asarray(cov, dtype=np.float64)[np.ix_(keep, keep)]

    scores = np.full(n_samples, np.nan, dtype=np.float64)
    if np.any(feasible):
        direction = target_keep - mean
        weights = np.linalg.solve(np.asarray(cov, dtype=np.float64), direction)
        denom = np.dot(direction, weights)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scores[feasible] = (x_keep[feasible] - mean) @ weights / denom
    return np.asarray(scores, dtype=np.float32)


def match_library(spectra, library: dict, background=None) -> dict:
    """library name->(B,). One shared background. Return {name: scores (N,)}."""
    x = np.asarray(spectra)
    if background is None:
        if x.ndim != 2:
            raise ValueError("spectra must have shape (N, B)")
        feasible = np.isfinite(np.asarray(x, dtype=np.float64)).all(axis=1)
        if not np.any(feasible):
            blank = np.full(x.shape[0], np.nan, dtype=np.float32)
            return {name: blank.copy() for name in library}
        background = background_stats(np.asarray(x[feasible], dtype=np.float64))
    return {
        name: matched_filter(x, spectrum, background=background)
        for name, spectrum in library.items()
    }
