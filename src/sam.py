"""Vectorized spectral angle mapper."""

import numpy as np


def spectral_angles(spectra: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """spectra (N,B), targets (M,B). Angle in radians (N,M).
    NaN-safe: for each target, use bands finite in that target AND in each spectrum. Implement without a Python loop over N if you can (loop over M is ok). Cosine clipped to [-1,1]. Empty overlap -> nan.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if spectra.ndim != 2 or targets.ndim != 2:
        raise ValueError("spectra must be (N, B) and targets must be (M, B)")
    n, bands = spectra.shape
    m, target_bands = targets.shape
    if target_bands != bands:
        raise ValueError("spectra and targets must have the same number of bands")

    angles = np.empty((n, m), dtype=np.float64)
    if n == 0 or m == 0:
        return angles

    spec_finite = np.isfinite(spectra)
    for j in range(m):
        target = targets[j]
        mask = spec_finite & np.isfinite(target)
        spec_v = np.where(mask, spectra, 0.0)
        targ_v = np.where(mask, target, 0.0)
        denom = np.linalg.norm(spec_v, axis=1) * np.linalg.norm(targ_v, axis=1)
        valid = mask.any(axis=1) & (denom > 0.0)
        cosang = np.full(n, np.nan, dtype=np.float64)
        np.divide(np.sum(spec_v * targ_v, axis=1), denom, out=cosang, where=valid)
        np.clip(cosang, -1.0, 1.0, out=cosang)
        angles[:, j] = np.arccos(cosang)
    return angles


def sam_classify(spectra: np.ndarray, library: dict, max_angle_rad: float = 0.12):
    """Return labels int32 (N,) with -1 where best angle > max_angle_rad or non-finite, angles float32 (N,) best angle, names list aligned to label ids 0..M-1.
    """
    spectra = np.asarray(spectra)
    if spectra.ndim != 2:
        raise ValueError("spectra must be (N, B)")
    n = spectra.shape[0]
    names = list(library.keys())
    if not names:
        return (
            np.full(n, -1, dtype=np.int32),
            np.full(n, np.nan, dtype=np.float32),
            names,
        )

    targets = np.stack(
        [np.asarray(library[name], dtype=np.float64) for name in names],
        axis=0,
    )
    ang = spectral_angles(spectra, targets)
    nearest = np.argmin(np.where(np.isfinite(ang), ang, np.inf), axis=1)
    best = ang[np.arange(n), nearest]
    labels = nearest.astype(np.int32, copy=True)
    labels[(~np.isfinite(best)) | (best > max_angle_rad)] = np.int32(-1)
    return labels, best.astype(np.float32), names
