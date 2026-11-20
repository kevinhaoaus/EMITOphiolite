"""Vectorized spectral angle mapper.

Angles are computed with matrix products. Non-finite samples are replaced
with zeros and weighted by a pairwise finite-band mask, so a band contributes
only when it is finite in both the pixel and that library target.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["spectral_angles", "classify", "classification_map"]

AngleMatrix = NDArray[np.float32]
IndexArray = NDArray[np.int64]
AngleVector = NDArray[np.float64]
ClassMap = NDArray[np.int16]


def spectral_angles(
    X: np.ndarray,
    library: dict[str, np.ndarray],
) -> tuple[AngleMatrix, list[str]]:
    """Spectral angle from each pixel to each library spectrum.

    Parameters
    ----------
    X:
        Spectra with shape ``(n, bands)``.
    library:
        Mineral name to spectrum. Each spectrum has length ``bands``.
        Column order is insertion order.

    Returns
    -------
    angles:
        ``(n, m)`` float32 angles in radians. A pair is ``NaN`` when the
        pixel and target share no finite bands, or when either masked
        vector has zero length.
    names:
        Library keys, one per column of ``angles``.
    """
    pixels = np.asarray(X, dtype=np.float64)
    if pixels.ndim != 2:
        raise ValueError(f"X must have shape (n, bands), got {pixels.shape}")

    names = list(library)
    n_pixels, n_bands = pixels.shape
    n_minerals = len(names)
    if n_minerals == 0:
        return np.empty((n_pixels, 0), dtype=np.float32), names

    targets = np.empty((n_minerals, n_bands), dtype=np.float64)
    for index, name in enumerate(names):
        spectrum = np.asarray(library[name], dtype=np.float64).reshape(-1)
        if spectrum.shape != (n_bands,):
            raise ValueError(
                f"library[{name!r}] must have shape ({n_bands},), got {spectrum.shape}"
            )
        targets[index] = spectrum

    pixel_finite = np.isfinite(pixels)
    target_finite = np.isfinite(targets)
    pixels = np.where(pixel_finite, pixels, 0.0)
    targets = np.where(target_finite, targets, 0.0)
    pixel_weight = pixel_finite.astype(np.float64, copy=False)
    target_weight = target_finite.astype(np.float64, copy=False)

    # Pairwise mask weight w_ik * v_jk is applied inside the products:
    # dots ignore a band unless both spectra were finite (NaNs are already 0),
    # and each norm is summed only on bands finite in the other spectrum.
    dots = pixels @ targets.T
    pixel_norm_sq = (pixels * pixels) @ target_weight.T
    target_norm_sq = pixel_weight @ (targets * targets).T
    overlap = pixel_weight @ target_weight.T
    denom = np.sqrt(pixel_norm_sq) * np.sqrt(target_norm_sq)
    valid = (overlap > 0.0) & (denom > 0.0)

    cosang = np.full((n_pixels, n_minerals), np.nan, dtype=np.float64)
    np.divide(dots, denom, out=cosang, where=valid)
    np.clip(cosang, -1.0, 1.0, out=cosang)
    with np.errstate(invalid="ignore"):
        radians = np.arccos(cosang)
    return radians.astype(np.float32, copy=False), names


def classify(
    angles: np.ndarray,
    names: list[str],
    max_angle: float = 0.10,
) -> tuple[IndexArray, AngleVector]:
    """Index of the smallest spectral angle, if it is within ``max_angle``.

    Parameters
    ----------
    angles:
        ``(n, m)`` angles in radians, column-aligned with ``names``.
    names:
        Library names for the ``m`` columns. Length must equal ``m``.
    max_angle:
        Reject a pixel when its smallest angle is greater than this value.
        The default, ``0.10`` rad, is the historical SAM threshold.
        An angle equal to ``max_angle`` is accepted.

    Returns
    -------
    best_index:
        ``(n,)`` integer column of the closest spectrum. ``-1`` where the
        smallest angle is greater than ``max_angle`` or is not finite.
    best_angle:
        ``(n,)`` smallest angle. ``NaN`` when the pixel has no finite angle.
    """
    angle_matrix = np.asarray(angles)
    if angle_matrix.ndim != 2:
        raise ValueError(f"angles must have shape (n, m), got {angle_matrix.shape}")
    n_pixels, n_minerals = angle_matrix.shape
    if len(names) != n_minerals:
        raise ValueError(
            f"names length {len(names)} does not match angles.shape[1] ({n_minerals})"
        )

    best_index = np.full(n_pixels, -1, dtype=np.int64)
    best_angle = np.full(n_pixels, np.nan, dtype=np.float64)
    if n_pixels == 0 or n_minerals == 0:
        return best_index, best_angle

    finite = np.isfinite(angle_matrix)
    chosen = np.argmin(np.where(finite, angle_matrix, np.inf), axis=1)
    rows = np.arange(n_pixels)
    chosen_finite = finite[rows, chosen]
    best_angle = angle_matrix[rows, chosen].astype(np.float64, copy=True)
    best_angle[~chosen_finite] = np.nan

    best_index = chosen.astype(np.int64, copy=True)
    best_index[(~np.isfinite(best_angle)) | (best_angle > max_angle)] = np.int64(-1)
    return best_index, best_angle


def classification_map(
    best_index: np.ndarray,
    valid_hw: np.ndarray,
) -> ClassMap:
    """Place per-pixel labels onto the image grid.

    ``best_index`` follows the ``True`` cells of ``valid_hw`` in C order,
    and ``len(best_index)`` must equal ``valid_hw.sum()``. Cells outside
    the mask are ``-1``.

    Parameters
    ----------
    best_index:
        ``(n_valid,)`` class indices, including ``-1`` for rejected pixels.
    valid_hw:
        Boolean mask with shape ``(rows, cols)``.

    Returns
    -------
    numpy.ndarray
        int16 array with shape ``(rows, cols)``.
    """
    labels = np.asarray(best_index).reshape(-1)
    spatial = np.asarray(valid_hw, dtype=bool)
    if spatial.ndim != 2:
        raise ValueError(f"valid_hw must have shape (rows, cols), got {spatial.shape}")
    n_valid = int(spatial.sum())
    if labels.shape != (n_valid,):
        raise ValueError(
            f"best_index length {labels.shape[0]} != valid_hw.sum() ({n_valid})"
        )

    image = np.full(spatial.shape, -1, dtype=np.int16)
    image[spatial] = labels.astype(np.int16, copy=False)
    return image
