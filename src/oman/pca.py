"""Principal components for a hyperspectral pixel matrix.

``X`` is ``(n_pixels, n_bands)``. Rows that are not fully finite are dropped
before the scaler and PCA are fit. Columns are standardized with
:class:`sklearn.preprocessing.StandardScaler` inside the fit, matching the
older scripts that standardized spectra and then took the eigen-decomposition
of the covariance.

When the number of finite pixels exceeds :data:`PCA_FULL_FIT_MAX_ROWS`, the
scaler and PCA are fit on a random subset of :data:`PCA_SUBSAMPLE_ROWS` rows
(``random_state``) and then applied to every finite row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCA_FULL_FIT_MAX_ROWS: int = 2_000_000
PCA_SUBSAMPLE_ROWS: int = 200_000

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class PcaResult:
    """PCA of the finite rows of a hyperspectral matrix.

    Attributes
    ----------
    scores:
        ``(n_finite, k)`` scores of every fully finite row, in the same order
        as those rows appear in ``X``.
    components:
        ``(k, n_bands)`` loadings in the standardized feature space.
    explained_variance_ratio:
        ``(k,)`` fraction of standardized variance captured by each component,
        sorted descending.
    mean_:
        ``(n_bands,)`` column means used by the scaler.
    scale_:
        ``(n_bands,)`` column standard deviations used by the scaler.
    finite_row_mask:
        ``(n_pixels,)`` boolean mask. ``True`` marks rows of ``X`` that were
        fully finite and therefore have a row in ``scores``.
    """

    scores: FloatArray
    components: FloatArray
    explained_variance_ratio: FloatArray
    mean_: FloatArray
    scale_: FloatArray
    finite_row_mask: BoolArray

    @property
    def explained_variance_ratio_(self) -> FloatArray:
        """sklearn-style alias of :attr:`explained_variance_ratio`."""
        return self.explained_variance_ratio


def fit_pca(
    X: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
) -> PcaResult:
    """Standardize ``X`` and fit PCA on its fully finite rows.

    Parameters
    ----------
    X:
        Reflectance (or other spectra) with shape ``(n_pixels, n_bands)``.
        A few non-finite values are allowed; any row containing one is
        excluded from the fit and from ``scores``.
    n_components:
        Requested rank. The actual rank ``k`` is
        ``min(n_components, n_bands, n_finite - 1)``. The default request is
        10. At least two finite rows are required.
    random_state:
        Seed for the subset draw when ``n_finite`` exceeds
        :data:`PCA_FULL_FIT_MAX_ROWS`, and for sklearn's randomized SVD.

    Returns
    -------
    PcaResult
        Scores for every finite row, loadings, variance ratios, scaler
        parameters, and the finite-row mask.
    """
    spectra = np.asarray(X)
    if spectra.ndim != 2:
        raise ValueError(f"X must have shape (n_pixels, n_bands), got {spectra.shape}")
    if not np.issubdtype(spectra.dtype, np.floating):
        spectra = spectra.astype(np.float64, copy=False)

    n_pixels, n_bands = spectra.shape
    if n_pixels == 0 or n_bands == 0:
        raise ValueError("X has no pixels or no bands to decompose")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    finite_row_mask = np.isfinite(spectra).all(axis=1)
    n_finite = int(finite_row_mask.sum())
    k = int(min(n_components, n_bands, n_finite - 1))
    if k < 1:
        raise ValueError(
            "PCA needs at least 2 fully finite rows and 1 band; "
            f"got n_finite={n_finite}, n_bands={n_bands}"
        )

    finite = spectra[finite_row_mask]
    fit_rows = finite
    if n_finite > PCA_FULL_FIT_MAX_ROWS:
        rng = np.random.default_rng(random_state)
        n_draw = min(PCA_SUBSAMPLE_ROWS, n_finite)
        subset = rng.choice(n_finite, size=n_draw, replace=False)
        fit_rows = finite[subset]

    # Rank cannot exceed the rows actually passed to PCA.
    k = int(min(k, fit_rows.shape[0] - 1))
    if k < 1:
        raise ValueError(
            "PCA needs at least 2 rows in the fit sample; "
            f"got {fit_rows.shape[0]}"
        )

    scaler = StandardScaler()
    scaler.fit(fit_rows)
    standardized_fit = scaler.transform(fit_rows)

    pca = PCA(n_components=k, random_state=random_state)
    pca.fit(standardized_fit)
    scores = pca.transform(scaler.transform(finite))

    return PcaResult(
        scores=np.asarray(scores, dtype=np.float64),
        components=np.asarray(pca.components_, dtype=np.float64),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        mean_=np.asarray(scaler.mean_, dtype=np.float64).copy(),
        scale_=np.asarray(scaler.scale_, dtype=np.float64).copy(),
        finite_row_mask=finite_row_mask,
    )


def scores_to_map(
    scores_1d: np.ndarray,
    finite_row_mask: np.ndarray,
    valid_hw: np.ndarray,
) -> FloatArray:
    """Scatter 1-D scores back onto the image grid.

    Convention: ``finite_row_mask`` indexes the flattened valid pixels, so
    its length equals ``valid_hw.sum()``. Rows of ``scores_1d`` align with
    the ``True`` entries of that mask (``finite_row_mask.sum() == len(scores_1d)``).
    Pixels outside ``valid_hw``, and valid pixels whose mask entry is
    ``False`` or whose score is non-finite, are ``NaN``.

    Parameters
    ----------
    scores_1d:
        ``(n_finite,)`` scores, one per ``True`` entry of ``finite_row_mask``.
    finite_row_mask:
        Boolean mask of length ``valid_hw.sum()``.
    valid_hw:
        Boolean image mask of shape ``(rows, cols)``.

    Returns
    -------
    numpy.ndarray
        Float array with shape ``valid_hw.shape``.
    """
    spatial = np.asarray(valid_hw, dtype=bool)
    if spatial.ndim != 2:
        raise ValueError(f"valid_hw must have shape (rows, cols), got {spatial.shape}")

    mask = np.asarray(finite_row_mask, dtype=bool)
    scores = np.asarray(scores_1d, dtype=np.float64).reshape(-1)
    n_valid = int(spatial.sum())
    if mask.shape != (n_valid,):
        raise ValueError(
            "finite_row_mask length must equal valid_hw.sum() "
            f"({n_valid}), got {mask.shape}"
        )
    n_scores = int(mask.sum())
    if scores.shape != (n_scores,):
        raise ValueError(
            "scores rows must align with True entries of finite_row_mask: "
            f"expected ({n_scores},), got {scores.shape}"
        )

    image = np.full(spatial.shape, np.nan, dtype=np.float64)
    if n_scores == 0:
        return image
    placed = scores.copy()
    placed[~np.isfinite(placed)] = np.nan
    valid_flat = np.flatnonzero(spatial.ravel())
    image.ravel()[valid_flat[mask]] = placed
    return image


def rgb_from_scores(
    scores: np.ndarray,
    finite_row_mask: np.ndarray,
    valid_hw: np.ndarray,
    bands: tuple[int, int, int] = (0, 1, 2),
) -> NDArray[np.float32]:
    """Build a 2–98 percentile stretched RGB image from score columns.

    Each selected component is mapped with :func:`scores_to_map`, stretched
    independently to ``[0, 1]`` using the 2nd and 98th percentiles of its
    finite values, and cast to ``float32``. Non-finite samples stay ``0``.

    Parameters
    ----------
    scores:
        ``(n_finite, k)`` component scores.
    finite_row_mask:
        Mask whose length equals ``valid_hw.sum()``. Score rows align with
        its ``True`` entries.
    valid_hw:
        Boolean image mask of shape ``(rows, cols)``.
    bands:
        Three column indices of ``scores`` used as red, green, and blue.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(rows, cols, 3)`` with values in ``[0, 1]``.
    """
    if len(bands) != 3:
        raise ValueError(f"bands must contain 3 component indices, got {bands}")

    table = np.asarray(scores, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError(f"scores must have shape (n_finite, k), got {table.shape}")

    channels = np.empty(valid_hw.shape + (3,), dtype=np.float32)
    for channel, band in enumerate(bands):
        if band < 0 or band >= table.shape[1]:
            raise ValueError(
                f"band index {band} is outside scores with {table.shape[1]} components"
            )
        mapped = scores_to_map(table[:, band], finite_row_mask, valid_hw)
        channels[:, :, channel] = _percentile_stretch(mapped)
    return channels


def _percentile_stretch(channel: FloatArray) -> NDArray[np.float32]:
    """Map finite values to ``[0, 1]`` with a 2–98 percentile stretch.

    Non-finite inputs stay ``0``. A zero percentile range leaves those
    pixels at ``0`` rather than dividing by zero.
    """
    stretched = np.zeros(channel.shape, dtype=np.float32)
    finite = np.isfinite(channel)
    if not np.any(finite):
        return stretched

    low, high = np.nanpercentile(channel, (2, 98))
    span = float(high - low)
    if span <= 0.0:
        return stretched

    scaled = (channel - low) / span
    np.clip(scaled, 0.0, 1.0, out=scaled)
    stretched[finite] = scaled[finite].astype(np.float32, copy=False)
    return stretched
