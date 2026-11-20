"""Hyperspectral PCA for an EMIT reflectance scene.

Exact principal components via scikit-learn, optionally on spectra that have
been standardized band-wise. The fit uses a deterministic solver (covariance
eigen-decomposition when there are at least as many pixels as bands, otherwise
a full SVD) so component signs and scores do not depend on a random draw.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_transform_pca(
    spectra: np.ndarray,
    n_components: int = 8,
    standardize: bool = True,
) -> dict:
    """Fit PCA on finite spectra of shape ``(N, B)``.

    If ``standardize`` is true, a ``StandardScaler`` is fit on ``spectra`` and
    PCA is fit in that standardized space. ``n_components`` is capped at
    ``min(requested, N - 1, B)`` because a centered cloud of ``N`` spectra has
    rank at most ``N - 1``.

    Returns a dict with:

    - ``scores``: ``(N, k)`` float32 scores.
    - ``components``: ``(k, B)`` loadings. These are in the original band
      space when spectra are not standardized, and in standardized space when
      they are.
    - ``scaler_mean`` and ``scaler_scale``: length-``B`` vectors when
      standardized, otherwise ``None``.
    - ``explained_variance_ratio``: ``(k,)`` fraction of variance per component.
    - ``n_components``: ``k``.
    - ``mean``: PCA centering vector in the (possibly standardized) fit space,
      used by :func:`transform_pca` to score pixels outside the fit sample.
    """
    X = np.asarray(spectra, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("spectra must have shape (N, B)")
    if X.size and not np.isfinite(X).all():
        raise ValueError("spectra must be finite")

    n_samples, n_bands = X.shape
    requested = int(n_components)
    if requested < 0:
        raise ValueError("n_components must be non-negative")
    k = min(requested, n_bands, max(n_samples - 1, 0))

    scaler_mean = None
    scaler_scale = None
    X_fit = X
    if standardize and n_samples > 0 and n_bands > 0:
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        X_fit = np.asarray(scaler.fit_transform(X), dtype=np.float64)
        scaler_mean = np.asarray(scaler.mean_, dtype=np.float64).copy()
        scaler_scale = np.asarray(scaler.scale_, dtype=np.float64).copy()

    if k < 1:
        return {
            "scores": np.zeros((n_samples, 0), dtype=np.float32),
            "components": np.zeros((0, n_bands), dtype=np.float64),
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
            "explained_variance_ratio": np.zeros(0, dtype=np.float64),
            "n_components": 0,
            "mean": np.zeros(n_bands, dtype=np.float64),
        }

    pca, scores = _fit_exact_pca(X_fit, k)
    return {
        "scores": np.asarray(scores, dtype=np.float32),
        "components": np.asarray(pca.components_, dtype=np.float64).copy(),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ).copy(),
        "n_components": int(k),
        "mean": np.asarray(pca.mean_, dtype=np.float64).copy(),
    }


def transform_pca(spectra, fitted: dict) -> np.ndarray:
    """Project ``(N, B)`` spectra with a model from :func:`fit_transform_pca`.

    Applies the same standardization and centering as the fit, so scoring the
    fit sample reproduces ``fitted["scores"]``. Rows with any non-finite band
    come back NaN. Returns float32 ``(N, k)``.
    """
    X = np.asarray(spectra, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("spectra must have shape (N, B)")
    components = np.asarray(fitted["components"], dtype=np.float64)
    k = int(components.shape[0])
    out = np.full((X.shape[0], k), np.nan, dtype=np.float32)
    if k == 0 or X.shape[0] == 0:
        return out
    finite = np.isfinite(X).all(axis=1)
    if not np.any(finite):
        return out
    Z = X[finite]
    if fitted.get("scaler_mean") is not None:
        Z = (Z - np.asarray(fitted["scaler_mean"])) / np.asarray(fitted["scaler_scale"])
    Z = Z - np.asarray(fitted["mean"], dtype=np.float64)
    out[finite] = (Z @ components.T).astype(np.float32, copy=False)
    return out


def _fit_exact_pca(X: np.ndarray, k: int) -> tuple[PCA, np.ndarray]:
    """Fit PCA without a randomized solver.

    EMIT cubes have far more pixels than bands, so the covariance
    eigen-decomposition is the smaller exact problem. Older scikit-learn
    builds do not offer that solver; a full SVD is the exact fallback.
    """
    n_samples, n_bands = X.shape
    solvers = ("covariance_eigh", "full") if n_samples >= n_bands else ("full",)
    last_error: Exception | None = None
    for solver in solvers:
        pca = PCA(n_components=k, svd_solver=solver, whiten=False, copy=True)
        try:
            scores = pca.fit_transform(X)
        except ValueError as exc:
            message = str(exc).lower()
            if "svd_solver" not in message and solver != "covariance_eigh":
                raise
            last_error = exc
            continue
        return pca, scores
    if last_error is not None:
        raise last_error
    raise RuntimeError("PCA fit failed")


def scores_to_image(scores, pixel_index, shape_hw) -> np.ndarray:
    """Scatter ``(N, k)`` scores into an ``(H, W, k)`` image.

    ``pixel_index`` is ``(N, 2)`` row/column coordinates. Pixels that are not
    listed stay NaN.
    """
    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError("scores must have shape (N, k)")
    index = np.asarray(pixel_index)
    if index.ndim != 2 or index.shape[1] != 2:
        raise ValueError("pixel_index must have shape (N, 2)")
    if index.shape[0] != values.shape[0]:
        raise ValueError("pixel_index and scores must have the same length")

    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    n_components = values.shape[1]
    image = np.full((height, width, n_components), np.nan, dtype=np.float32)
    if values.shape[0] == 0:
        return image

    rows = index[:, 0].astype(np.int64, copy=False)
    cols = index[:, 1].astype(np.int64, copy=False)
    image[rows, cols, :] = values
    return image


def pca_rgb(score_image) -> np.ndarray:
    """PC1–PC3 as an RGB image, each axis stretched on its own 2–98 percentiles.

    The result is ``(H, W, 3)`` float32 in ``[0, 1]``. Non-finite scores become
    0. If fewer than three components are present, the available components are
    repeated (PC1, PC2, PC1, ...).
    """
    image = np.asarray(score_image)
    if image.ndim != 3:
        raise ValueError("score_image must have shape (H, W, k)")

    height, width, n_components = image.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    if n_components < 1 or height == 0 or width == 0:
        return rgb

    for channel in range(3):
        rgb[:, :, channel] = _stretch_2_98(image[:, :, channel % n_components])
    return rgb


def _stretch_2_98(band: np.ndarray) -> np.ndarray:
    """Map one component plane to ``[0, 1]`` using its finite 2nd and 98th percentiles."""
    values = np.asarray(band, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out

    low, high = np.percentile(values[finite], (2.0, 98.0))
    span = float(high - low)
    if not np.isfinite(span) or span <= 0.0:
        return out

    scaled = (values - low) / span
    np.clip(scaled, 0.0, 1.0, out=scaled)
    out[finite] = scaled[finite]
    return out
