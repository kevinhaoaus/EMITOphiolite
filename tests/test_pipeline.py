"""Synthetic checks for the Oman hyperspectral pipeline.

The cube is 30 x 20 x 25, with three spatial blocks and small Gaussian noise
from ``numpy.random.default_rng(0)``. These tests do not read an EMIT granule.

``prepare_matrix`` stores one row of ``X`` for every True pixel in ``valid``,
so ``valid.sum() == X.shape[0]``. Continuum removal can still write NaNs into
an already-kept row when the upper hull is undefined or non-positive. Those
NaNs sit inside ``X``; they do not add pixels beyond the mask.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oman.matched_filter import background_stats, invert_cov, matched_filter_scores
from oman.mnf import fit_mnf
from oman.pca import fit_pca
from oman.preprocess import good_band_mask, prepare_matrix
from oman.sam import classify, spectral_angles

WAVELENGTHS = np.linspace(500.0, 2300.0, 25)
BLOCK_SLICES = (slice(0, 10), slice(10, 20), slice(20, 30))
# Centers sit in three kept windows: VNIR, 1450-1800, and past 1950.
_DIP_CENTERS_NM = (900.0, 1600.0, 2250.0)


def _block_ids(n_rows: int = 30, n_cols: int = 20) -> np.ndarray:
    ids = np.empty((n_rows, n_cols), dtype=np.int8)
    for label, rows in enumerate(BLOCK_SLICES):
        ids[rows, :] = label
    return ids


def _make_cube(wavelengths: np.ndarray) -> np.ndarray:
    """Reflectance with a different broad absorption in each row-block."""
    baseline = 0.52 - 0.10 * (wavelengths - wavelengths[0]) / (
        wavelengths[-1] - wavelengths[0]
    )
    cube = np.empty((30, 20, wavelengths.size), dtype=np.float64)
    for label, rows in enumerate(BLOCK_SLICES):
        dip = 0.30 * np.exp(
            -0.5 * ((wavelengths - _DIP_CENTERS_NM[label]) / 320.0) ** 2
        )
        cube[rows, :, :] = baseline - dip
    rng = np.random.default_rng(0)
    cube += rng.normal(0.0, 0.004, size=cube.shape)
    return cube


def _finite_rows(prepared: dict) -> tuple[np.ndarray, np.ndarray]:
    """Continuum-removed rows that are fully finite, with block labels."""
    spectra = np.asarray(prepared["X"], dtype=np.float64)
    valid = np.asarray(prepared["valid"], dtype=bool)
    if spectra.shape[0] != int(valid.sum()):
        raise AssertionError(
            "valid.sum() must equal X.shape[0]; continuum NaNs belong in rows of X"
        )
    finite = np.isfinite(spectra).all(axis=1)
    labels = _block_ids(*valid.shape)[valid][finite]
    return spectra[finite], labels


@pytest.fixture(scope="module")
def cube() -> np.ndarray:
    return _make_cube(WAVELENGTHS)


@pytest.fixture(scope="module")
def prepared(cube: np.ndarray) -> dict:
    return prepare_matrix(cube, WAVELENGTHS, use_scene_mask=False)


def test_prepare_matrix_finite_rows(prepared: dict) -> None:
    spectra = np.asarray(prepared["X"])
    valid = np.asarray(prepared["valid"], dtype=bool)
    assert valid.shape == (30, 20)
    assert spectra.ndim == 2
    n_valid = int(valid.sum())
    # Same count: extra continuum NaNs, if any, are inside existing rows.
    assert spectra.shape[0] == n_valid
    assert n_valid >= 10
    n_finite = int(np.isfinite(spectra).all(axis=1).sum())
    assert n_finite >= 10
    assert np.asarray(prepared["wavelengths"]).shape[0] == spectra.shape[1]


def test_fit_pca_variance_and_score_shape(prepared: dict) -> None:
    spectra = np.asarray(prepared["X"], dtype=np.float64)
    result = fit_pca(spectra, n_components=5, random_state=0)
    ratios = np.asarray(result.explained_variance_ratio, dtype=np.float64)
    assert ratios.ndim == 1
    assert ratios.size >= 1
    assert np.all(np.isfinite(ratios))
    assert np.all(ratios[:-1] + 1e-8 >= ratios[1:])
    assert float(ratios.sum()) <= 1.0 + 1e-6
    n_finite = int(np.asarray(result.finite_row_mask).sum())
    k = int(ratios.shape[0])
    assert result.scores.shape == (n_finite, k)


def test_fit_mnf_keeps_spatial_shape(cube: np.ndarray) -> None:
    k = 3
    result = fit_mnf(cube, n_components=k)
    scores = np.asarray(result["scores"])
    assert scores.shape == (cube.shape[0], cube.shape[1], k)
    first = scores[:, :, 0]
    assert int(np.isfinite(first).sum()) >= 10
    assert float(np.nanstd(first)) > 1e-6


def test_matched_filter_scores_higher_on_target_block(prepared: dict) -> None:
    spectra, labels = _finite_rows(prepared)
    target_label = 0
    target = spectra[labels == target_label].mean(axis=0)
    assert np.isfinite(target).all()
    mean, covariance = background_stats(spectra)
    scores = np.asarray(
        matched_filter_scores(spectra, target, mean, invert_cov(covariance)),
        dtype=np.float64,
    )
    inside = scores[labels == target_label]
    assert int(np.isfinite(inside).sum()) >= 10
    inside_mean = float(np.nanmean(inside))
    for other in (1, 2):
        outside = scores[labels == other]
        assert int(np.isfinite(outside).sum()) >= 10
        assert inside_mean > float(np.nanmean(outside))


def test_sam_recovers_block_class(prepared: dict) -> None:
    spectra, labels = _finite_rows(prepared)
    library = {
        "block0": spectra[labels == 0].mean(axis=0),
        "block1": spectra[labels == 1].mean(axis=0),
    }
    angles, names = spectral_angles(spectra, library)
    best_index, _best_angle = classify(angles, names, max_angle=1.0)
    best_index = np.asarray(best_index)
    name_to_index = {name: index for index, name in enumerate(names)}
    for name, label in (("block0", 0), ("block1", 1)):
        chosen = labels == label
        assert int(chosen.sum()) >= 10
        accuracy = float(np.mean(best_index[chosen] == name_to_index[name]))
        assert accuracy > 0.6


def test_good_band_mask_drops_1340_1450() -> None:
    mask = np.asarray(good_band_mask(WAVELENGTHS), dtype=bool)
    inside = (WAVELENGTHS >= 1340.0) & (WAVELENGTHS <= 1450.0)
    assert inside.any()
    assert not np.any(mask[inside])
    assert np.any(mask[WAVELENGTHS < 1340.0])
