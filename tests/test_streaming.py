"""Checks for the streaming (row-chunk) path used by pipeline.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import mnf, pca_map, preprocess  # noqa: E402


def _cube(seed=0, h=24, w=18, b=30):
    rng = np.random.default_rng(seed)
    wl = np.linspace(500.0, 2400.0, b)
    base = 0.5 - 0.1 * (wl - 500.0) / 1900.0
    cube = base + rng.normal(0.0, 0.01, size=(h, w, b))
    cube[: h // 2] -= 0.15 * np.exp(-0.5 * ((wl - 2300.0) / 60.0) ** 2)
    return cube.astype(np.float32), wl


def test_transform_pca_matches_fit_scores():
    cube, wl = _cube()
    spectra, _ = preprocess.prepare_spectra(cube, wl)
    fitted = pca_map.fit_transform_pca(spectra, n_components=4)
    again = pca_map.transform_pca(spectra, fitted)
    assert np.allclose(again, fitted["scores"], atol=1e-4)


def test_global_shadow_threshold_matches_percentile():
    cube, wl = _cube(1)
    bright = preprocess.swir_brightness(cube, wl)
    thr = np.percentile(bright[np.isfinite(bright)], 8.0)
    a, ia = preprocess.prepare_spectra(cube, wl)
    b, ib = preprocess.prepare_spectra(cube, wl, shadow_threshold=thr)
    assert np.array_equal(ia, ib)
    assert np.allclose(a, b, equal_nan=True)


def test_fit_mnf_from_samples_matches_fit_mnf():
    cube, _ = _cube(2)
    cube = cube.astype(np.float64)
    valid = np.ones(cube.shape[:2], dtype=bool)
    ref = mnf.fit_mnf(cube, valid, n_components=5)
    diffs = (cube[:, 1:] - cube[:, :-1]).reshape(-1, cube.shape[-1])
    noise = mnf.noise_covariance_from_differences(diffs, cube.shape[-1])
    got = mnf.fit_mnf_from_samples(cube.reshape(-1, cube.shape[-1]), noise, n_components=5)
    assert np.allclose(got["eigenvalues"], ref["eigenvalues"])
    assert np.all(np.diff(got["eigenvalues"]) <= 1e-8)
    s_ref = mnf.transform_mnf(cube, valid, ref).reshape(-1, 5)
    s_got = mnf.transform_mnf_pixels(cube.reshape(-1, cube.shape[-1]), got)
    assert np.allclose(np.abs(s_ref), np.abs(s_got), atol=1e-4)


def test_pipeline_streams_small_granule(tmp_path):
    import h5py

    import pipeline

    cube, wl = _cube(3, h=40, w=20, b=60)
    cube[0:2, 0:3] = -9999.0
    path = tmp_path / "rfl.nc"
    with h5py.File(path, "w") as h:
        h["reflectance"] = cube
        h.create_group("sensor_band_parameters")["wavelengths"] = (wl / 1000.0).astype(np.float32)
    boxes = tmp_path / "boxes.json"
    boxes.write_text('{"exclude_boxes": [[5, 9, 2, 6]]}', encoding="utf-8")
    out = tmp_path / "out"
    pipeline.main([
        "--reflectance", str(path), "--usgs", str(ROOT / "usgs"), "--out", str(out),
        "--mask-boxes", str(boxes), "--components", "4", "--chunk-rows", "7", "--max-fit-pixels", "200",
    ])
    for name in ("pca_rgb.png", "mnf_rgb.png", "sam_labels.png", "mf_best_labels.png"):
        assert (out / name).stat().st_size > 1000
    pca = np.load(out / "maps" / "pca_scores.npy")
    assert pca.shape[:2] == (40, 20) and pca.dtype == np.float32
    assert np.isnan(pca[5:9, 2:6]).all() and np.isnan(pca[0:2, 0:3]).all()
    assert np.isfinite(pca).any()
