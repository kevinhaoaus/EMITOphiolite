"""Synthetic-cube checks for the hyperspectral pipeline.

Runnable as ``python tests/test_synthetic.py``. No netCDF and no plots.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

H, W, B = 20, 16, 40
_CHECKS = 0


def check(condition, message=""):
    global _CHECKS
    assert bool(condition), message
    _CHECKS += 1


def need(module, filename):
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == module or missing == "src" or missing.startswith("src."):
            raise AssertionError(f"missing {filename}") from exc
        raise


def _spectrum(wavelengths, center, width, depth, base=0.55):
    dip = depth * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    return base - dip


def make_scene(seed=0):
    """H=20, W=16, B=40 cube with two mineral blocks, noise, and fill."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(500.0, 2400.0, B)
    # Punch a few otherwise-clear bands into the 1400 and 1900 nm water gaps,
    # and park one SWIR band on the narrow 2320 nm feature.
    wavelengths[16] = 1400.0
    wavelengths[18] = 1415.0
    wavelengths[28] = 1900.0
    wavelengths[29] = 1920.0
    wavelengths[37] = 2320.0

    broad = _spectrum(wavelengths, 900.0, 80.0, 0.22)
    narrow = _spectrum(wavelengths, 2320.0, 18.0, 0.20)
    cube = np.broadcast_to(np.full(B, 0.55), (H, W, B)).copy()
    cube += rng.normal(0.0, 0.004, size=cube.shape)
    cube[4:10, 1:7] = broad + rng.normal(0.0, 0.004, size=(6, 6, B))
    cube[12:18, 9:15] = narrow + rng.normal(0.0, 0.004, size=(6, 6, B))
    cube[0:3, 0:3] = -9999.0
    return cube, wavelengths, broad, narrow


def _close_plots():
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


def test_good_band_mask():
    preprocess = need("src.preprocess", "src/preprocess.py")
    _, wavelengths, _, _ = make_scene()
    mask = np.asarray(preprocess.good_band_mask(wavelengths), dtype=bool)
    water = ((wavelengths > 1320.0) & (wavelengths < 1460.0)) | (
        (wavelengths > 1780.0) & (wavelengths < 1970.0)
    )
    check(bool(water.any()), "synthetic wavelengths missed the water regions")
    check(not bool(np.any(mask & water)), "good_band_mask kept water bands")
    check(bool(np.any(mask & (wavelengths < 1000.0))), "good_band_mask kept no VNIR")
    check(bool(np.any(mask & (wavelengths > 2000.0))), "good_band_mask kept no SWIR")


def test_prepare_spectra():
    preprocess = need("src.preprocess", "src/preprocess.py")
    cube, wavelengths, _, _ = make_scene()
    spectra, pixel_index = preprocess.prepare_spectra(cube, wavelengths)
    spectra = np.asarray(spectra)
    pixel_index = np.asarray(pixel_index)
    check(spectra.shape[0] < H * W, "prepare_spectra kept every pixel")
    check(spectra.shape[0] == pixel_index.shape[0], "pixel index length mismatch")
    check(spectra.shape[0] > 0, "prepare_spectra dropped every pixel")
    check(bool(np.isfinite(spectra).all()), "prepare_spectra returned non-finite values")


def test_fit_transform_pca():
    preprocess = need("src.preprocess", "src/preprocess.py")
    pca_map = need("src.pca_map", "src/pca_map.py")
    cube, wavelengths, _, _ = make_scene()
    spectra, _ = preprocess.prepare_spectra(cube, wavelengths)
    spectra = np.asarray(spectra, dtype=np.float64)
    fitted = pca_map.fit_transform_pca(spectra, n_components=8)
    scores = np.asarray(fitted["scores"])
    k = int(fitted["n_components"])
    ratio = np.asarray(fitted["explained_variance_ratio"], dtype=np.float64)
    check(scores.shape == (spectra.shape[0], k), f"PCA scores shape {scores.shape}")
    check(k <= 8, f"PCA kept {k} components")
    check(ratio.shape == (k,), "explained_variance_ratio length")
    total = float(np.sum(ratio))
    check(total > 0.0, f"explained variance sum {total}")
    check(total <= 1.0001, f"explained variance sum {total}")


def test_fit_mnf():
    mnf = need("src.mnf", "src/mnf.py")
    cube, _, _, _ = make_scene()
    valid = np.any(cube != -9999.0, axis=-1)
    fitted = mnf.fit_mnf(cube, valid, n_components=8)
    scores = np.asarray(mnf.transform_mnf(cube, valid, fitted))
    eigenvalues = np.asarray(fitted["eigenvalues"], dtype=np.float64)
    k = int(scores.shape[-1])
    check(scores.shape == (H, W, k), f"MNF scores shape {scores.shape}")
    check(bool(np.isfinite(scores[valid]).all()), "MNF scores not finite on valid pixels")
    check(eigenvalues.shape == (k,), f"eigenvalues shape {eigenvalues.shape}")
    check(k <= 8 and k > 0, f"MNF component count {k}")
    if k >= 2:
        check(
            bool(np.all(eigenvalues[:-1] + 1e-8 >= eigenvalues[1:])),
            f"eigenvalues not descending: {eigenvalues}",
        )
    else:
        check(True, "single MNF eigenvalue")


def test_matched_filter():
    matched = need("src.matched_filter", "src/matched_filter.py")
    cube, _, _, _ = make_scene()
    block = cube[4:10, 1:7].reshape(-1, B)
    background_mask = np.ones((H, W), dtype=bool)
    background_mask[4:10, 1:7] = False
    background_mask[12:18, 9:15] = False
    background_mask[0:3, 0:3] = False
    noise = cube[background_mask]
    target = np.mean(block, axis=0)
    background = matched.background_stats(noise)
    block_scores = np.asarray(matched.matched_filter(block, target, background=background))
    noise_scores = np.asarray(matched.matched_filter(noise, target, background=background))
    check(bool(np.isfinite(block_scores).all()), "matched filter scores not finite on the block")
    check(bool(np.isfinite(noise_scores).all()), "matched filter scores not finite on noise")
    check(
        float(np.mean(block_scores)) > float(np.mean(noise_scores)),
        f"block mean {np.mean(block_scores)} vs noise mean {np.mean(noise_scores)}",
    )


def test_sam_classify():
    sam = need("src.sam", "src/sam.py")
    _, _, broad, narrow = make_scene()
    library = {"broad_900": broad, "narrow_2320": narrow}
    labels, angles, names = sam.sam_classify(broad.reshape(1, -1), library)
    label = int(np.asarray(labels).reshape(-1)[0])
    angle = float(np.asarray(angles).reshape(-1)[0])
    check(0 <= label < len(names), f"SAM label {label}")
    check(names[label] == "broad_900", f"SAM chose {names[label] if 0 <= label < len(names) else label}")
    check(abs(angle) < 1e-4, f"SAM angle {angle} is not near 0")


def test_apply_exclude_boxes():
    masks = need("src.masks", "src/masks.py")
    valid = np.ones((H, W), dtype=bool)
    boxes = [[4, 7, 2, 5], [-3, 2, 14, 40]]
    masked = masks.apply_exclude_boxes(valid, boxes)
    check(masked.shape == valid.shape, "exclude mask changed shape")
    check(not bool(masked[4:7, 2:5].any()), "exclude box left pixels True")
    check(bool(valid.all()), "apply_exclude_boxes modified the input mask")
    check(not bool(masked[0:2, 14:16].any()), "hanging box was not clipped onto the array")
    check(bool(masked[10, 8]), "exclude boxes cleared a pixel outside every box")


def test_slice_window_serpentine():
    preprocess = need("src.preprocess", "src/preprocess.py")
    windows = need("src.feature_windows", "src/feature_windows.py")
    cube, wavelengths, _, _ = make_scene()
    mask = np.asarray(preprocess.good_band_mask(wavelengths), dtype=bool)
    good = wavelengths[mask]
    sliced, sliced_w = windows.slice_window(cube[:, :, mask], good, "serpentine")
    sliced_w = np.asarray(sliced_w)
    check(sliced_w.shape[0] < good.shape[0], "serpentine window is not narrower than the good bands")
    check(sliced_w.shape[0] >= 1, "serpentine window contained no bands")
    check(np.asarray(sliced).shape[-1] == sliced_w.shape[0], "sliced spectra do not match wavelengths")


def run_all():
    test_good_band_mask()
    test_prepare_spectra()
    test_fit_transform_pca()
    test_fit_mnf()
    test_matched_filter()
    test_sam_classify()
    test_apply_exclude_boxes()
    test_slice_window_serpentine()


if __name__ == "__main__":
    try:
        run_all()
        print("ok", _CHECKS)
    finally:
        _close_plots()
