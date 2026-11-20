"""Manual spatial exclude boxes loaded from JSON and applied to a valid mask."""

from __future__ import annotations

import json

import numpy as np


def load_exclude_boxes(path) -> list:
    """Return ``exclude_boxes`` from a manual-mask JSON file as a list of boxes."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    boxes = payload["exclude_boxes"]
    return [[int(value) for value in box] for box in boxes]


def apply_exclude_boxes(valid_mask, boxes) -> np.ndarray:
    """valid_mask bool (H,W). Set box interiors to False. Clip boxes to the array. Return a copy."""
    out = np.asarray(valid_mask).astype(bool, copy=True)
    if out.ndim != 2:
        raise ValueError("valid_mask must have shape (H, W)")
    height, width = out.shape
    for box in boxes:
        row_start, row_end, col_start, col_end = (int(value) for value in box)
        row_start = min(max(row_start, 0), height)
        row_end = min(max(row_end, 0), height)
        col_start = min(max(col_start, 0), width)
        col_end = min(max(col_end, 0), width)
        if row_start < row_end and col_start < col_end:
            out[row_start:row_end, col_start:col_end] = False
    return out


QUALITY_FLAGS = ("Cloud flag", "Cirrus flag", "Water flag", "Spacecraft Flag")


def load_quality_mask(
    mask_path,
    reflectance_path,
    flags=QUALITY_FLAGS,
    check_geolocation: bool = True,
    tol_deg: float = 1e-3,
):
    """Read an EMIT L2A MASK granule and return ``(bad, message)``.

    ``bad`` is a bool ``(downtrack, crosstrack)`` array that is True where any
    of ``flags`` (default cloud, cirrus, water, spacecraft) is set, or ``None``
    when the mask must not be used. Only the named flag layers of the ``mask``
    variable are read; the per-wavelength ``band_mask`` (data bands) and the
    aggregate flag are ignored. The mask is rejected when its downtrack /
    crosstrack shape differs from the reflectance cube, and, when
    ``check_geolocation`` is true, when ``location/lat`` / ``location/lon``
    differ from the reflectance granule by more than ``tol_deg`` (a granule from
    a neighbouring scene can share the 1280 x 1242 shape but not the footprint).
    """
    import h5py

    with h5py.File(str(reflectance_path), "r") as rfl, h5py.File(str(mask_path), "r") as msk:
        rfl_hw = tuple(int(v) for v in rfl["reflectance"].shape[:2])
        var = msk["mask"]
        mask_hw = tuple(int(v) for v in var.shape[:2])
        if mask_hw != rfl_hw:
            return None, f"quality mask shape {mask_hw} != reflectance {rfl_hw}; not used"
        if check_geolocation and "location" in rfl and "location" in msk:
            worst = 0.0
            for name in ("lat", "lon"):
                a = np.asarray(rfl["location"][name][()], dtype=np.float64)
                b = np.asarray(msk["location"][name][()], dtype=np.float64)
                ok = (a > -9000) & (b > -9000)
                if np.any(ok):
                    worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]))))
            if worst > tol_deg:
                return None, (
                    f"quality mask shape matches {mask_hw} but its lat/lon differ from the "
                    f"reflectance granule by up to {worst:.4f} deg (different scene); not used"
                )
        names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in msk["sensor_band_parameters"]["mask_bands"][()]
        ]
        lookup = {n.strip().lower(): i for i, n in enumerate(names)}
        bad = np.zeros(mask_hw, dtype=bool)
        used = []
        for flag in flags:
            idx = lookup.get(flag.strip().lower())
            if idx is None:
                continue
            layer = np.asarray(var[:, :, idx], dtype=np.float64)
            bad |= np.isfinite(layer) & (layer > 0) & (layer > -9000)
            used.append(names[idx])
    return bad, f"quality mask used ({', '.join(used)}): {int(bad.sum())} pixels flagged"
