"""Named reflectance windows for EMIT feature extraction.

Oman ophiolite mapping cares about serpentine (lizardite/antigorite),
carbonate, and Fe oxides. Hematite covers the Fe-oxide VNIR range.
Chlorite, carbonate, and alteration are the original SWIR windows.
Serpentine (2280–2340 nm) is the marginal addition: the narrower Mg-OH
interval of lizardite and antigorite, on the short-wavelength side of
the broader carbonate window.
"""

from __future__ import annotations

import numpy as np

WINDOWS = {
    "hematite": (550, 950),
    "chlorite": (2180, 2265),
    "carbonate": (2280, 2380),
    "alteration": (2100, 2380),
    "serpentine": (2280, 2340),
}


def band_index(wavelengths_nm, wmin, wmax) -> np.ndarray:
    """Boolean mask of bands with wavelength in ``[wmin, wmax]`` (nm, inclusive)."""
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    return (wavelengths >= wmin) & (wavelengths <= wmax)


def slice_window(spectra, wavelengths_nm, name: str) -> tuple:
    """Slice ``spectra`` on its last axis to the named window.

    Returns ``(spectra_sub, wvl_sub)``. ``spectra`` may be a spectrum,
    a ``(pixels, bands)`` matrix, or a ``(..., bands)`` cube.
    """
    spectra = np.asarray(spectra)
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    wmin, wmax = WINDOWS[name]
    keep = band_index(wavelengths, wmin, wmax)
    if keep.ndim != 1 or spectra.shape[-1] != keep.shape[0]:
        raise ValueError(
            "wavelengths_nm must be 1-D and match the last axis of spectra"
        )
    return spectra[..., keep], wavelengths[keep]
