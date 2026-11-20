"""Load EMIT reflectance, wavelength, mineral-id, and quality-mask arrays.

NetCDF files are opened with h5netcdf when that package imports, and with
the netCDF4 engine otherwise. Datasets are closed before the arrays are
returned. No dask chunks are requested.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np
import xarray as xr

PathArg = str | PathLike[str]

_FILL_MAX = -9999.0
_REFLECTANCE_MAX = 1.5
_MICRON_THRESHOLD = 100.0
_DATA_MASK_BANDS = frozenset({5, 6})


def _netcdf_engine() -> str:
    """Return ``h5netcdf`` when it is installed, otherwise ``netcdf4``."""
    try:
        import h5netcdf  # noqa: F401
    except ImportError:
        return "netcdf4"
    return "h5netcdf"


def load_wavelengths(path: PathArg) -> np.ndarray:
    """Read EMIT band-center wavelengths in nanometers.

    Wavelengths live in group ``sensor_band_parameters``, variable
    ``wavelengths``. When the maximum value is below 100 the centers are
    in microns and are multiplied by 1000.

    Parameters
    ----------
    path:
        EMIT netCDF granule that contains
        ``sensor_band_parameters/wavelengths``.

    Returns
    -------
    numpy.ndarray
        Wavelengths in nanometers, shape ``(bands,)``, dtype float64.
    """
    with xr.open_dataset(
        path,
        group="sensor_band_parameters",
        engine=_netcdf_engine(),
    ) as dataset:
        wavelengths = np.array(dataset["wavelengths"].values, dtype=np.float64, copy=True)
    if np.nanmax(wavelengths) < _MICRON_THRESHOLD:
        wavelengths *= 1000.0
    return wavelengths


def load_reflectance(path: PathArg) -> tuple[np.ndarray, np.ndarray]:
    """Load an EMIT L2A reflectance cube and its wavelengths.

    The cube is the ``reflectance`` variable, unitless on a 0–1 scale.
    EMIT fill values ``<= -9999`` and values ``> 1.5`` are replaced with NaN.

    Parameters
    ----------
    path:
        EMIT L2A reflectance netCDF file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(cube, wavelengths_nm)``. ``cube`` is float32 with shape
        ``(rows, cols, bands)``. ``wavelengths_nm`` is float64 with shape
        ``(bands,)``.
    """
    with xr.open_dataset(path, engine=_netcdf_engine()) as dataset:
        cube = np.array(dataset["reflectance"].values, dtype=np.float32, copy=True)
    invalid = (cube <= _FILL_MAX) | (cube > _REFLECTANCE_MAX)
    cube[invalid] = np.nan
    return cube, load_wavelengths(path)


def load_mineral_id(path: PathArg) -> np.ndarray:
    """Load EMIT L2B ``group_1_mineral_id`` as floating-point values.

    Non-finite entries are set to NaN.

    Parameters
    ----------
    path:
        EMIT L2B mineral identification netCDF file.

    Returns
    -------
    numpy.ndarray
        Mineral ids as float64, with NaN where the source value was not finite.
    """
    with xr.open_dataset(path, engine=_netcdf_engine()) as dataset:
        mineral_id = np.array(
            dataset["group_1_mineral_id"].values,
            dtype=np.float64,
            copy=True,
        )
    mineral_id[~np.isfinite(mineral_id)] = np.nan
    return mineral_id


def load_quality_mask(
    path: PathArg,
    flag_bands: Sequence[int] = (0, 1, 2, 3, 4),
) -> np.ndarray:
    """Build a pixel quality mask from an EMIT L2A mask file.

    Reads the ``mask`` variable. A pixel is True where any selected flag
    band is greater than 0. Bands 5 and 6 are data bands, not quality
    flags, and cannot be selected.

    Parameters
    ----------
    path:
        EMIT L2A mask netCDF file.
    flag_bands:
        Quality-flag band indices. Defaults to ``(0, 1, 2, 3, 4)``.

    Returns
    -------
    numpy.ndarray
        Boolean array with shape ``(rows, cols)``.

    Raises
    ------
    ValueError
        If ``flag_bands`` includes band 5 or band 6.
    """
    bands = tuple(int(band) for band in flag_bands)
    if any(band in _DATA_MASK_BANDS for band in bands):
        raise ValueError(
            "Bands 5 and 6 are EMIT mask data bands, not quality flags."
        )
    with xr.open_dataset(path, engine=_netcdf_engine()) as dataset:
        mask = np.array(dataset["mask"].values, copy=True)
    selected = mask[:, :, bands]
    return np.any(selected > 0, axis=-1)
