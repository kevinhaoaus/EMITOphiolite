"""Read EMIT L2A reflectance cubes from netCDF.
Returns a float32 (H, W, B) cube and wavelengths in nanometers.
Fill values at or below -999 and non-finite samples become NaN.
Row helpers report scene shape and yield contiguous row chunks.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import xarray as xr

_ENGINES = ("h5netcdf", "netcdf4", None)
_FILL_AT_OR_BELOW = -999.0
_MICRONS_IF_MAX_BELOW = 100.0


def _open_dataset(path: str, group: str | None = None) -> xr.Dataset:
    last_error: Exception | None = None
    for engine in _ENGINES:
        kwargs: dict = {}
        if engine is not None:
            kwargs["engine"] = engine
        if group is not None:
            kwargs["group"] = group
        try:
            return xr.open_dataset(path, **kwargs)
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def load_reflectance(path: str) -> tuple:
    """Return (cube float32 (H,W,B) with non-finite and fill <= -999 replaced by nan, wavelengths_nm float64 (B,)).
    Wavelengths stored in microns (max < 100) must be converted to nm.
    Use xarray. Try engine h5netcdf then netcdf4 then default.
    Reflectance variable name is reflectance.
    """
    dataset = _open_dataset(path)
    try:
        cube = np.array(dataset["reflectance"].values, dtype=np.float32, copy=True)
    finally:
        dataset.close()

    band_group = _open_dataset(path, group="sensor_band_parameters")
    try:
        wavelengths = np.array(
            band_group["wavelengths"].values, dtype=np.float64, copy=True
        )
    finally:
        band_group.close()

    finite = wavelengths[np.isfinite(wavelengths)]
    if finite.size and float(finite.max()) < _MICRONS_IF_MAX_BELOW:
        wavelengths *= 1000.0

    invalid = ~np.isfinite(cube) | (cube <= _FILL_AT_OR_BELOW)
    cube[invalid] = np.nan
    return cube, wavelengths


def scene_shape(cube) -> tuple:
    height, width, bands = cube.shape
    return height, width, bands


def iter_row_chunks(cube, chunk_rows: int = 128) -> Iterator[tuple]:
    """Yield (row0, row1, chunk ndarray)."""
    rows = int(cube.shape[0])
    for row0 in range(0, rows, chunk_rows):
        row1 = min(row0 + chunk_rows, rows)
        yield row0, row1, np.asarray(cube[row0:row1])


class ReflectanceReader:
    """Lazy, row-sliceable view of an EMIT L2A ``reflectance`` variable.

    Only the rows requested by ``reader[row0:row1]`` are read from disk, so a
    1.85 GB granule never has to be loaded at once. ``row_start``/``row_stop``
    restrict the view to a spatial subset of downtrack rows; indices passed to
    ``__getitem__`` are relative to that subset. Returned chunks are float32
    ``(rows, W, B)`` with non-finite values and fills ``<= -999`` set to NaN,
    the same convention as :func:`load_reflectance`.
    """

    def __init__(self, path: str, row_start: int = 0, row_stop: int | None = None):
        import h5py

        self.path = str(path)
        self._file = h5py.File(self.path, "r")
        self._var = self._file["reflectance"]
        full_rows = int(self._var.shape[0])
        row_stop = full_rows if row_stop is None else min(int(row_stop), full_rows)
        row_start = max(int(row_start), 0)
        if row_start >= row_stop:
            raise ValueError(f"empty row range {row_start}:{row_stop}")
        self.row_start = row_start
        self.row_stop = row_stop
        self.full_shape = tuple(int(v) for v in self._var.shape)
        wavelengths = np.array(
            self._file["sensor_band_parameters"]["wavelengths"][()], dtype=np.float64
        )
        finite = wavelengths[np.isfinite(wavelengths)]
        if finite.size and float(finite.max()) < _MICRONS_IF_MAX_BELOW:
            wavelengths = wavelengths * 1000.0
        self.wavelengths = wavelengths

    @property
    def shape(self) -> tuple:
        return (self.row_stop - self.row_start, self.full_shape[1], self.full_shape[2])

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("ReflectanceReader supports contiguous row slices only")
        start, stop, _ = key.indices(self.shape[0])
        data = np.array(
            self._var[self.row_start + start : self.row_start + stop], dtype=np.float32
        )
        data[~np.isfinite(data) | (data <= _FILL_AT_OR_BELOW)] = np.nan
        return data

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_reflectance(path: str, row_start: int = 0, row_stop: int | None = None) -> ReflectanceReader:
    """Open ``path`` lazily for :func:`iter_row_chunks`. See :class:`ReflectanceReader`."""
    return ReflectanceReader(path, row_start=row_start, row_stop=row_stop)
