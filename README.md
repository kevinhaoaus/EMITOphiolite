# EMIT Oman ophiolite pipeline

This pipeline reads an EMIT L2A reflectance cube and writes mineral-mapping products: a principal-component RGB, a real minimum noise fraction (MNF) RGB, a matched filter against the USGS spectral library, a spectral angle mapper (SAM), and a PCA RGB inside each diagnostic wavelength window (hematite, chlorite, carbonate, alteration, and serpentine).

## How to run

Use Python 3.11 (`py -3.11`); the default `python` (3.14) does not have the science stack. From the project root:

```
py -3.11 pipeline.py --reflectance data\EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc --usgs usgs --out outputs --mask-boxes data\manual_mask_boxes.json --components 8
```

Smoke test on the first 256 downtrack rows:

```
py -3.11 pipeline.py --reflectance data\EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc --usgs usgs --out outputs\subset --mask-boxes data\manual_mask_boxes.json --components 8 --rows 0:256
```

Tests: `py -3.11 -m pytest tests` (or `py -3.11 tests\test_synthetic.py`).

Flags:

- `--reflectance`, `--usgs`, `--out` (required).
- `--mask-boxes` JSON of half-open `[row0, row1, col0, col1)` exclude boxes in full-granule coordinates (`data\manual_mask_boxes.json` is for granule `..._2605105_013` only).
- `--components` (default 8): PCA and MNF components. Diagnostic windows always use at most 5.
- `--max-fit-pixels` (default 80000): cap on sampled spectra for the PCA fit, the MNF signal covariance, and the matched-filter background.
- `--chunk-rows` (default 128): downtrack rows read per chunk.
- `--noise-fit-pixels` (default `max-fit-pixels / 2`): cap on sampled along-track difference spectra for the MNF noise covariance.
- `--rows START:STOP`: process only a downtrack row range (half-open).
- `--quality-mask PATH`: an EMIT L2A MASK granule. Only the cloud, cirrus, water, and spacecraft flag layers are used (never the per-band data mask). It is applied only if its downtrack/crosstrack shape matches the cube **and** its `location/lat`/`lon` agree within 0.001 deg; `--force-quality-mask` skips the lat/lon check. Note: `data\EMIT_L2A_MASK_001_20260220T072240_2605105_012.nc` has the same 1280 x 1242 shape as `..._013` but belongs to the previous scene (footprint offset about 0.57 deg), so the pipeline reports it and does not use it.
- `--seed` (default 0): random seed for the fit samples.

Outputs land in `--out`: `pca_rgb.png`, `pca_explained.txt`, `mnf_rgb.png`, `mnf_eigenvalues.txt`, `mf_{name}.png`, `mf_best_labels.png`, `sam_labels.png`, `sam_angles.png`, and `window_{name}_pca_rgb.png`. Score maps are kept on disk as `.npy` memmaps in `--out\maps\`: `pca_scores.npy` (H, W, k), `mnf_scores.npy` (H, W, k), `mf_{name}.npy` (H, W), `sam_labels.npy` (H, W, int16, -1 = unlabeled), `sam_angles.npy` (H, W, radians), and `window_{name}_pca.npy` (H, W, <=5). Non-processed pixels are NaN.

## Memory-safe chunking

The reflectance cube is never loaded whole. `src.io_cube.open_reflectance` opens the `reflectance` variable lazily with h5py (fills <= -999 become NaN, wavelengths in nm), and `src.io_cube.iter_row_chunks` yields `--chunk-rows` downtrack rows at a time.

1. **Pass 1** (raw reflectance, cheap): for each chunk, drop bad bands (`good_band_mask`), apply the exclude boxes and the optional quality mask, record per-pixel SWIR brightness, and reservoir-sample (uniform, without replacement) up to `--max-fit-pixels` finite pixels for the MNF signal covariance and up to `--noise-fit-pixels` one-pixel along-track (downtrack) differences where both pixels are valid, including pairs that straddle chunk boundaries.
2. The 8th-percentile SWIR shadow threshold is taken once over the whole scene (or row range) and passed to `prepare_spectra(..., shadow_threshold=...)`, so every chunk uses the same cutoff instead of a per-chunk percentile.
3. A random sample of non-shadow pixels is gathered row-chunk by row-chunk and run through `prepare_spectra` (NDVI > 0.25 removed, Savitzky-Golay 11/2, upper-hull continuum removal). Up to `--max-fit-pixels` finite prepared spectra fit the PCA, the matched-filter background (Ledoit-Wolf), and the five window PCAs. The MNF is solved with `scipy.linalg.eigh(signal_cov, noise_cov)`, eigenvalues descending.
4. **Pass 2**: each chunk is prepared again with the same threshold, and every valid pixel is projected (PCA, MNF, matched filter, SAM, window PCAs) straight into the `.npy` memmaps.
5. The PNGs are rendered from those memmaps with `src.plotting` (independent 2-98 % stretches for the RGBs).

Peak memory on the full `..._013` scene is about 2 GB with the defaults.

## L2B mineral quicklook

```
py -3.11 scripts\mineral_quicklook.py
```

writes `outputs\emit_l2b_mineral_map.png` from `data\EMIT_L2B_MIN_001_20260326T084318_2608506_030.nc` (`group_1_mineral_id`, int16, 1920 x 1242). The 12 most common ids are colored. Names come from `mineral_metadata/name`, joined on `mineral_metadata/index == map id` (the index starts at 1, so the array position is not the id). An id with no matching index is shown as its number, id 0 is labeled "unclassified", and the fill value -9999 is not drawn.

## Copied inputs

These inputs were copied with the project:

- EMIT L2A reflectance and mask netCDFs in `data/`
- EMIT L2B mineral product in `data/`
- USGS splib07 AVIRIS spectra in `usgs/`
- Geologic map GeoTIFFs in `data/maps/`

The numbered scripts from the earlier project are in `legacy/`.

## What changed from the legacy scripts

- MNF is a noise-adjusted transform (along-track shift noise, then the generalized eigenproblem). The legacy scripts saved a covariance PCA under the name MNF.
- SAM is vectorized over pixels. The legacy mapper looped in Python.
- The matched filter estimates the background covariance with Ledoit-Wolf shrinkage.
- Bad-band selection, smoothing, shadow and NDVI filtering, and continuum removal are one shared preprocessing path.
- Diagnostic windows include a serpentine interval (2280-2340 nm) for lizardite and antigorite, along with hematite, chlorite, carbonate, and alteration.
