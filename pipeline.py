"""EMIT Oman ophiolite mapping: PCA, MNF, matched filter, SAM, diagnostic windows."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEARCH_NOTE = ROOT / "data" / "REFLECTANCE_SEARCH.txt"


def _load(module_name: str, names: list[str]) -> dict:
    """Import module_name and return the requested attributes.

    On failure, print the missing module name and exit 1.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or module_name
        print(f"Missing module: {missing}", file=sys.stderr)
        if missing != module_name:
            print(f"Needed while importing {module_name}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    found = {}
    for name in names:
        if not hasattr(module, name):
            print(f"Missing module: {module_name} (no attribute {name})", file=sys.stderr)
            raise SystemExit(1)
        found[name] = getattr(module, name)
    return found


def _missing_reflectance(path: Path) -> None:
    print(f"Reflectance cube not found: {path}")
    print(f"Expected filename: {path.name}")
    if SEARCH_NOTE.is_file():
        print(SEARCH_NOTE.read_text(encoding="utf-8"))
    raise SystemExit(2)


def _scatter(values: np.ndarray, pixel_index: np.ndarray, hw: tuple[int, int], fill) -> np.ndarray:
    values = np.asarray(values)
    dtype = values.dtype if np.issubdtype(values.dtype, np.integer) else np.float32
    image = np.full(hw, fill, dtype=dtype)
    if values.size == 0:
        return image
    rows = pixel_index[:, 0]
    cols = pixel_index[:, 1]
    image[rows, cols] = values.astype(dtype, copy=False)
    return image


def _write_series(path: Path, header: str, values: np.ndarray) -> None:
    lines = [header]
    for i, value in enumerate(np.asarray(values).reshape(-1), start=1):
        lines.append(f"{i}\t{float(value):.8g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finite_rows(spectra: np.ndarray, pixel_index: np.ndarray):
    keep = np.all(np.isfinite(spectra), axis=1)
    return spectra[keep], pixel_index[keep], int(keep.sum()), int(spectra.shape[0])


def _best_labels(score_map: dict, pixel_index: np.ndarray, hw: tuple[int, int]):
    names = list(score_map)
    if not names or pixel_index.shape[0] == 0:
        labels = np.full(hw, -1, dtype=np.int32)
        return labels, names
    stacked = np.column_stack([np.asarray(score_map[name], dtype=np.float64) for name in names])
    labels_1d = np.full(stacked.shape[0], -1, dtype=np.int32)
    usable = np.any(np.isfinite(stacked), axis=1)
    if np.any(usable):
        filled = np.where(np.isfinite(stacked), stacked, -np.inf)
        labels_1d[usable] = np.argmax(filled[usable], axis=1).astype(np.int32)
    return _scatter(labels_1d, pixel_index, hw, -1), names


def _peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (NaN if unavailable)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return counters.PeakWorkingSetSize / 2**20
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


class _Reservoir:
    """Uniform random sample (without replacement) of at most ``k`` rows, streamed.

    Every candidate gets a uniform random key; the ``k`` smallest keys seen so
    far are kept. Rows are only materialized for the candidates that survive
    the per-chunk cut, so a chunk never has to be copied in full.
    """

    def __init__(self, k: int, bands: int, rng: np.random.Generator):
        self.k = max(int(k), 0)
        self.rng = rng
        self.seen = 0
        self.keys = np.empty(0, dtype=np.float64)
        self.rows = np.empty((0, bands), dtype=np.float32)
        self.index = np.empty((0, 2), dtype=np.int32)

    def add(self, n: int, take) -> None:
        """``take(sel)`` returns (rows (m,B), index (m,2)) for candidate positions ``sel``."""
        if n <= 0 or self.k == 0:
            return
        self.seen += n
        keys = self.rng.random(n)
        sel = np.arange(n)
        if n > self.k:
            sel = np.argpartition(keys, self.k - 1)[: self.k]
        rows, index = take(sel)
        keys = np.concatenate([self.keys, keys[sel]])
        rows = np.concatenate([self.rows, np.asarray(rows, dtype=np.float32)])
        index = np.concatenate([self.index, np.asarray(index, dtype=np.int32)])
        if keys.shape[0] > self.k:
            keep = np.argpartition(keys, self.k - 1)[: self.k]
            keys, rows, index = keys[keep], rows[keep], index[keep]
        self.keys, self.rows, self.index = keys, rows, index


def _parse_rows(text: str | None, full_rows: int) -> tuple[int, int]:
    if not text:
        return 0, full_rows
    start, _, stop = text.partition(":")
    row0 = int(start) if start.strip() else 0
    row1 = int(stop) if stop.strip() else full_rows
    return max(row0, 0), min(row1, full_rows)


def _open_maps(directory: Path, specs: dict) -> dict:
    """Create float32/int16 .npy memmaps on disk: name -> (shape, dtype, fill)."""
    from numpy.lib.format import open_memmap

    directory.mkdir(parents=True, exist_ok=True)
    maps = {}
    for name, (shape, dtype, fill) in specs.items():
        mm = open_memmap(str(directory / f"{name}.npy"), mode="w+", dtype=dtype, shape=shape)
        mm[...] = fill
        maps[name] = mm
    return maps


def run(args: argparse.Namespace) -> None:
    started = time.time()
    reflectance = Path(args.reflectance)
    if not reflectance.is_file():
        _missing_reflectance(reflectance)

    io_cube = _load("src.io_cube", ["open_reflectance", "iter_row_chunks"])
    preprocess = _load("src.preprocess", ["good_band_mask", "prepare_spectra", "swir_brightness"])
    pca_map = _load("src.pca_map", ["fit_transform_pca", "transform_pca", "pca_rgb"])
    mnf_mod = _load(
        "src.mnf",
        ["noise_covariance_from_differences", "fit_mnf_from_samples", "transform_mnf_pixels", "mnf_rgb"],
    )
    library_mod = _load("src.library", ["load_library"])
    matched = _load("src.matched_filter", ["match_library", "background_stats"])
    plotting = _load("src.plotting", ["save_rgb", "save_score", "save_labels"])
    sam_mod = _load("src.sam", ["sam_classify"])
    windows_mod = _load("src.feature_windows", ["WINDOWS", "slice_window"])
    masks = _load("src.masks", ["load_exclude_boxes", "apply_exclude_boxes", "load_quality_mask"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunk_rows = max(int(args.chunk_rows), 1)
    max_fit = max(int(args.max_fit_pixels), 10)
    max_noise = max(int(args.noise_fit_pixels or max_fit // 2), 10)
    rng = np.random.default_rng(int(args.seed))

    # Lazy reader: only row slices are ever read from the 1.85 GB file.
    probe = io_cube["open_reflectance"](str(reflectance))
    full_rows, full_cols, _all_bands = probe.full_shape
    probe.close()
    row_start, row_stop = _parse_rows(args.rows, full_rows)
    cube = io_cube["open_reflectance"](str(reflectance), row_start, row_stop)
    wavelengths_all = np.asarray(cube.wavelengths, dtype=np.float64)
    keep = np.asarray(preprocess["good_band_mask"](wavelengths_all), dtype=bool)
    wavelengths = wavelengths_all[keep]
    height, width = cube.shape[0], cube.shape[1]
    bands = int(keep.sum())
    hw = (height, width)
    print(
        f"reflectance {cube.full_shape} rows {row_start}:{row_stop} -> scene {hw}, "
        f"good bands {bands}/{wavelengths_all.shape[0]}, chunk rows {chunk_rows}"
    )

    # Pixels to exclude, in full-granule coordinates, then cut to the row range.
    excluded_full = np.zeros((full_rows, full_cols), dtype=bool)
    if args.mask_boxes:
        box_path = Path(args.mask_boxes)
        if not box_path.is_file():
            print(f"Mask box file not found: {box_path}", file=sys.stderr)
            raise SystemExit(1)
        boxes = masks["load_exclude_boxes"](str(box_path))
        valid_full = np.asarray(masks["apply_exclude_boxes"](~excluded_full, boxes), dtype=bool)
        excluded_full |= ~valid_full
        print(f"exclude boxes {len(boxes)} excluded pixels {int((~valid_full).sum())} (half-open, full-granule coords)")
    if args.quality_mask:
        bad, message = masks["load_quality_mask"](
            args.quality_mask, str(reflectance), check_geolocation=not args.force_quality_mask
        )
        print(message)
        if bad is not None:
            excluded_full |= bad
    excluded = excluded_full[row_start:row_stop]
    del excluded_full

    def good_chunk(row0: int, row1: int, raw: np.ndarray) -> np.ndarray:
        chunk = np.ascontiguousarray(raw[:, :, keep])
        chunk[excluded[row0:row1]] = np.nan
        return chunk

    # ---- Pass 1: scene-wide SWIR brightness + MNF signal/noise reservoirs (raw reflectance).
    brightness = np.full(hw, np.nan, dtype=np.float32)
    mnf_signal = _Reservoir(max_fit, bands, rng)
    mnf_noise = _Reservoir(max_noise, bands, rng)
    prev_row = None
    prev_finite = None
    n_chunks = -(-height // chunk_rows)
    for chunk_no, (row0, row1, raw) in enumerate(io_cube["iter_row_chunks"](cube, chunk_rows), start=1):
        print(f"pass 1 chunk {chunk_no}/{n_chunks}", flush=True)
        chunk = good_chunk(row0, row1, raw)
        del raw
        finite = np.isfinite(chunk).all(axis=-1)
        brightness[row0:row1] = preprocess["swir_brightness"](chunk, wavelengths)
        rr, cc = np.nonzero(finite)
        mnf_signal.add(rr.size, lambda s: (chunk[rr[s], cc[s]], np.column_stack([rr[s] + row0, cc[s]])))
        # Along-track (downtrack, axis 0) one-pixel differences, both pixels valid.
        pr, pc = np.nonzero(finite[1:] & finite[:-1])
        mnf_noise.add(
            pr.size,
            lambda s: (chunk[pr[s] + 1, pc[s]] - chunk[pr[s], pc[s]], np.column_stack([pr[s] + row0, pc[s]])),
        )
        if prev_row is not None:  # pair that straddles the chunk boundary
            bc = np.nonzero(prev_finite & finite[0])[0]
            mnf_noise.add(
                bc.size,
                lambda s: (chunk[0, bc[s]] - prev_row[bc[s]], np.column_stack([np.full(s.size, row0 - 1), bc[s]])),
            )
        prev_row = chunk[-1].copy()
        prev_finite = finite[-1].copy()
        del chunk
    finite_bright = np.isfinite(brightness)
    if not np.any(finite_bright):
        print("No valid pixels after masking.", file=sys.stderr)
        raise SystemExit(1)
    shadow_threshold = float(np.percentile(brightness[finite_bright], 8.0))
    print(
        f"pass 1: finite pixels {int(finite_bright.sum())}, scene-wide SWIR shadow threshold "
        f"(8th pct) {shadow_threshold:.5f}, MNF signal sample {mnf_signal.rows.shape[0]}/{mnf_signal.seen}, "
        f"noise pairs {mnf_noise.rows.shape[0]}/{mnf_noise.seen}"
    )

    # ---- Fit sample of prepared spectra: random non-shadow pixels, gathered row-chunk by row-chunk.
    candidates = np.flatnonzero(finite_bright & (brightness >= shadow_threshold))
    n_pick = min(candidates.size, int(max_fit * 1.2) + 10)  # headroom for NDVI/flat drops
    picked = np.sort(rng.choice(candidates, size=n_pick, replace=False))
    del candidates
    pick_rows, pick_cols = np.divmod(picked, width)
    gathered = np.empty((picked.size, bands), dtype=np.float32)
    for row0, row1, raw in io_cube["iter_row_chunks"](cube, chunk_rows):
        lo, hi = np.searchsorted(pick_rows, [row0, row1])
        if hi > lo:
            chunk = good_chunk(row0, row1, raw)
            gathered[lo:hi] = chunk[pick_rows[lo:hi] - row0, pick_cols[lo:hi]]
            del chunk
        del raw
    sample, sample_idx = preprocess["prepare_spectra"](
        gathered[np.newaxis], wavelengths, shadow_threshold=shadow_threshold
    )
    del gathered
    sample = np.asarray(sample)
    order = rng.permutation(sample.shape[0])[:max_fit]
    sample = sample[order]
    finite_sample = sample[np.all(np.isfinite(sample), axis=1)]
    print(f"fit sample: {finite_sample.shape[0]} finite prepared spectra (cap {max_fit})")
    if finite_sample.shape[0] < 2:
        print("No valid pixels after preprocessing.", file=sys.stderr)
        raise SystemExit(1)

    pca = pca_map["fit_transform_pca"](finite_sample, n_components=args.components)
    _write_series(out / "pca_explained.txt", "component\texplained_variance_ratio", pca["explained_variance_ratio"])

    noise_cov = mnf_mod["noise_covariance_from_differences"](mnf_noise.rows, bands)
    mnf = mnf_mod["fit_mnf_from_samples"](mnf_signal.rows, noise_cov, n_components=args.components)
    _write_series(out / "mnf_eigenvalues.txt", "component\teigenvalue", mnf["eigenvalues"])
    del mnf_signal, mnf_noise

    usgs = Path(args.usgs)
    if not usgs.is_dir():
        print(f"USGS library directory not found: {usgs}", file=sys.stderr)
        raise SystemExit(1)
    library = library_mod["load_library"](str(usgs), wavelengths, continuum=True, smooth=True)
    print(f"library targets {len(library)}")
    if not library:
        print("USGS library is empty; skipping matched filter and SAM.", file=sys.stderr)
        raise SystemExit(1)
    lib_names = list(library)
    background = matched["background_stats"](finite_sample)

    window_fits = {}
    for name in windows_mod["WINDOWS"]:
        window_spectra, window_wvl = windows_mod["slice_window"](finite_sample, wavelengths, name)
        n_bands = int(window_spectra.shape[1])
        n_comp = min(5, n_bands)
        if window_spectra.shape[0] < 2 or n_comp < 1:
            print(f"Skipping window {name}: bands={n_bands}")
            continue
        window_fits[name] = (pca_map["fit_transform_pca"](window_spectra, n_components=n_comp), window_wvl)
    del finite_sample, sample

    # ---- Pass 2: prepare every chunk with the scene-wide threshold, score, write memmaps.
    k_pca = int(pca["n_components"])
    k_mnf = int(np.asarray(mnf["eigenvectors"]).shape[1])
    specs = {
        "pca_scores": ((height, width, k_pca), np.float32, np.nan),
        "mnf_scores": ((height, width, k_mnf), np.float32, np.nan),
        "sam_labels": ((height, width), np.int16, -1),
        "sam_angles": ((height, width), np.float32, np.nan),
    }
    for name in lib_names:
        specs[f"mf_{name}"] = ((height, width), np.float32, np.nan)
    for name, (fit, _w) in window_fits.items():
        specs[f"window_{name}_pca"] = ((height, width, int(fit["n_components"])), np.float32, np.nan)
    maps = _open_maps(out / "maps", specs)
    n_prepared = 0
    for chunk_no, (row0, row1, raw) in enumerate(io_cube["iter_row_chunks"](cube, chunk_rows), start=1):
        chunk = good_chunk(row0, row1, raw)
        del raw
        finite = np.isfinite(chunk).all(axis=-1)
        if np.any(finite):
            fr, fc = np.nonzero(finite)
            maps["mnf_scores"][fr + row0, fc] = mnf_mod["transform_mnf_pixels"](chunk[finite], mnf)
        spectra, pixel_index = preprocess["prepare_spectra"](chunk, wavelengths, shadow_threshold=shadow_threshold)
        del chunk
        spectra = np.asarray(spectra)
        if spectra.shape[0]:
            n_prepared += spectra.shape[0]
            rows = pixel_index[:, 0].astype(np.int64) + row0
            cols = pixel_index[:, 1].astype(np.int64)
            maps["pca_scores"][rows, cols] = pca_map["transform_pca"](spectra, pca)
            scores = matched["match_library"](spectra, library, background=background)
            for n in lib_names:
                maps[f"mf_{n}"][rows, cols] = np.asarray(scores[n], dtype=np.float32)
            labels, angles, _ = sam_mod["sam_classify"](spectra, library)
            maps["sam_labels"][rows, cols] = np.asarray(labels, dtype=np.int16)
            maps["sam_angles"][rows, cols] = np.asarray(angles, dtype=np.float32)
            for name, (fit, _w) in window_fits.items():
                window_spectra, _ = windows_mod["slice_window"](spectra, wavelengths, name)
                maps[f"window_{name}_pca"][rows, cols] = pca_map["transform_pca"](window_spectra, fit)
        print(f"pass 2 chunk {chunk_no}/{n_chunks} rows {row0 + row_start}-{row1 + row_start}: prepared {spectra.shape[0]} (elapsed {time.time() - started:.0f}s)", flush=True)
    cube.close()
    for mm in maps.values():
        mm.flush()
    print(f"pass 2: prepared spectra {n_prepared}; score memmaps in {out / 'maps'}")

    # ---- Render the documented PNGs from the on-disk maps.
    plotting["save_rgb"](str(out / "pca_rgb.png"), pca_map["pca_rgb"](np.asarray(maps["pca_scores"])), "PCA RGB")
    print(f"pca components {k_pca} -> {out / 'pca_rgb.png'}")
    plotting["save_rgb"](str(out / "mnf_rgb.png"), mnf_mod["mnf_rgb"](np.asarray(maps["mnf_scores"])), "MNF RGB")
    print(f"mnf -> {out / 'mnf_rgb.png'}")

    for name in lib_names:
        plotting["save_score"](str(out / f"mf_{name}.png"), np.asarray(maps[f"mf_{name}"]), f"Matched filter {name}")
    stacked = np.stack([np.asarray(maps[f"mf_{name}"]) for name in lib_names], axis=-1)
    usable = np.any(np.isfinite(stacked), axis=-1)
    best = np.full(hw, -1, dtype=np.int32)
    if np.any(usable):
        best[usable] = np.argmax(np.where(np.isfinite(stacked[usable]), stacked[usable], -np.inf), axis=1)
    del stacked
    plotting["save_labels"](str(out / "mf_best_labels.png"), best, lib_names, "Matched filter best target")
    print(f"matched filter maps {len(lib_names)}")

    plotting["save_labels"](str(out / "sam_labels.png"), np.asarray(maps["sam_labels"]), lib_names, "SAM labels")
    plotting["save_score"](str(out / "sam_angles.png"), np.asarray(maps["sam_angles"]), "SAM angle (rad)")
    print(f"sam -> {out / 'sam_labels.png'}")

    for name, (fit, window_wvl) in window_fits.items():
        dest = out / f"window_{name}_pca_rgb.png"
        plotting["save_rgb"](
            str(dest),
            pca_map["pca_rgb"](np.asarray(maps[f"window_{name}_pca"])),
            f"PCA RGB {name} ({window_wvl.shape[0]} bands)",
        )
        print(f"window {name} bands {window_wvl.shape[0]} -> {dest}")

    del maps
    print(f"done -> {out} in {time.time() - started:.0f}s, peak memory {_peak_memory_mb():.0f} MB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMIT Oman ophiolite reflectance pipeline")
    parser.add_argument("--reflectance", required=True, help="EMIT L2A reflectance netCDF")
    parser.add_argument("--usgs", required=True, help="Directory of USGS spectra")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--mask-boxes",
        default=None,
        help="JSON exclude boxes (row0, row1, col0, col1)",
    )
    parser.add_argument("--components", type=int, default=8, help="PCA and MNF components")
    parser.add_argument(
        "--max-fit-pixels",
        type=int,
        default=80000,
        help="Max sampled spectra for the PCA fit, MNF signal covariance, and matched-filter background",
    )
    parser.add_argument("--chunk-rows", type=int, default=128, help="Downtrack rows read per chunk")
    parser.add_argument(
        "--noise-fit-pixels",
        type=int,
        default=None,
        help="Max sampled along-track difference spectra for the MNF noise covariance (default max-fit-pixels/2)",
    )
    parser.add_argument("--rows", default=None, help="Downtrack row subset START:STOP (half-open), e.g. 0:256")
    parser.add_argument(
        "--quality-mask",
        default=None,
        help="EMIT L2A MASK netCDF; cloud/cirrus/water/spacecraft flags, used only if shape and lat/lon match",
    )
    parser.add_argument(
        "--force-quality-mask",
        action="store_true",
        help="Use --quality-mask when only the shape matches (skip the lat/lon footprint check)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the fit samples")
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
