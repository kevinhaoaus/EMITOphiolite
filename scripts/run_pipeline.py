"""Run PCA, MNF, matched filtering, and SAM on an EMIT cube or a synthetic scene.

The src directory is inserted on sys.path so this script runs without an
editable install. Analysis functions live in oman.*; this file only wires them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oman.io import load_mineral_id, load_quality_mask, load_reflectance
from oman.library import apply_library_continuum, load_library
from oman.matched_filter import run_matched_filters
from oman.matched_filter import scores_to_map as mf_scores_to_map
from oman.mnf import fit_mnf, mnf_rgb
from oman.pca import fit_pca, rgb_from_scores, scores_to_map
from oman.plotting import (
    save_class_map,
    save_rgb,
    save_scalar_map,
    save_spectra,
    save_variance,
)
from oman.preprocess import (
    FEATURE_WINDOWS,
    continuum_remove,
    good_band_mask,
    prepare_matrix,
    subset_bands,
)
from oman.sam import classification_map, classify, spectral_angles

# Keep the name available: mineral-id products are a separate EMIT input and
# are not part of this reflectance CLI.
_MINERAL_ID_LOADER = load_mineral_id

MAX_FULL_MATCHED_FILTERS = 12
MIN_FINITE_BANDS = 5

# Substrings that make a library mineral relevant to each feature window.
WINDOW_MINERAL_PARTS = {
    "hematite": ("hematite", "goethite", "magnetite"),
    "chlorite": ("chlorite",),
    "carbonate": ("calcite", "dolomite"),
    "alteration": ("muscovite", "kaolinite", "montmorillonite", "chlorite"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PCA, MNF, matched filter, and SAM for an EMIT reflectance cube. "
            "Omit --reflectance, or pass --synthetic, to run a small synthetic scene."
        )
    )
    parser.add_argument(
        "--reflectance",
        type=Path,
        default=None,
        help="EMIT L2A reflectance netCDF. Omit to build a synthetic cube.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional EMIT L2A quality mask netCDF. Used only for a real scene.",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("data/raw/usgs"),
        help="Folder of USGS splib07 text spectra (default: data/raw/usgs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Directory for PNG maps and summary.json (default: outputs).",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=8,
        help="Number of PCA and MNF components (default: 8).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Build a synthetic cube even if --reflectance is set.",
    )
    return parser.parse_args(argv)


def resolve_library(path: Path) -> Path:
    if path.is_dir():
        return path
    rooted = ROOT / path
    if rooted.is_dir():
        return rooted
    return path


def synthetic_scene() -> tuple[np.ndarray, np.ndarray]:
    """80 x 60 x 40 cube, 450–2400 nm, three absorption blocks, plus noise."""
    rows, cols, bands = 80, 60, 40
    wavelengths = np.linspace(450.0, 2400.0, bands)
    continuum = 0.2 + 0.35 * np.exp(-0.5 * ((wavelengths - 1100.0) / 1200.0) ** 2)
    cube = np.broadcast_to(continuum, (rows, cols, bands)).astype(np.float32).copy()

    blocks = (
        (slice(0, 26), slice(0, 20), 880.0, 45.0, 0.55),
        (slice(27, 53), slice(20, 40), 2240.0, 30.0, 0.4),
        (slice(54, 80), slice(40, 60), 2335.0, 25.0, 0.45),
    )
    for row_slice, col_slice, center, width, depth in blocks:
        dip = depth * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        cube[row_slice, col_slice, :] *= (1.0 - dip).astype(np.float32)

    noise = np.random.default_rng(0).normal(0.0, 0.008, size=cube.shape)
    cube = np.clip(cube + noise.astype(np.float32), 0.0, 1.0).astype(np.float32)
    return cube, wavelengths


def load_scene(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return cube, wavelengths_nm, and whether the cube is synthetic."""
    if args.synthetic or args.reflectance is None:
        if args.reflectance is not None and args.synthetic:
            print("Using the synthetic cube because --synthetic is set.")
        cube, wavelengths = synthetic_scene()
        print(f"Synthetic cube {cube.shape}, wavelengths {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm")
        return cube, wavelengths, True

    reflectance_path = args.reflectance
    if not reflectance_path.is_file():
        rooted = ROOT / reflectance_path
        if rooted.is_file():
            reflectance_path = rooted
    cube, wavelengths = load_reflectance(reflectance_path)
    print(f"Loaded reflectance {tuple(cube.shape)} from {reflectance_path}")
    return cube, np.asarray(wavelengths, dtype=np.float64), False


def apply_quality_mask(prepared: dict, mask_path: Path | None, synthetic: bool) -> str:
    """Drop flagged pixels when a real scene has a matching quality mask.

    load_quality_mask is True where an EMIT flag band is set, so those pixels
    are removed from the valid set rather than kept.
    """
    if synthetic or mask_path is None:
        return "No quality mask applied."

    valid = np.asarray(prepared["valid"], dtype=bool)
    flagged = np.asarray(load_quality_mask(mask_path), dtype=bool)
    if flagged.shape != valid.shape:
        print(
            f"Warning: quality mask shape {tuple(flagged.shape)} does not match "
            f"valid pixels {tuple(valid.shape)}; skipping the mask."
        )
        return "Quality mask skipped because its shape does not match the cube."

    keep = valid & ~flagged
    matrix = np.asarray(prepared["X"])
    prepared["X"] = matrix[keep[valid]]
    prepared["valid"] = keep
    removed = int(valid.sum() - keep.sum())
    print(f"Quality mask removed {removed} flagged pixels; {int(keep.sum())} remain.")
    return f"Quality mask removed {removed} flagged pixels."


def iter_target_groups(
    targets: dict[str, np.ndarray],
    band_mask: np.ndarray | None = None,
    parts: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Group spectra that share a finite band set.

    Each group is ``(band_mask, {name: finite spectrum})``. A mineral is
    skipped when fewer than 5 of the requested bands are finite. The matched
    filter requires a fully finite target, so bands that are NaN in that
    spectrum are left out of its group.
    """
    chosen: list[tuple[str, np.ndarray]] = []
    for name in sorted(targets):
        if parts is not None and not any(part in name.lower() for part in parts):
            continue
        spectrum = np.asarray(targets[name], dtype=np.float64).reshape(-1)
        finite = np.isfinite(spectrum)
        if band_mask is not None:
            finite = finite & np.asarray(band_mask, dtype=bool)
        if int(finite.sum()) < MIN_FINITE_BANDS:
            continue
        chosen.append((name, finite))
        if limit is not None and len(chosen) >= limit:
            break

    groups: dict[bytes, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for name, finite in chosen:
        band_key = finite.tobytes()
        if band_key not in groups:
            groups[band_key] = (finite, {})
        spectrum = np.asarray(targets[name], dtype=np.float64).reshape(-1)
        groups[band_key][1][name] = spectrum[finite]
    return list(groups.values())


def scatter_mf(scores: np.ndarray, valid: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map a matched-filter score vector onto the image.

    Scores may line up with every valid pixel (NaN where X is non-finite) or
    with only the fully finite rows of the matrix that was filtered.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    spatial = np.asarray(valid, dtype=bool)
    n_valid = int(spatial.sum())
    if values.size == n_valid:
        return mf_scores_to_map(values, spatial, spatial.shape)

    finite_rows = np.isfinite(matrix).all(axis=1)
    if finite_rows.shape[0] == n_valid and values.size == int(finite_rows.sum()):
        tight = np.zeros(spatial.shape, dtype=bool)
        tight.ravel()[np.flatnonzero(spatial.ravel())[finite_rows]] = True
        return mf_scores_to_map(values, tight, spatial.shape)

    raise ValueError(
        f"Matched-filter scores length {values.size} matches neither "
        f"{n_valid} valid pixels nor {int(np.isfinite(matrix).all(axis=1).sum())} finite rows."
    )


def file_token(name: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in name)
    token = "_".join(part for part in token.split("_") if part)
    return token or "mineral"


def window_band_mask(wavelengths: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    waves = np.asarray(wavelengths, dtype=np.float64)
    inside = (waves >= float(low)) & (waves <= float(high))
    good = np.asarray(good_band_mask(waves), dtype=bool)
    if good.shape == inside.shape:
        inside = inside & good
    return inside


def save_matched_filters(
    out_dir: Path,
    matrix: np.ndarray,
    valid: np.ndarray,
    targets: dict[str, np.ndarray],
    prefix: str,
    title_prefix: str,
) -> list[str]:
    """Score finite rows only. ``targets`` must already be finite and band-aligned."""
    if not targets:
        return []
    finite_rows = np.isfinite(matrix).all(axis=1)
    usable = np.asarray(matrix)[finite_rows]
    if usable.shape[0] < 2:
        print(f"Skipping {title_prefix.strip()}: fewer than 2 finite spectra.")
        return []
    score_map = run_matched_filters(usable, targets)
    written: list[str] = []
    for name in sorted(score_map):
        image = scatter_mf(score_map[name], valid, matrix)
        filename = f"{prefix}{file_token(name)}.png"
        save_scalar_map(out_dir / filename, image, f"{title_prefix}{name}")
        written.append(filename)
    return written


def save_target_groups(
    out_dir: Path,
    matrix: np.ndarray,
    valid: np.ndarray,
    groups: list[tuple[np.ndarray, dict[str, np.ndarray]]],
    prefix: str,
    title_prefix: str,
) -> list[str]:
    written: list[str] = []
    for band_mask, group in groups:
        written.extend(
            save_matched_filters(
                out_dir,
                np.asarray(matrix)[:, band_mask],
                valid,
                group,
                prefix=prefix,
                title_prefix=title_prefix,
            )
        )
    return written


def feature_window_filters(
    out_dir: Path,
    matrix: np.ndarray,
    valid: np.ndarray,
    wavelengths: np.ndarray,
    targets: dict[str, np.ndarray],
) -> list[str]:
    written: list[str] = []
    for window_name, bounds in FEATURE_WINDOWS.items():
        parts = WINDOW_MINERAL_PARTS.get(str(window_name).lower())
        if not parts:
            continue
        low, high = bounds
        band_mask = window_band_mask(wavelengths, (float(low), float(high)))
        groups = iter_target_groups(targets, band_mask=band_mask, parts=parts)
        mineral_names = [name for _mask, group in groups for name in group]
        if not mineral_names:
            print(f"Skipping feature window {window_name}: no relevant mineral with enough finite bands.")
            continue
        print(
            f"Matched filter {window_name} ({int(band_mask.sum())} bands): "
            + ", ".join(mineral_names)
        )
        written.extend(
            save_target_groups(
                out_dir,
                matrix,
                valid,
                groups,
                prefix=f"mf_{file_token(str(window_name))}_",
                title_prefix=f"MF {window_name} ",
            )
        )
    return written


def run(args: argparse.Namespace) -> dict:
    if args.components < 1:
        raise SystemExit("--components must be at least 1")

    cube, wavelengths_nm, synthetic = load_scene(args)
    prepared = prepare_matrix(
        cube,
        wavelengths_nm,
        use_scene_mask=not synthetic,
    )
    mask_note = apply_quality_mask(prepared, args.mask, synthetic)

    valid = np.asarray(prepared["valid"], dtype=bool)
    matrix = np.asarray(prepared["X"])
    wavelengths = np.asarray(prepared["wavelengths"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != int(valid.sum()):
        raise SystemExit(
            f"prepare_matrix returned X shape {matrix.shape} for {int(valid.sum())} valid pixels."
        )
    if int(valid.sum()) < 2:
        raise SystemExit("Fewer than 2 valid pixels remain after preprocessing.")

    library_dir = resolve_library(args.library)
    library = load_library(library_dir, wavelengths)
    targets = apply_library_continuum(library, wavelengths, continuum_remove)
    print(f"Library {library_dir}: {len(targets)} continuum-removed spectra")

    out_dir = args.out
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pca = fit_pca(matrix, n_components=args.components)
    if pca.scores.shape[1] < 3:
        raise SystemExit(
            f"PCA produced {pca.scores.shape[1]} components; RGB needs at least 3. "
            "Raise --components."
        )
    pc1 = scores_to_map(pca.scores[:, 0], pca.finite_row_mask, valid)
    pca_rgb = rgb_from_scores(pca.scores, pca.finite_row_mask, valid)
    save_rgb(out_dir / "pca_rgb.png", pca_rgb, "PCA RGB")
    save_variance(out_dir / "pca_variance.png", pca.explained_variance_ratio_, "PCA explained variance")
    save_scalar_map(out_dir / "pca_pc1.png", pc1, "PCA PC1")

    good_cube, _good_wavelengths = subset_bands(cube, wavelengths_nm)
    good_cube = np.array(good_cube, dtype=np.float32, copy=True)
    if good_cube.shape[:2] != valid.shape:
        raise SystemExit(
            f"Good-band cube shape {good_cube.shape[:2]} does not match valid {valid.shape}."
        )
    good_cube[~valid] = np.nan
    mnf = fit_mnf(good_cube, n_components=args.components)
    mnf_scores = np.asarray(mnf["scores"])
    save_rgb(out_dir / "mnf_rgb.png", mnf_rgb(mnf_scores), "MNF RGB")
    save_scalar_map(out_dir / "mnf_c1.png", mnf_scores[:, :, 0], "MNF C1")

    full_groups = iter_target_groups(targets, limit=MAX_FULL_MATCHED_FILTERS)
    full_names = [name for _mask, group in full_groups for name in group]
    print("Full-spectrum matched filters: " + ", ".join(full_names))
    save_target_groups(
        out_dir,
        matrix,
        valid,
        full_groups,
        prefix="mf_",
        title_prefix="MF ",
    )
    feature_window_filters(out_dir, matrix, valid, wavelengths, targets)

    angles, names = spectral_angles(matrix, targets)
    best_index, best_angle = classify(angles, names)
    class_image = classification_map(best_index, valid)
    angle_image = mf_scores_to_map(np.asarray(best_angle, dtype=np.float64), valid, valid.shape)
    save_class_map(out_dir / "sam_map.png", class_image, names, "SAM classification")
    save_scalar_map(out_dir / "sam_best_angle.png", angle_image, "SAM best angle")
    save_spectra(
        out_dir / "library_spectra.png",
        wavelengths,
        targets,
        "USGS library, continuum removed",
        "Continuum-removed reflectance",
    )

    if synthetic:
        note = (
            "Synthetic 80x60x40 cube from 450 to 2400 nm with three Gaussian "
            "absorption blocks plus noise; scene mask off. " + mask_note
        )
    else:
        note = "Real EMIT reflectance with the manual scene mask on. " + mask_note

    summary = {
        "shapes": {
            "cube": list(np.asarray(cube).shape),
            "wavelengths_nm": list(np.asarray(wavelengths_nm).shape),
            "prepared_wavelengths": list(wavelengths.shape),
            "valid": list(valid.shape),
            "X": list(matrix.shape),
            "pca_scores": list(np.asarray(pca.scores).shape),
            "mnf_scores": list(mnf_scores.shape),
        },
        "n_valid_pixels": int(valid.sum()),
        "pca_explained_variance_ratio": [
            float(value) for value in np.asarray(pca.explained_variance_ratio_).ravel()
        ],
        "note": " ".join(note.split()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
