"""Quicklook of EMIT L2B group 1 mineral ids.

Colors the 12 most common values in group_1_mineral_id. Legend text is
joined from mineral_metadata where index == map id (not array position).
Id 0 is labeled unclassified. The fill value -9999 is not drawn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "EMIT_L2B_MIN_001_20260326T084318_2608506_030.nc"
DEFAULT_OUTPUT = ROOT / "outputs" / "emit_l2b_mineral_map.png"
MAX_IDS = 12
FILL_VALUE = np.int16(-9999)
# Id 0 covers most of the scene; keep it grey so the mineral classes stand out.
UNCLASSIFIED_COLOR = "#d9d9d9"
OTHER_COLOR = "#6b6b6b"
PALETTE = (
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628",
    "#f781bf", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save an EMIT L2B mineral-id quicklook.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    return parser.parse_args()


def _text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "replace").strip()
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value.item())
    return str(value).strip()


def _fill_value(attrs) -> int:
    fill = attrs.get("_FillValue")
    if fill is None:
        return int(FILL_VALUE)
    return int(np.asarray(fill).reshape(-1)[0])


def load_mineral_map(path: Path) -> tuple[np.ndarray, dict[int, str], int]:
    """Return group_1 ids, {index: name} from mineral_metadata, and the fill value."""
    with h5py.File(path, "r") as handle:
        mineral = handle["group_1_mineral_id"]
        values = np.asarray(mineral[()], dtype=np.int16)
        fill_value = _fill_value(mineral.attrs)
        index = np.asarray(handle["mineral_metadata/index"][()])
        names = np.asarray(handle["mineral_metadata/name"][()])

    names_by_id = {int(raw_index): _text(raw_name) for raw_index, raw_name in zip(index, names)}
    return values, names_by_id, fill_value


def is_fill(values: np.ndarray, fill_value: int) -> np.ndarray:
    """True where the value is the product fill or the documented -9999 sentinel."""
    return (values == fill_value) | (values == FILL_VALUE)


def label_for_id(mineral_id: int, names_by_id: dict[int, str]) -> str:
    if mineral_id == 0:
        return "unclassified"
    name = names_by_id.get(mineral_id)
    if name:
        return name
    return str(mineral_id)


def top_ids(values: np.ndarray, fill_value: int, k: int = MAX_IDS) -> tuple[np.ndarray, np.ndarray]:
    valid = values[~is_fill(values, fill_value)]
    if valid.size == 0:
        return np.array([], dtype=np.int16), np.array([], dtype=np.int64)
    unique_ids, counts = np.unique(valid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if unique_ids.size > k:
        order = order[:k]
    return unique_ids[order], counts[order]


def class_colors(ids: np.ndarray) -> list[str]:
    colors: list[str] = []
    palette_i = 0
    for mineral_id in ids:
        if int(mineral_id) == 0:
            colors.append(UNCLASSIFIED_COLOR)
        else:
            colors.append(PALETTE[palette_i % len(PALETTE)])
            palette_i += 1
    return colors


def render(
    values: np.ndarray,
    ids: np.ndarray,
    labels: list[str],
    fill_value: int,
    output: Path,
    show: bool,
    other_count: int = 0,
) -> None:
    fill = is_fill(values, fill_value)
    display = np.full(values.shape, np.nan, dtype=float)
    if ids.size:
        for class_index, mineral_id in enumerate(ids):
            display[values == mineral_id] = float(class_index)
        if other_count:
            other = ~fill & np.isnan(display)
            display[other] = float(len(ids))
    elif other_count:
        display[~fill] = 0.0

    colors = class_colors(ids)
    legend_items = list(zip(colors, labels))
    if other_count:
        colors = colors + [OTHER_COLOR]
        legend_items.append((OTHER_COLOR, "other"))

    cmap = ListedColormap(colors or ["#ffffff"])
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    vmax = max(len(colors) - 0.5, 0.5)
    ax.imshow(display, cmap=cmap, interpolation="nearest", vmin=-0.5, vmax=vmax)
    if legend_items:
        handles = [
            Patch(facecolor=color, edgecolor="none", label=label)
            for color, label in legend_items
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            title="mineral",
            frameon=False,
        )
    ax.set_title("EMIT L2B group 1 mineral id")
    ax.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input NetCDF not found: {args.input}", file=sys.stderr)
        return 2

    values, names_by_id, fill_value = load_mineral_map(args.input)
    ids, counts = top_ids(values, fill_value)
    labels = [label_for_id(int(mineral_id), names_by_id) for mineral_id in ids]
    valid_count = int(np.count_nonzero(~is_fill(values, fill_value)))
    other_count = valid_count - int(counts.sum()) if ids.size else valid_count

    print("Top 12 group_1 mineral ids:")
    print(f"{'id':>6} {'count':>10}  label")
    for mineral_id, count, label in zip(ids, counts, labels):
        print(f"{int(mineral_id):>6} {int(count):>10}  {label}")
    if other_count:
        print(f"{'other':>6} {other_count:>10}  other")

    name_20 = names_by_id.get(20)
    label_20 = label_for_id(20, names_by_id)
    print(f"id 20 legend: {label_20}")
    print(f"name where index == 20: {name_20}")
    print(f"id 20 label matches index==20 name: {label_20 == name_20}")

    render(values, ids, labels, fill_value, args.output, args.show, other_count)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
