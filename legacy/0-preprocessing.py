# EMIT L2B Mineral Visualization
# Uses REAL mineral IDs from EMIT mineral product
# Saves HIGH-RES JPG

# ---------------------------------------------------
# INSTALLS
# ---------------------------------------------------
# pip install xarray netCDF4 matplotlib numpy

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------------------------------
# LOAD EMIT MINERAL FILE
# ---------------------------------------------------
file_path = "EMIT_L2B_MIN_001_20260326T084318_2608506_030.nc"

print("Loading EMIT mineral dataset...")

ds = xr.open_dataset(file_path)

print("\nAvailable variables:")
print(list(ds.data_vars))

# ---------------------------------------------------
# LOAD MINERAL IDS
# ---------------------------------------------------
group1 = ds["group_1_mineral_id"].values
group2 = ds["group_2_mineral_id"].values

# Use Group 1 for visualization
mineral_map = group1.copy()

rows, cols = mineral_map.shape

print(f"\nRows: {rows}")
print(f"Cols: {cols}")

# ---------------------------------------------------
# EMIT CORE MINERALS
# ---------------------------------------------------
minerals = [
    "Calcite",
    "Chlorite",
    "Dolomite",
    "Goethite",
    "Gypsum",
    "Hematite",
    "Illite+Muscovite",
    "Kaolinite",
    "Montmorillonite",
    "Vermiculite"
]

# ---------------------------------------------------
# 10 DISTINCT COLORS
# ---------------------------------------------------
colors = [
    "#f4f1de",
    "#588157",
    "#ddb892",
    "#bc6c25",
    "#e9edc9",
    "#d00000",
    "#6a4c93",
    "#ffb703",
    "#219ebc",
    "#90be6d"
]

cmap = ListedColormap(colors)

# ---------------------------------------------------
# NORMALIZE IDs INTO 0-9
# ---------------------------------------------------
# EMIT IDs may not already be 0-9

unique_ids = np.unique(mineral_map)
unique_ids = unique_ids[np.isfinite(unique_ids)]

print("\nMineral IDs found:")
print(unique_ids)

# Create remap dictionary
id_to_index = {}

for i, mid in enumerate(unique_ids[:10]):
    id_to_index[mid] = i

# Create display image
display_map = np.full(mineral_map.shape, -1)

for mineral_id, idx in id_to_index.items():
    display_map[mineral_map == mineral_id] = idx

# ---------------------------------------------------
# PLOT
# ---------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 20))

fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

im = ax.imshow(display_map, cmap=cmap)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
ax.set_title(
    "EMIT Mineral Classification Map",
    fontsize=28,
    color="white",
    pad=20
)

ax.axis("off")

# ---------------------------------------------------
# LEGEND
# ---------------------------------------------------
handles = []

for i, mineral in enumerate(minerals):
    patch = plt.Line2D(
        [0],
        [0],
        marker='s',
        color='w',
        label=mineral,
        markerfacecolor=colors[i],
        markersize=14
    )
    handles.append(patch)

legend = ax.legend(
    handles=handles,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    fontsize=14,
    facecolor="#161b22",
    edgecolor="white"
)

for text in legend.get_texts():
    text.set_color("white")

# ---------------------------------------------------
# SAVE HIGH-RES JPG
# ---------------------------------------------------
output_file = "EMIT_Mineral_Map_HighRes.jpg"

plt.savefig(
    output_file,
    dpi=600,
    bbox_inches='tight',
    format='jpg',
    facecolor=fig.get_facecolor()
)

print(f"\nSaved image:")
print(output_file)

# ---------------------------------------------------
# SHOW
# ---------------------------------------------------
plt.show()