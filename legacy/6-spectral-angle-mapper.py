import os
import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter


# =========================================================
# OUTPUT DIRECTORY
# =========================================================
out_dir = "6-spectral-angle-mapper"
os.makedirs(out_dir, exist_ok=True)


# =========================================================
# 1. LOAD EMIT DATA
# =========================================================
file_path = "EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc"

ds = xr.open_dataset(file_path)
data = ds["reflectance"].values.astype(np.float32)

rows, cols, bands = data.shape


# =========================================================
# LOAD WAVELENGTHS
# =========================================================
wvl_ds = xr.open_dataset(file_path, group="sensor_band_parameters")
wavelengths = wvl_ds["wavelengths"].values

if wavelengths.max() < 100:
    wavelengths *= 1000


# =========================================================
# 2. MANUAL MASK
# =========================================================
mask = np.ones((rows, cols), dtype=bool)

mask[291:384, 1047:1165] = False
mask[500:602, 74:170] = False
mask[287:330, 151:218] = False
mask[254:314, 211:308] = False
mask[714:819, 88:236] = False
mask[1152:1192, 944:1021] = False
mask[837:899, 1:68] = False
mask[347:461, 42:184] = False
mask[414:470, 175:300] = False
mask[467:530, 307:406] = False
mask[559:623, 419:537] = False
mask[506:547, 545:618] = False
mask[801:850, 350:410] = False
mask[1035:1078, 560:611] = False
mask[90:124, 1029:1097] = False
mask[0:61, 1065:1137] = False
mask[599:651, 176:237] = False
mask[703:731, 331:382] = False
mask[656:747, 252:323] = False
mask[1159:1194, 462:510] = False
mask[1155:1197, 773:902] = False
mask[779:865, 630:666] = False


data = data.copy()
data[~mask] = np.nan


# =========================================================
# 3. SHADOW MASK
# =========================================================
brightness = np.nanmean(data, axis=2)
shadow_threshold = np.nanpercentile(brightness, 10)
shadow_mask = brightness < shadow_threshold

data[shadow_mask] = np.nan


# =========================================================
# 4. REMOVE BAD BANDS
# =========================================================
valid_band = (
    ((wavelengths > 450) & (wavelengths < 1340)) |
    ((wavelengths > 1450) & (wavelengths < 1800)) |
    ((wavelengths > 1950) & (wavelengths < 2450))
)

data = data[:, :, valid_band]
wavelengths = wavelengths[valid_band]


# =========================================================
# 5. RESHAPE
# =========================================================
X_full = data.reshape(-1, data.shape[-1])

valid_pixels = np.all(np.isfinite(X_full), axis=1)
X = X_full[valid_pixels]


# =========================================================
# 6. SAVGOL SMOOTHING
# =========================================================
def smooth_spectrum(spec):
    if np.sum(np.isfinite(spec)) < 11:
        return spec
    try:
        return savgol_filter(spec, 11, 2)
    except:
        return spec


print("Smoothing EMIT spectra...")
for i in range(X.shape[0]):
    X[i] = smooth_spectrum(X[i])


# =========================================================
# 7. CONTINUUM REMOVAL
# =========================================================
def continuum_remove_spectrum(wvl, spec):
    valid = np.isfinite(spec)
    w = wvl[valid]
    s = spec[valid]

    if len(s) < 5:
        return np.full_like(spec, np.nan)

    try:
        points = np.column_stack([w, s])
        hull = ConvexHull(points)
        hull_vertices = np.sort(hull.vertices)

        upper = []
        for v in hull_vertices:
            while (
                len(upper) >= 2 and
                ((s[upper[-1]] - s[upper[-2]]) /
                 (w[upper[-1]] - w[upper[-2]] + 1e-10))
                <
                ((s[v] - s[upper[-1]]) /
                 (w[v] - w[upper[-1]] + 1e-10))
            ):
                upper.pop()
            upper.append(v)

        upper = np.array(upper)

        continuum_interp = interp1d(
            w[upper], s[upper],
            bounds_error=False,
            fill_value="extrapolate"
        )

        continuum = continuum_interp(w)
        continuum[continuum <= 0] = np.nan

        cr = s / continuum

        out = np.full_like(spec, np.nan)
        out[valid] = cr

        return out

    except:
        return np.full_like(spec, np.nan)


print("Continuum removing EMIT spectra...")
X_cr = np.zeros_like(X)

for i in range(X.shape[0]):
    X_cr[i] = continuum_remove_spectrum(wavelengths, X[i])

X = X_cr


# =========================================================
# 8. NORMALIZATION
# =========================================================
norms = np.linalg.norm(X, axis=1, keepdims=True)
X = X / (norms + 1e-10)


# =========================================================
# 9–13. LOAD USGS LIBRARY
# =========================================================
usgs_folder = "usgs"

wavelength_file = glob.glob(os.path.join(usgs_folder, "*Wavelengths*.txt"))[0]

def load_usgs_spectrum(filepath):
    vals = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                vals.append(float(line.strip()))
            except:
                continue
    vals = np.array(vals, dtype=np.float32)
    vals[vals < -1e30] = np.nan
    return vals


usgs_wavelengths = load_usgs_spectrum(wavelength_file)

if usgs_wavelengths.max() < 100:
    usgs_wavelengths *= 1000


all_files = glob.glob(os.path.join(usgs_folder, "*.txt"))
mineral_files = [f for f in all_files if "Wavelengths" not in f]


library = {}

for filepath in mineral_files:

    name = os.path.basename(filepath).split("_")[2] \
        if len(os.path.basename(filepath).split("_")) >= 3 \
        else os.path.basename(filepath)

    spec = load_usgs_spectrum(filepath)

    n = min(len(spec), len(usgs_wavelengths))
    spec = spec[:n]
    lib_wvl = usgs_wavelengths[:n]

    valid = np.isfinite(spec) & np.isfinite(lib_wvl)

    interp = interp1d(
        lib_wvl[valid],
        spec[valid],
        bounds_error=False,
        fill_value=np.nan
    )

    resampled = interp(wavelengths)
    resampled = smooth_spectrum(resampled)
    resampled = continuum_remove_spectrum(wavelengths, resampled)

    resampled = resampled / (np.linalg.norm(resampled) + 1e-10)

    library[name] = resampled


print(f"Loaded {len(library)} minerals")


# =========================================================
# 14. SAM FUNCTION
# =========================================================
def spectral_angle(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]

    if len(a) == 0:
        return np.nan

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan

    cosang = np.clip(np.dot(a, b) / denom, -1, 1)
    return np.arccos(cosang)


# =========================================================
# 15. SAM CLASSIFICATION
# =========================================================
mineral_names = list(library.keys())

angles = np.zeros((X.shape[0], len(mineral_names)), dtype=np.float32)

for i, mineral in enumerate(mineral_names):
    ref = library[mineral]
    print("SAM:", mineral)

    for j in range(X.shape[0]):
        angles[j, i] = spectral_angle(X[j], ref)


best_match = np.argmin(angles, axis=1)
best_angle = np.min(angles, axis=1)

threshold = 0.10
best_match[best_angle > threshold] = -1


# =========================================================
# 16. REBUILD IMAGE
# =========================================================
full_class = np.full(X_full.shape[0], -1)
full_class[valid_pixels] = best_match
mineral_map = full_class.reshape(rows, cols)

full_angle = np.full(X_full.shape[0], np.nan)
full_angle[valid_pixels] = best_angle
angle_map = full_angle.reshape(rows, cols)


# =========================================================
# 17. SAVE OUTPUTS
# =========================================================
np.save(os.path.join(out_dir, "mineral_map.npy"), mineral_map)
np.save(os.path.join(out_dir, "angle_map.npy"), angle_map)
np.save(os.path.join(out_dir, "best_match.npy"), best_match)
np.save(os.path.join(out_dir, "best_angle.npy"), best_angle)

with open(os.path.join(out_dir, "mineral_labels.txt"), "w") as f:
    for i, name in enumerate(mineral_names):
        f.write(f"{i}: {name}\n")


# =========================================================
# 18. HIGH-RES PLOTS
# =========================================================
plt.figure(figsize=(16, 12), dpi=400)

im = plt.imshow(mineral_map, cmap="tab20")
cbar = plt.colorbar(im)
cbar.set_ticks(range(len(mineral_names)))
cbar.set_ticklabels(mineral_names)

plt.title("EMIT SAM Mineral Classification")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "sam_mineral_map.png"),
            dpi=400, bbox_inches="tight")
plt.show()


plt.figure(figsize=(16, 12), dpi=400)

plt.imshow(angle_map, cmap="inferno_r")
plt.colorbar(label="Spectral Angle")
plt.title("SAM Confidence")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "sam_confidence_map.png"),
            dpi=400, bbox_inches="tight")
plt.show()


# =========================================================
# 19. LIBRARY PLOT
# =========================================================
plt.figure(figsize=(14, 8), dpi=400)

for mineral in mineral_names:
    plt.plot(wavelengths, library[mineral], label=mineral)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Continuum Removed Reflectance")
plt.title("USGS Mineral Library (EMIT-aligned)")
plt.legend()
plt.grid()2

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "usgs_library.png"),
            dpi=400, bbox_inches="tight")
plt.show()