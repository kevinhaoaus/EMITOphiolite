# python -m pip install numpy xarray matplotlib sklearn

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================================================
# 1. LOAD DATA
# =========================================================
file_path = "EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc"

ds = xr.open_dataset(file_path)
data = ds["reflectance"].values
rows, cols, bands = data.shape

wvl = xr.open_dataset(file_path, group="sensor_band_parameters")
wavelengths = wvl["wavelengths"].values


# =========================================================
# 2. MANUAL MASK (PASTE FROM YOUR TOOL)
# =========================================================
mask = np.ones((rows, cols), dtype=bool)

# -------- COPY THIS --------
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
# ----------------------------

# Apply mask
data = data.copy()
data[~mask] = np.nan


# =========================================================
# 3. SHADOW MASKING (illumination removal)
# =========================================================
brightness = np.nanmean(data, axis=2)
shadow_mask = brightness < np.nanpercentile(brightness, 10)
data[shadow_mask] = np.nan


# =========================================================
# 4. SPECTRAL MASK (remove absorption bands)
# =========================================================
valid_band = (
    (wavelengths < 1340) |
    ((wavelengths > 1450) & (wavelengths < 1800)) |
    (wavelengths > 1950)
)

data = data[:, :, valid_band]
wavelengths = wavelengths[valid_band]


# =========================================================
# 5. RESHAPE
# =========================================================
X_full = data.reshape(-1, data.shape[-1])
valid = np.all(np.isfinite(X_full), axis=1)
X = X_full[valid]


# =========================================================
# 6. CONTINUUM REMOVAL (approx)
# =========================================================
def continuum_removal(X):
    # normalize spectra by convex hull approximation
    X = X / (np.max(X, axis=1, keepdims=True) + 1e-10)
    return X

X = continuum_removal(X)


# =========================================================
# 7. STANDARDIZE
# =========================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)


# =========================================================
# 8. MNF TRANSFORM (Noise-adjusted PCA approximation)
# =========================================================
# Step 1: PCA whitening-like transform
cov = np.cov(X, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)

# sort descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# MNF transform
X_mnf = X @ eigvecs


# keep top components (signal-dominant)
n_components = 10
X_mnf = X_mnf[:, :n_components]


# =========================================================
# 9. COSINE-STYLE NORMALIZATION
# =========================================================
X_mnf = X_mnf / (np.linalg.norm(X_mnf, axis=1, keepdims=True) + 1e-10)


# =========================================================
# 10. KMEANS CLUSTERING (cosine-like space)
# =========================================================
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_mnf)


# =========================================================
# 11. RECONSTRUCT MAP
# =========================================================
full = np.full(X_full.shape[0], -1)
full[valid] = labels

cluster_map = full.reshape(rows, cols)


# =========================================================
# 12. PLOT CLUSTER MAP
# =========================================================
plt.figure(figsize=(10, 8))
plt.imshow(cluster_map, cmap="tab20")
plt.colorbar(label="Cluster ID")
plt.title("EMIT Clustering (MNF + Continuum + Cosine Space)")
plt.axis("off")
plt.show()


# =========================================================
# 13. MEAN SPECTRA
# =========================================================
plt.figure(figsize=(10, 6))

for i in range(k):
    mean_spec = X[labels == i].mean(axis=0)
    plt.plot(mean_spec, label=f"Cluster {i}")

plt.title("Cluster Mean Spectra (Processed)")
plt.xlabel("MNF/Normalized Band Index")
plt.ylabel("Relative Value")
plt.legend()
plt.grid()
plt.show()