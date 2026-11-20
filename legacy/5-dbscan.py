import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN

from joblib import Parallel, delayed
import multiprocessing

# =========================================================
# SETTINGS
# =========================================================
FILE_PATH = "EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc"

row_chunk = 100
k_final = 5
N_JOBS = multiprocessing.cpu_count()

# =========================================================
# LOAD (LAZY)
# =========================================================
ds = xr.open_dataset(FILE_PATH, chunks={"downtrack": row_chunk})

cube = ds["reflectance"]   # lazy dask-backed
rows, cols, bands = cube.shape

wvl = xr.open_dataset(FILE_PATH, group="sensor_band_parameters")
wavelengths = wvl["wavelengths"].values

# =========================================================
# MASK (same as yours)
# =========================================================
mask = np.ones((rows, cols), dtype=bool)

mask[291:384, 1047:1165] = False
mask[500:602, 74:170] = False
# ... (keep all your mask lines)

# =========================================================
# BAND FILTER
# =========================================================
valid_band = (
    (wavelengths < 1340) |
    ((wavelengths > 1450) & (wavelengths < 1800)) |
    (wavelengths > 1950)
)

wavelengths = wavelengths[valid_band]

# =========================================================
# HELPERS
# =========================================================
def process_chunk(r):
    r_end = min(r + row_chunk, rows)

    chunk = cube[r:r_end, :, :].values.astype(np.float32)

    # apply mask
    chunk[~mask[r:r_end]] = np.nan

    # brightness mask
    brightness = np.nanmean(chunk, axis=2)
    shadow = brightness < np.nanpercentile(brightness, 10)
    chunk[shadow] = np.nan

    # band filter
    chunk = chunk[:, :, valid_band]

    flat = chunk.reshape(-1, chunk.shape[-1])

    valid = np.all(np.isfinite(flat), axis=1)
    flat = flat[valid]

    if len(flat) == 0:
        return None

    # continuum removal
    flat = flat / (np.max(flat, axis=1, keepdims=True) + 1e-10)

    return flat


# =========================================================
# 1. SCALER (STREAM)
# =========================================================
scaler = StandardScaler()

for r in range(0, rows, row_chunk):
    data = process_chunk(r)
    if data is not None:
        scaler.partial_fit(data)

# =========================================================
# 2. KMEANS (SCALABLE CORE MODEL)
# =========================================================
kmeans = MiniBatchKMeans(
    n_clusters=k_final,
    batch_size=8192,
    random_state=0
)

for r in range(0, rows, row_chunk):
    data = process_chunk(r)
    if data is not None:
        data = scaler.transform(data)
        kmeans.partial_fit(data)

# =========================================================
# 3. OPTIONAL: DBSCAN ON SAMPLE ONLY
# =========================================================
def sample_chunk(r):
    data = process_chunk(r)
    if data is None:
        return None
    data = scaler.transform(data)
    return data[:2000]

samples = Parallel(n_jobs=N_JOBS)(
    delayed(sample_chunk)(r) for r in range(0, rows, row_chunk)
)

X_sample = np.vstack([s for s in samples if s is not None])

dbscan = DBSCAN(eps=0.1, min_samples=10)
labels_dbscan_sample = dbscan.fit_predict(X_sample)

# =========================================================
# 4. CLUSTER MAP (PARALLEL)
# =========================================================
def predict_chunk(r):
    r_end = min(r + row_chunk, rows)
    h = r_end - r

    chunk = cube[r:r_end, :, :].values.astype(np.float32)
    chunk[~mask[r:r_end]] = np.nan

    brightness = np.nanmean(chunk, axis=2)
    shadow = brightness < np.nanpercentile(brightness, 10)
    chunk[shadow] = np.nan

    chunk = chunk[:, :, valid_band]

    flat = chunk.reshape(-1, chunk.shape[-1])
    valid = np.all(np.isfinite(flat), axis=1)

    labels = np.full(flat.shape[0], np.nan)

    if np.any(valid):
        data = flat[valid]
        data = data / (np.max(data, axis=1, keepdims=True) + 1e-10)
        data = scaler.transform(data)

        labels_valid = kmeans.predict(data)
        labels[valid] = labels_valid

    return r, labels.reshape(h, cols)

results = Parallel(n_jobs=N_JOBS)(
    delayed(predict_chunk)(r) for r in range(0, rows, row_chunk)
)

cluster_map = np.full((rows, cols), np.nan)
for r, block in results:
    cluster_map[r:r + block.shape[0]] = block

# =========================================================
# 5. AVERAGE SPECTRA (STREAMING + PARALLEL)
# =========================================================
def stats_chunk(r):
    data = process_chunk(r)
    if data is None:
        return None

    data_scaled = scaler.transform(data)
    labels = kmeans.predict(data_scaled)

    sums = np.zeros((k_final, data.shape[1]))
    counts = np.zeros(k_final)

    for k in range(k_final):
        mask_k = labels == k
        if np.any(mask_k):
            sums[k] = data[mask_k].sum(axis=0)
            counts[k] = mask_k.sum()

    return sums, counts

results = Parallel(n_jobs=N_JOBS)(
    delayed(stats_chunk)(r) for r in range(0, rows, row_chunk)
)

cluster_sums = np.zeros((k_final, len(wavelengths)))
cluster_counts = np.zeros(k_final)

for res in results:
    if res is not None:
        s, c = res
        cluster_sums += s
        cluster_counts += c

avg_spectra = cluster_sums / cluster_counts[:, None]

# =========================================================
# 6. PLOT SPECTRA
# =========================================================
for k in range(k_final):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, avg_spectra[k])
    plt.title(f"Cluster {k} Average Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.grid(True)
    plt.savefig(f"emit_cluster_{k}.png")
    plt.close()

# =========================================================
# 7. MAP
# =========================================================
plt.figure(figsize=(12, 10))
plt.imshow(cluster_map, cmap="tab20")
plt.colorbar()
plt.title("EMIT Clustering (Memory Efficient)")
plt.axis("off")
plt.show()