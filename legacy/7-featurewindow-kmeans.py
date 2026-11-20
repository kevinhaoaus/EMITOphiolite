import os
import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


# =========================================================
# CONFIG
# =========================================================
FILE_PATH = "EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc"

OUT_DIR = "memory_efficient_feature_window_kmeans"

os.makedirs(OUT_DIR, exist_ok=True)

CHUNK_ROWS = 128
N_CLUSTERS = 8
PCA_COMPONENTS = 5

# ---------------------------------------------------------
# PROCESS WINDOWS
# ---------------------------------------------------------
WINDOWS = {

    "hematite": (550, 950),

    "chlorite": (2180, 2265),

    "carbonate": (2280, 2380),

    "alteration": (2100, 2380),
}


# =========================================================
# LOAD DATA LAZILY
# =========================================================
print("Opening dataset...")

ds = xr.open_dataset(
    FILE_PATH,
    chunks={"downtrack": CHUNK_ROWS}
)

reflectance = ds["reflectance"]

rows, cols, bands = reflectance.shape

print(f"Scene shape: {rows} x {cols} x {bands}")


# =========================================================
# LOAD WAVELENGTHS
# =========================================================
wvl_ds = xr.open_dataset(
    FILE_PATH,
    group="sensor_band_parameters"
)

wavelengths = wvl_ds["wavelengths"].values.astype(np.float32)

if wavelengths.max() < 100:
    wavelengths *= 1000


# =========================================================
# VALID BAND FILTER
# =========================================================
valid_band = (
    ((wavelengths > 450) & (wavelengths < 1340)) |
    ((wavelengths > 1450) & (wavelengths < 1800)) |
    ((wavelengths > 1950) & (wavelengths < 2450))
)

wavelengths = wavelengths[valid_band]

reflectance = reflectance[:, :, valid_band]

print(f"Using {len(wavelengths)} valid bands")


# =========================================================
# INDICES
# =========================================================
red_idx = np.argmin(np.abs(wavelengths - 660))
nir_idx = np.argmin(np.abs(wavelengths - 850))

swir_mask = wavelengths > 2000


# =========================================================
# UTILITIES
# =========================================================
def stretch(img):

    valid = np.isfinite(img)

    if np.sum(valid) == 0:
        return np.zeros_like(img)

    p2, p98 = np.nanpercentile(img, (2, 98))

    out = (img - p2) / (p98 - p2 + 1e-10)

    return np.clip(out, 0, 1)


def smooth_spectrum(spec):

    if np.sum(np.isfinite(spec)) < 11:
        return spec

    try:
        return savgol_filter(spec, 9, 2)

    except:
        return spec


def continuum_remove_spectrum(wvl, spec):

    valid = np.isfinite(spec)

    if np.sum(valid) < 5:
        return np.full_like(spec, np.nan)

    w = wvl[valid]
    s = spec[valid]

    try:

        points = np.column_stack([w, s])

        hull = ConvexHull(points)

        hull_vertices = np.sort(hull.vertices)

        upper = []

        for v in hull_vertices:

            while (
                len(upper) >= 2 and
                (
                    (s[upper[-1]] - s[upper[-2]]) /
                    (w[upper[-1]] - w[upper[-2]] + 1e-10)
                )
                <
                (
                    (s[v] - s[upper[-1]]) /
                    (w[v] - w[upper[-1]] + 1e-10)
                )
            ):
                upper.pop()

            upper.append(v)

        upper = np.array(upper)

        continuum_interp = interp1d(
            w[upper],
            s[upper],
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


def preprocess_chunk(chunk):

    # -----------------------------------------------------
    # SHADOW MASK
    # -----------------------------------------------------
    brightness = np.nanmean(
        chunk[:, :, swir_mask],
        axis=2
    )

    shadow_threshold = np.nanpercentile(
        brightness,
        10
    )

    shadow_mask = brightness < shadow_threshold

    chunk[shadow_mask] = np.nan

    # -----------------------------------------------------
    # NDVI MASK
    # -----------------------------------------------------
    red = chunk[:, :, red_idx]
    nir = chunk[:, :, nir_idx]

    ndvi = (nir - red) / (nir + red + 1e-10)

    veg_mask = ndvi > 0.2

    chunk[veg_mask] = np.nan

    # -----------------------------------------------------
    # RESHAPE
    # -----------------------------------------------------
    X = chunk.reshape(-1, chunk.shape[-1])

    finite = np.isfinite(X)

    valid_fraction = np.sum(
        finite,
        axis=1
    ) / X.shape[1]

    enough_valid = valid_fraction > 0.95

    all_finite = np.all(
        finite,
        axis=1
    )

    std = np.nanstd(
        X,
        axis=1
    )

    nonflat = std > 1e-6

    valid_pixels = (
        enough_valid &
        all_finite &
        nonflat
    )

    X = X[valid_pixels]

    if len(X) == 0:
        return None

    # -----------------------------------------------------
    # SMOOTH + CONTINUUM REMOVE
    # -----------------------------------------------------
    out = np.zeros_like(X)

    for i in range(len(X)):

        spec = smooth_spectrum(X[i])

        spec = continuum_remove_spectrum(
            wavelengths,
            spec
        )

        out[i] = spec

    # -----------------------------------------------------
    # NORMALIZE
    # -----------------------------------------------------
    norms = np.linalg.norm(
        out,
        axis=1,
        keepdims=True
    )

    out = out / (norms + 1e-10)

    return out


def get_window(X, wmin, wmax):

    mask = (
        (wavelengths >= wmin) &
        (wavelengths <= wmax)
    )

    return X[:, mask], wavelengths[mask]


# =========================================================
# MAIN LOOP
# =========================================================
for name, (wmin, wmax) in WINDOWS.items():

    print("\n===================================")
    print(f"PROCESSING: {name}")
    print("===================================")

    # -----------------------------------------------------
    # PASS 1:
    # FIT SCALER + INCREMENTAL PCA
    # -----------------------------------------------------
    scaler = StandardScaler()

    ipca = IncrementalPCA(
        n_components=PCA_COMPONENTS
    )

    print("Fitting scaler + PCA...")

    for r0 in range(0, rows, CHUNK_ROWS):

        r1 = min(r0 + CHUNK_ROWS, rows)

        print(f"Chunk {r0}:{r1}")

        chunk = (
            reflectance[r0:r1]
            .compute()
            .values
            .astype(np.float32)
        )

        X = preprocess_chunk(chunk)

        del chunk

        if X is None:
            continue

        X_sub, wvl_sub = get_window(
            X,
            wmin,
            wmax
        )

        finite = np.all(
            np.isfinite(X_sub),
            axis=1
        )

        X_sub = X_sub[finite]

        if len(X_sub) < PCA_COMPONENTS:
            continue

        scaler.partial_fit(X_sub)

        X_scaled = scaler.transform(X_sub)

        ipca.partial_fit(X_scaled)

        del X
        del X_sub
        del X_scaled

        gc.collect()

    # -----------------------------------------------------
    # PASS 2:
    # TRAIN KMEANS
    # -----------------------------------------------------
    print("Training MiniBatchKMeans...")

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=10000,
        random_state=42
    )

    for r0 in range(0, rows, CHUNK_ROWS):

        r1 = min(r0 + CHUNK_ROWS, rows)

        print(f"KMeans chunk {r0}:{r1}")

        chunk = (
            reflectance[r0:r1]
            .compute()
            .values
            .astype(np.float32)
        )

        X = preprocess_chunk(chunk)

        del chunk

        if X is None:
            continue

        X_sub, _ = get_window(
            X,
            wmin,
            wmax
        )

        finite = np.all(
            np.isfinite(X_sub),
            axis=1
        )

        X_sub = X_sub[finite]

        if len(X_sub) == 0:
            continue

        X_scaled = scaler.transform(X_sub)

        X_pca = ipca.transform(X_scaled)

        kmeans.partial_fit(X_pca)

        del X
        del X_sub
        del X_scaled
        del X_pca

        gc.collect()

    # -----------------------------------------------------
    # OUTPUT ARRAYS
    # -----------------------------------------------------
    cluster_map = np.full(
        (rows, cols),
        -1,
        dtype=np.int16
    )

    pca_rgb = np.full(
        (rows, cols, 3),
        np.nan,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # MEAN SPECTRA
    # -----------------------------------------------------
    cluster_sums = np.zeros(
        (N_CLUSTERS, len(wvl_sub)),
        dtype=np.float64
    )

    cluster_counts = np.zeros(
        N_CLUSTERS,
        dtype=np.int64
    )

    # -----------------------------------------------------
    # PASS 3:
    # PREDICT + SAVE MAPS
    # -----------------------------------------------------
    print("Predicting clusters...")

    for r0 in range(0, rows, CHUNK_ROWS):

        r1 = min(r0 + CHUNK_ROWS, rows)

        print(f"Predict chunk {r0}:{r1}")

        chunk = (
            reflectance[r0:r1]
            .compute()
            .values
            .astype(np.float32)
        )

        # -------------------------------------------------
        # MASKING AGAIN
        # -------------------------------------------------
        brightness = np.nanmean(
            chunk[:, :, swir_mask],
            axis=2
        )

        shadow_threshold = np.nanpercentile(
            brightness,
            10
        )

        shadow_mask = brightness < shadow_threshold

        chunk[shadow_mask] = np.nan

        red = chunk[:, :, red_idx]

        nir = chunk[:, :, nir_idx]

        ndvi = (nir - red) / (nir + red + 1e-10)

        veg_mask = ndvi > 0.2

        chunk[veg_mask] = np.nan

        # -------------------------------------------------
        # RESHAPE
        # -------------------------------------------------
        X_raw = chunk.reshape(
            -1,
            chunk.shape[-1]
        )

        finite = np.isfinite(X_raw)

        valid_fraction = np.sum(
            finite,
            axis=1
        ) / X_raw.shape[1]

        enough_valid = valid_fraction > 0.95

        all_finite = np.all(
            finite,
            axis=1
        )

        std = np.nanstd(
            X_raw,
            axis=1
        )

        nonflat = std > 1e-6

        valid_pixels = (
            enough_valid &
            all_finite &
            nonflat
        )

        valid_idx = np.where(valid_pixels)[0]

        if len(valid_idx) == 0:
            continue

        X = X_raw[valid_pixels]

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------
        out = np.zeros_like(X)

        for i in range(len(X)):

            spec = smooth_spectrum(X[i])

            spec = continuum_remove_spectrum(
                wavelengths,
                spec
            )

            out[i] = spec

        norms = np.linalg.norm(
            out,
            axis=1,
            keepdims=True
        )

        out = out / (norms + 1e-10)

        # -------------------------------------------------
        # WINDOW
        # -------------------------------------------------
        X_sub, _ = get_window(
            out,
            wmin,
            wmax
        )

        finite = np.all(
            np.isfinite(X_sub),
            axis=1
        )

        valid_idx = valid_idx[finite]

        X_sub = X_sub[finite]

        if len(X_sub) == 0:
            continue

        # -------------------------------------------------
        # PCA
        # -------------------------------------------------
        X_scaled = scaler.transform(X_sub)

        X_pca = ipca.transform(X_scaled)

        labels = kmeans.predict(X_pca)

        # -------------------------------------------------
        # STORE CLUSTERS
        # -------------------------------------------------
        rr = valid_idx // cols

        cc = valid_idx % cols

        rr = rr.reshape(-1)

        cc = cc.reshape(-1)

        global_rr = rr + r0

        cluster_map[global_rr, cc] = labels

        # -------------------------------------------------
        # PCA RGB
        # -------------------------------------------------
        rgb = np.dstack([

            stretch(X_pca[:, 0]),

            stretch(X_pca[:, 1]),

            stretch(X_pca[:, 2]),
        ])[0]

        pca_rgb[global_rr, cc] = rgb

        # -------------------------------------------------
        # MEAN SPECTRA
        # -------------------------------------------------
        for c in range(N_CLUSTERS):

            mask = labels == c

            if np.sum(mask) == 0:
                continue

            cluster_sums[c] += np.sum(
                X_sub[mask],
                axis=0
            )

            cluster_counts[c] += np.sum(mask)

        del chunk
        del X_raw
        del X
        del out
        del X_sub
        del X_scaled
        del X_pca

        gc.collect()

    # =====================================================
    # SAVE CLUSTER MAP
    # =====================================================
    np.save(
        os.path.join(
            OUT_DIR,
            f"{name}_clusters.npy"
        ),
        cluster_map
    )

    plt.figure(
        figsize=(14, 12),
        dpi=300
    )

    cmap = plt.cm.get_cmap(
        "tab20",
        N_CLUSTERS
    )

    im = plt.imshow(
        cluster_map,
        cmap=cmap,
        vmin=0,
        vmax=N_CLUSTERS - 1
    )

    cbar = plt.colorbar(im)

    cbar.set_label("Cluster")

    plt.title(
        f"{name.upper()} Feature Window KMeans"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"{name}_clusters.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # SAVE PCA RGB
    # =====================================================
    plt.figure(
        figsize=(14, 12),
        dpi=300
    )

    plt.imshow(pca_rgb)

    plt.title(
        f"{name.upper()} PCA RGB\n"
        "(R=PC1 G=PC2 B=PC3)"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"{name}_pca_rgb.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # SAVE MEAN SPECTRA
    # =====================================================
    plt.figure(
        figsize=(12, 6),
        dpi=300
    )

    for c in range(N_CLUSTERS):

        if cluster_counts[c] == 0:
            continue

        mean_spec = (
            cluster_sums[c] /
            cluster_counts[c]
        )

        plt.plot(
            wvl_sub,
            mean_spec,
            label=f"Cluster {c}"
        )

    plt.xlabel("Wavelength (nm)")

    plt.ylabel(
        "Continuum Removed Reflectance"
    )

    plt.title(
        f"{name.upper()} Cluster Mean Spectra"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"{name}_cluster_spectra.png"
        ),
        dpi=300
    )

    plt.close()

    gc.collect()


# =========================================================
# GEOLOGICAL RGB COMPOSITE
# =========================================================
print("\nCreating geological composite...")

carbonate = np.load(
    os.path.join(
        OUT_DIR,
        "carbonate_clusters.npy"
    )
)

chlorite = np.load(
    os.path.join(
        OUT_DIR,
        "chlorite_clusters.npy"
    )
)

hematite = np.load(
    os.path.join(
        OUT_DIR,
        "hematite_clusters.npy"
    )
)

geo_rgb = np.dstack([

    stretch(carbonate),

    stretch(chlorite),

    stretch(hematite),
])

plt.figure(
    figsize=(14, 12),
    dpi=400
)

plt.imshow(geo_rgb)

plt.title(
    "Geological Cluster Composite\n"
    "R=Carbonate\n"
    "G=Chlorite\n"
    "B=Hematite"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "geological_cluster_rgb.png"
    ),
    dpi=400,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# DONE
# =========================================================
print("\n===================================")
print("DONE")
print("===================================")
print(f"Outputs saved to: {OUT_DIR}")