"""Output discovery, server-side map rendering, pixel queries, spectra and L2B."""
from __future__ import annotations

import io
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
DATA = ROOT / "data"
REFL = DATA / "EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc"
L2B = DATA / "EMIT_L2B_MIN_001_20260326T084318_2608506_030.nc"
BOXES = DATA / "manual_mask_boxes.json"
USGS = ROOT / "usgs"
_lock = threading.Lock()


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


def safe_path(r: str, must_exist=True) -> Path:
    p = (ROOT / r).resolve()
    o = OUT.resolve()
    if p != o and o not in p.parents:
        raise ValueError("path must be inside outputs/")
    if must_exist and not p.exists():
        raise FileNotFoundError(r)
    return p


# ----------------------------------------------------------------- outputs
def output_dirs() -> list[dict]:
    if not OUT.is_dir():
        return []
    res = []
    for d in [OUT] + sorted(x for x in OUT.rglob("*") if x.is_dir()):
        if d.name == "maps" or "maps" in d.relative_to(OUT).parts:
            continue
        pngs = list(d.glob("*.png"))
        npys = list(d.glob("*.npy")) + (list((d / "maps").rglob("*.npy")) if (d / "maps").is_dir() else [])
        if pngs or npys:
            res.append({"dir": rel(d), "png": len(pngs), "npy": len(npys)})
    return res


def pretty(stem: str) -> tuple[str, str]:
    s = stem
    for pre, grp, label in (("mf_best_labels", "Matched filter", "Best target"), ("pca_rgb", "PCA", "PCA RGB"),
                            ("mnf_rgb", "MNF", "MNF RGB"), ("sam_labels", "SAM", "Labels"),
                            ("sam_angles", "SAM", "Angles (rad)"), ("emit_l2b_mineral_map", "L2B", "Mineral map")):
        if s == pre:
            return grp, label
    if s.startswith("mf_"):
        return "Matched filter", s[3:]
    if s.startswith("window_"):
        return "Feature windows", s[7:].replace("_pca_rgb", "").replace("_pca", "") + " window"
    if s.startswith("pca"):
        return "PCA", s
    if s.startswith("mnf"):
        return "MNF", s
    if s.startswith("sam"):
        return "SAM", s
    return "Other", s


_mm: dict = {}


def load_npy(p: Path):
    key = (str(p), p.stat().st_mtime)
    a = _mm.get(key)
    if a is None:
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        _mm[key] = a
    return a


def list_layers(d: str) -> dict:
    base = safe_path(d)
    layers, texts = [], []
    for p in sorted(base.glob("*.png")):
        grp, label = pretty(p.stem)
        try:
            from PIL import Image
            with Image.open(p) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            w = h = None
        layers.append({"id": rel(p), "type": "figure", "group": grp, "label": label, "file": rel(p),
                       "w": w, "h": h, "size": p.stat().st_size, "mtime": p.stat().st_mtime})
    npys = list(base.glob("*.npy")) + (sorted((base / "maps").rglob("*.npy")) if (base / "maps").is_dir() else [])
    for p in npys:
        try:
            a = load_npy(p)
        except Exception as exc:  # noqa: BLE001
            layers.append({"id": rel(p), "type": "error", "group": "Maps", "label": p.stem, "error": str(exc)})
            continue
        if a.ndim not in (2, 3):
            continue
        kind = "multi" if a.ndim == 3 else ("labels" if np.issubdtype(a.dtype, np.integer) else "scalar")
        grp, label = pretty(p.stem)
        layers.append({"id": rel(p), "type": kind, "group": grp, "label": label, "file": rel(p),
                       "shape": list(a.shape), "dtype": str(a.dtype), "mtime": p.stat().st_mtime})
    for p in sorted(base.glob("*.txt")) + sorted(base.glob("*.json")) + (
            sorted((base / "maps").glob("*.json")) if (base / "maps").is_dir() else []):
        if p.stat().st_size < 200_000:
            texts.append({"name": rel(p), "content": p.read_text(encoding="utf-8", errors="replace")})
    return {"dir": d, "layers": layers, "texts": texts}


def list_images(d: str) -> list[dict]:
    try:
        base = safe_path(d)
    except Exception:  # noqa: BLE001
        return []
    return [{"name": p.name, "file": rel(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime}
            for p in sorted(base.glob("*.png"))]


# ----------------------------------------------------------------- rendering
PALETTE = ["#2dd4bf", "#f59e0b", "#60a5fa", "#f472b6", "#a3e635", "#fb7185", "#c084fc", "#fbbf24",
           "#34d399", "#f87171", "#38bdf8", "#e879f9", "#facc15", "#4ade80", "#818cf8", "#fb923c"]


def _hex(c):
    return [int(c[i:i + 2], 16) for i in (1, 3, 5)]


def _stretch(a, lo, hi):
    f = a[np.isfinite(a)]
    if f.size == 0:
        return np.zeros(a.shape, np.float32), (0.0, 0.0)
    if f.size > 400_000:
        f = f[:: f.size // 400_000]
    l, h = np.percentile(f, [lo, hi])
    if h <= l:
        h = l + 1e-12
    return np.clip((a - l) / (h - l), 0, 1).astype(np.float32), (float(l), float(h))


_render_cache: dict = {}


def render(path: str, comps=(0, 1, 2), lo=2.0, hi=98.0, cmap="viridis", component=None):
    p = safe_path(path)
    key = (path, p.stat().st_mtime, tuple(comps), lo, hi, cmap, component)
    if key in _render_cache:
        return _render_cache[key]
    from PIL import Image
    a = load_npy(p)
    meta = {}
    if a.ndim == 3 and component is None:
        k = a.shape[2]
        chans, rng = [], []
        alpha = np.ones(a.shape[:2], bool)
        for c in comps:
            band = np.asarray(a[:, :, min(int(c), k - 1)], dtype=np.float32)
            alpha &= np.isfinite(band)
            s, r = _stretch(band, lo, hi)
            chans.append(np.nan_to_num(s))
            rng.append(r)
        rgb = (np.stack(chans, -1) * 255).astype(np.uint8)
        meta["ranges"] = rng
    else:
        band = a if a.ndim == 2 else a[:, :, int(component)]
        band = np.asarray(band)
        if np.issubdtype(band.dtype, np.integer):
            alpha = band >= 0
            lut = np.array([_hex(c) for c in PALETTE], np.uint8)
            rgb = lut[np.mod(np.where(alpha, band, 0), len(PALETTE))]
            meta["labels"] = sorted(int(v) for v in np.unique(band[alpha])[:64])
            meta["palette"] = PALETTE
        else:
            band = band.astype(np.float32)
            alpha = np.isfinite(band)
            s, r = _stretch(band, lo, hi)
            from matplotlib import colormaps
            lut = (colormaps[cmap](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            rgb = lut[(np.nan_to_num(s) * 255).astype(np.int32)]
            meta["ranges"] = [r]
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=3)
    out = (buf.getvalue(), meta)
    if len(_render_cache) > 48:
        _render_cache.clear()
    _render_cache[key] = out
    return out


def _clean(v):
    v = np.asarray(v)
    if v.ndim == 0:
        x = v.item()
        return None if isinstance(x, float) and not np.isfinite(x) else x
    return [_clean(x) for x in v]


def pixel(d: str, row: int, col: int) -> dict:
    base = safe_path(d)
    vals = []
    npys = list(base.glob("*.npy")) + (sorted((base / "maps").rglob("*.npy")) if (base / "maps").is_dir() else [])
    for p in npys:
        try:
            a = load_npy(p)
            if a.ndim in (2, 3) and 0 <= row < a.shape[0] and 0 <= col < a.shape[1]:
                vals.append({"layer": rel(p), "label": " · ".join(pretty(p.stem)), "value": _clean(a[row, col])})
        except Exception:  # noqa: BLE001
            pass
    in_box = [i for i, b in enumerate(boxes()["boxes"]) if b[0] <= row < b[1] and b[2] <= col < b[3]]
    return {"row": row, "col": col, "values": vals, "in_boxes": in_box}


def boxes() -> dict:
    try:
        j = json.loads(BOXES.read_text(encoding="utf-8"))
        return {"granule": j.get("granule"), "boxes": j.get("exclude_boxes", [])}
    except Exception:  # noqa: BLE001
        return {"granule": None, "boxes": []}


# ----------------------------------------------------------------- scene + spectra
_scene: dict = {}


def scene() -> dict:
    if _scene:
        return _scene
    info = {"file": REFL.name, "exists": REFL.exists()}
    if REFL.exists():
        with h5py.File(REFL, "r") as f:
            info["shape"] = list(f["reflectance"].shape)
            wl = np.array(f["sensor_band_parameters"]["wavelengths"][()], np.float64)
        if np.nanmax(wl) < 100:
            wl = wl * 1000
        info["wavelengths"] = wl
        from src.preprocess import good_band_mask
        info["keep"] = np.asarray(good_band_mask(wl), bool)
        info["good_bands"] = int(info["keep"].sum())
        info["size_gb"] = round(REFL.stat().st_size / 1e9, 2)
    _scene.update(info)
    return _scene


def scene_public() -> dict:
    s = scene()
    return {k: v for k, v in s.items() if k not in ("wavelengths", "keep")}


_lib: dict = {}


def library(cr: bool):
    if cr not in _lib:
        from src.library import load_library
        s = scene()
        wl = s["wavelengths"][s["keep"]]
        _lib[cr] = load_library(str(USGS), wl, continuum=cr, smooth=cr)
    return _lib[cr]


def _angle(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return None
    x, y = a[m], b[m]
    d = np.linalg.norm(x) * np.linalg.norm(y)
    if d == 0:
        return None
    return float(np.arccos(np.clip(np.dot(x, y) / d, -1, 1)))


def spectrum(row: int, col: int) -> dict:
    s = scene()
    H, W, _ = s["shape"]
    if not (0 <= row < H and 0 <= col < W):
        raise ValueError("pixel outside scene")
    with _lock, h5py.File(REFL, "r") as f:
        v = np.array(f["reflectance"][row, col, :], np.float64)
    v[~np.isfinite(v) | (v <= -999)] = np.nan
    keep = s["keep"]
    wl = s["wavelengths"][keep]
    raw = v[keep]
    out = {"row": row, "col": col, "wavelengths": wl.tolist(), "raw": _clean(raw), "cr": None, "matches": []}
    if np.isfinite(raw).all() and np.std(raw) > 1e-6:
        from src.preprocess import continuum_remove, smooth_spectra
        cr = continuum_remove(smooth_spectra(raw[None, :]), wl)[0]
        out["cr"] = _clean(cr)
        lib = library(True)
        libraw = library(False)
        m = []
        for name, spec in lib.items():
            m.append({"name": name, "angle": _angle(cr, np.asarray(spec, float)),
                      "cr": _clean(spec), "raw": _clean(libraw.get(name))})
        m.sort(key=lambda x: 9 if x["angle"] is None else x["angle"])
        out["matches"] = m
    else:
        out["note"] = "pixel has no valid reflectance (fill / NaN)"
    return out


# ----------------------------------------------------------------- L2B
_l2b: dict = {}


def _txt(v):
    return v.decode("utf-8", "replace").strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()


def l2b():
    key = L2B.stat().st_mtime
    if _l2b.get("key") == key:
        return _l2b
    with h5py.File(L2B, "r") as f:
        ds = f["group_1_mineral_id"]
        ids = np.array(ds[()], np.int16)
        fill = int(np.asarray(ds.attrs.get("_FillValue", [-9999])).reshape(-1)[0])
        md = f["mineral_metadata"]
        index = np.array(md["index"][()]).astype(np.int64)
        names = [_txt(x) for x in md["name"][()]]
        groups = np.array(md["group"][()]).astype(int) if "group" in md else None
        libs = [_txt(x) for x in md["library"][()]] if "library" in md else None
    lookup = {int(i): (names[k], int(groups[k]) if groups is not None else None, libs[k] if libs else None)
              for k, i in enumerate(index)}
    valid = ids != fill
    u, c = np.unique(ids[valid], return_counts=True)
    order = np.argsort(-c)
    classes = []
    vivid = 0
    for rank, j in enumerate(order):
        i = int(u[j])
        nm = lookup.get(i)
        if i == 0:
            name, color = "unclassified", "#334155"
        else:
            name = nm[0] if nm else str(i)
            if vivid < len(PALETTE):
                color = PALETTE[vivid]
            else:
                import colorsys
                r, g, b = colorsys.hls_to_rgb(((vivid * 137.508) % 360) / 360, 0.6, 0.45)
                color = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
            vivid += 1
        classes.append({"id": i, "name": name, "named": bool(nm) or i == 0, "count": int(c[j]),
                        "group": nm[1] if nm else None, "library": nm[2] if nm else None,
                        "color": color, "top12": rank < 12})
    _l2b.clear()
    _l2b.update(key=key, ids=ids, summary={
        "file": L2B.name, "shape": list(ids.shape), "fill": fill, "fill_count": int((~valid).sum()),
        "n_ids": len(classes), "n_metadata": len(names), "valid_pixels": int(valid.sum()), "classes": classes})
    return _l2b


# ----------------------------------------------------------------- pipeline facts + stats
import re as _re  # noqa: E402

_PATS = {
    "scene": r"reflectance \((\d+), (\d+), (\d+)\).*?good bands (\d+)/(\d+), chunk rows (\d+)",
    "boxes": r"exclude boxes (\d+) excluded pixels (\d+)",
    "qm_offset": r"lat/lon differ.*?by up to ([\d.]+) deg",
    "pass1": r"pass 1: finite pixels (\d+), .*?threshold \(8th pct\) ([\d.]+), MNF signal sample (\d+)/(\d+), noise pairs (\d+)/(\d+)",
    "fit": r"fit sample: (\d+) .*?cap (\d+)",
    "targets": r"library targets (\d+)",
    "prepared": r"pass 2: prepared spectra (\d+)",
    "pca": r"pca components (\d+)",
    "mf": r"matched filter maps (\d+)",
    "done": r"done -> .*? in (\d+)s, peak memory (\d+) MB",
}


def _read_series(p: Path):
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out.append(float(parts[1]))
                except ValueError:
                    pass
    return out


def _find_log(base: Path):
    cands = [base / "full_run.log", base / "run.log"]
    if base.name == "subset":
        cands = [base.parent / "subset_run.log", base / "subset_run.log"] + cands
    cands += sorted(base.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        if c.exists():
            return c
    return None


def _read_text_any(p: Path) -> str:
    raw = p.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def pipeline_facts(d: str = "outputs") -> dict:
    base = safe_path(d)
    log = _find_log(base)
    text = _read_text_any(log) if log else ""
    f: dict = {"dir": d, "log": rel(log) if log else None}
    for k, pat in _PATS.items():
        m = _re.search(pat, text)
        f[k] = [float(x) if "." in x else int(x) for x in m.groups()] if m else None
    f["pass1_chunks"] = len(_re.findall(r"pass 1 chunk \d+/\d+", text))
    f["chunks"] = [{"i": int(a), "n": int(b), "r0": int(c), "r1": int(e), "prepared": int(p), "elapsed": int(s)}
                   for a, b, c, e, p, s in _re.findall(
                       r"pass 2 chunk (\d+)/(\d+) rows (\d+)-(\d+): prepared (\d+) \(elapsed (\d+)s\)", text)]
    f["windows"] = [{"name": n, "bands": int(b)} for n, b in _re.findall(r"window (\w+) bands (\d+)", text)]
    f["pca_explained"] = _read_series(base / "pca_explained.txt")
    f["mnf_eigenvalues"] = _read_series(base / "mnf_eigenvalues.txt")
    tests = None
    runs_dir = Path(__file__).resolve().parent / "runs"
    for p in sorted(runs_dir.glob("*tests.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        m = _re.search(r"(\d+) passed(?:, (\d+) failed)?", p.read_text(encoding="utf-8", errors="replace"))
        if m:
            tests = {"passed": int(m.group(1)), "failed": int(m.group(2) or 0), "log": rel(p)}
            break
    f["tests"] = tests
    return f


_stats_cache: dict = {}


def _maps_dir(d: str) -> Path:
    base = safe_path(d)
    return base / "maps" if (base / "maps").is_dir() else base


def _sample(a, n=300_000):
    flat = np.asarray(a).reshape(-1) if a.size <= n else np.asarray(a.reshape(-1)[:: max(1, a.size // n)])
    return flat


def mf_stats(d: str = "outputs", bins: int = 60) -> dict:
    md = _maps_dir(d)
    files = sorted(md.glob("mf_*.npy"))
    key = ("mf", d, tuple((str(p), p.stat().st_mtime) for p in files))
    if key in _stats_cache:
        return _stats_cache[key]
    out = []
    for p in files:
        a = load_npy(p)
        s = _sample(a).astype(np.float64)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        lo, hi = np.percentile(s, [0.5, 99.5])
        h, e = np.histogram(s, bins=bins, range=(lo, hi))
        out.append({"name": p.stem[3:], "edges": e.tolist(), "counts": h.tolist(), "n": int(s.size),
                    "mean": float(s.mean()), "p50": float(np.median(s)), "p99": float(np.percentile(s, 99)),
                    "frac_gt_0_1": float((s > 0.1).mean())})
    res = {"dir": d, "minerals": out}
    _stats_cache[key] = res
    return res


def sam_stats(d: str = "outputs", bins: int = 60) -> dict:
    md = _maps_dir(d)
    lp, ap = md / "sam_labels.npy", md / "sam_angles.npy"
    key = ("sam", d, lp.exists() and lp.stat().st_mtime, ap.exists() and ap.stat().st_mtime)
    if key in _stats_cache:
        return _stats_cache[key]
    res: dict = {"dir": d, "labels": [], "angles": None}
    try:
        names = list(library(True).keys())
    except Exception:  # noqa: BLE001
        names = []
    if lp.exists():
        lab = np.asarray(load_npy(lp))
        u, c = np.unique(lab, return_counts=True)
        mapped = int(c[u >= 0].sum()) or 1
        res["labels"] = [{"label": int(v), "name": (names[v] if 0 <= v < len(names) else ("not mapped / > 0.12 rad" if v == -1 else str(v))),
                          "count": int(n), "frac": n / mapped} for v, n in zip(u, c)]
        res["mapped"] = mapped
        res["total"] = int(c.sum())
    if ap.exists():
        s = _sample(load_npy(ap)).astype(np.float64)
        s = s[np.isfinite(s)]
        if s.size:
            h, e = np.histogram(s, bins=bins, range=(0, float(np.percentile(s, 99.5))))
            res["angles"] = {"edges": e.tolist(), "counts": h.tolist(), "n": int(s.size), "median": float(np.median(s)),
                             "frac_le_012": float((s <= 0.12).mean())}
    _stats_cache[key] = res
    return res
