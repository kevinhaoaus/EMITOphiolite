"""Export a static, read-only copy of the EMIT dashboard into docs/ for GitHub Pages."""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import data, jobs  # noqa: E402

MAX_EDGE = 900
JPEG_QUALITY = 82  # used only as a size cap fallback; we write PNG
PNG_COMPRESS = 6


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=str, separators=(",", ":")), encoding="utf-8")


def _rel(p: str) -> str:
    return str(p).replace("\\", "/")


def compress_png(src: Path, dest: Path, max_edge: int = MAX_EDGE) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGBA") if im.mode in ("P", "LA") else im
        w, h = im.size
        if max(w, h) > max_edge:
            scale = max_edge / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        # Flatten semi-opaque onto the site ivory so we can drop the alpha and shrink.
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (245, 244, 238))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True, compress_level=PNG_COMPRESS)
        dest.write_bytes(buf.getvalue())
        return {"w": im.size[0], "h": im.size[1], "size": dest.stat().st_size}


def save_rgba(arr, dest: Path, max_edge: int = MAX_EDGE) -> dict:
    from PIL import Image as _Image
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = _Image.fromarray(arr, "RGBA")
    w, h = im.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), _Image.Resampling.LANCZOS)
    bg = _Image.new("RGB", im.size, (245, 244, 238))
    bg.paste(im, mask=im.split()[-1])
    buf = io.BytesIO()
    bg.save(buf, "PNG", optimize=True, compress_level=PNG_COMPRESS)
    dest.write_bytes(buf.getvalue())
    return {"w": bg.size[0], "h": bg.size[1], "size": dest.stat().st_size}


def render_name(path: str, lo=2, hi=98, cmap="viridis", component=None, r=0, g=1, b=2) -> str:
    stem = _rel(path).replace("/", "__")
    if component is not None:
        return f"{stem}__c{int(component)}__{cmap}__{lo}-{hi}.png"
    return f"{stem}__rgb{int(r)}{int(g)}{int(b)}__{cmap}__{lo}-{hi}.png"


def copy_frontend(docs: Path) -> None:
    src = HERE / "static"
    dest = docs / "static"
    dest.mkdir(parents=True, exist_ok=True)
    html = (src / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/static/app.css"', 'href="static/app.css"')
    html = html.replace('src="/static/app.js"', 'src="static/app.js"')
    if "window.STATIC" not in html:
        html = html.replace(
            '<script src="static/app.js"></script>',
            '<script>window.STATIC=true;</script>\n<script src="static/app.js"></script>',
        )
    (docs / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(src / "app.css", dest / "app.css")
    shutil.copy2(src / "app.js", dest / "app.js")
    (docs / ".nojekyll").write_text("", encoding="utf-8")


def export_json(docs: Path, dirs: list[str]) -> None:
    api = docs / "api"
    api.mkdir(parents=True, exist_ok=True)

    try:
        scene = data.scene_public()
    except Exception as exc:  # noqa: BLE001
        scene = {"error": str(exc), "shape": [1280, 1242, 285], "good_bands": 225}

    out_dirs = []
    for d in data.output_dirs():
        if _rel(d["dir"]) in dirs:
            out_dirs.append({**d, "dir": _rel(d["dir"])})

    status = {
        "root": "static",
        "scene": scene,
        "outputs": out_dirs,
        "l2b_exists": True,
        "boxes": len(data.boxes()["boxes"]),
        "job": None,
        "python": "static",
        "static": True,
    }
    _dump(api / "status.json", status)
    _dump(api / "dirs.json", out_dirs)
    _dump(api / "boxes.json", data.boxes())
    _dump(api / "jobs-current.json", None)

    try:
        flags = jobs.pipeline_flags()
        # Drop local machine paths from help text if huge; keep flags list.
        _dump(api / "pipeline" / "flags.json", flags)
    except Exception as exc:  # noqa: BLE001
        _dump(api / "pipeline" / "flags.json", {"flags": [], "help": "", "error": str(exc), "subset_flag": "--rows"})

    try:
        names = list(data.library(True).keys())
    except Exception as exc:  # noqa: BLE001
        names = ["Antigorite", "Augite", "Brucite", "Calcite", "Dolomite", "Enstatite",
                 "Kaolinite", "Lizardite", "Magnetite", "Montmorillonite", "Muscovite",
                 "Olivine", "Quartz"]
        print("library fallback:", exc)
    _dump(api / "library.json", {"names": names})

    # Prefer the full-run facts (CLI log), not a failed webapp job.
    facts = data.pipeline_facts("outputs")
    _dump(api / "pipeline" / "facts.json", facts)
    _dump(api / "stats" / "mf.json", data.mf_stats("outputs"))
    _dump(api / "stats" / "sam.json", data.sam_stats("outputs"))

    for d in dirs:
        L = data.list_layers(d)
        key = d.replace("/", "-")
        _dump(api / f"layers-{key}.json", L)
        if d == "outputs":
            _dump(api / "layers.json", L)

    # L2B
    try:
        bundle = data.l2b()
        _dump(api / "l2b" / "summary.json", bundle["summary"])
        (api / "l2b" / "ids.bin").write_bytes(bundle["ids"].astype("<i2").tobytes())
    except Exception as exc:  # noqa: BLE001
        print("L2B export failed:", exc)

    # Runs: real history plus a showcase record for the successful full scene.
    history = []
    try:
        history = jobs.MANAGER.history()
        for r in history:
            r["images"] = data.list_images(r.get("out_dir") or "outputs")
            r["command"] = (r.get("command") or "").replace("C:\\Users\\Default.DESKTOP-NUBLEQ1\\Desktop\\KernelHackthon-Geology+\\", "")
    except Exception as exc:  # noqa: BLE001
        print("history failed:", exc)

    showcase = {
        "id": "full-013-cli",
        "kind": "full",
        "command": "py -3.11 -u pipeline.py --reflectance data/EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc --usgs usgs --out outputs --components 8 --max-fit-pixels 80000 --chunk-rows 128 --mask-boxes data/manual_mask_boxes.json",
        "params": {"components": 8, "max_fit_pixels": 80000, "chunk_rows": 128, "mask_boxes": True},
        "out_dir": "outputs",
        "state": "success",
        "exit_code": 0,
        "started": None,
        "ended": None,
        "runtime": (facts.get("done") or [None, None])[0],
        "progress": 1.0,
        "phase": "done",
        "log": "full-013-cli",
        "n_lines": None,
        "images": data.list_images("outputs"),
        "showcase": True,
    }
    if facts.get("done"):
        showcase["runtime"] = facts["done"][0]
        showcase["peak_mb"] = facts["done"][1]
    runs = [showcase] + [r for r in history if r.get("id") != showcase["id"]]
    _dump(api / "runs.json", runs)

    # Logs
    logs_dir = api / "runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    src_log = ROOT / "outputs" / "full_run.log"
    if src_log.exists():
        (logs_dir / "full-013-cli.log").write_bytes(src_log.read_bytes())
    runs_src = HERE / "runs"
    if runs_src.is_dir():
        for p in runs_src.glob("*.log"):
            (logs_dir / p.name).write_bytes(p.read_bytes())

    # Copy last-run log text for the Run tab console
    if src_log.exists():
        (api / "last-run.log").write_bytes(src_log.read_bytes())


def copy_figures(docs: Path, dirs: list[str]) -> list[str]:
    copied = []
    skip = {"deck_usgs_spectra.png", "EMIT_Oman_Ophiolite_Defense.pptx"}
    for d in dirs:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*.png")):
            if p.name in skip or p.name.lower().endswith(".pptx"):
                continue
            dest = docs / "files" / d / p.name
            compress_png(p, dest)
            copied.append(str(dest.relative_to(docs)))
            print(f"  figure {d}/{p.name} -> {dest.stat().st_size // 1024} KB")
    return copied


def render_maps(docs: Path, quick: bool) -> list[str]:
    """Pre-render npy layers used by Map and Scores tabs."""
    out = []
    manifest = {}
    dest_dir = docs / "renders"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Default stretch set. Quick mode: only 2-98.
    stretches = [(2, 98)] if quick else [(0, 100), (2, 98), (5, 95), (10, 90)]
    cmap_mf = ["viridis"] if quick else ["viridis", "magma", "gray"]

    layers = data.list_layers("outputs")["layers"]
    maps = [l for l in layers if l.get("type") in ("multi", "scalar", "labels") and l.get("file")]

    # Prioritize Scores + common explorer views.
    def priority(l):
        f = l["file"]
        if "pca_scores" in f:
            return 0
        if "/mf_" in f or "\\mf_" in f:
            return 1
        if "sam_" in f or "mnf_scores" in f:
            return 2
        return 3

    maps.sort(key=priority)

    for layer in maps:
        path = layer["file"]
        kind = layer["type"]
        entries = {}
        jobs_spec = []
        if kind == "labels":
            jobs_spec.append(dict(lo=2, hi=98, cmap="viridis", component=None, r=0, g=1, b=2))
        elif kind == "scalar":
            for lo, hi in stretches:
                for cmap in cmap_mf:
                    if cmap != "viridis" and (lo, hi) != (2, 98):
                        continue
                    jobs_spec.append(dict(lo=lo, hi=hi, cmap=cmap, component=None, r=0, g=1, b=2))
        elif kind == "multi":
            # RGB default
            for lo, hi in stretches[:1]:
                jobs_spec.append(dict(lo=lo, hi=hi, cmap="viridis", component=None, r=0, g=1, b=2))
            k = int(layer["shape"][2]) if layer.get("shape") and len(layer["shape"]) > 2 else 8
            # Individual components: all PCs for pca_scores; first 3 otherwise in quick mode
            ncomp = k if "pca_scores" in path or not quick else min(3, k)
            cmap = "gray" if "pca_scores" in path else "viridis"
            for i in range(ncomp):
                for lo, hi in stretches:
                    jobs_spec.append(dict(lo=lo, hi=hi, cmap=cmap, component=i, r=0, g=1, b=2))

        for spec in jobs_spec:
            name = render_name(path, **spec)
            dest = dest_dir / name
            try:
                png, meta = data.render(
                    path,
                    (spec["r"], spec["g"], spec["b"]),
                    float(spec["lo"]),
                    float(spec["hi"]),
                    spec["cmap"],
                    spec["component"],
                )
                im = Image.open(io.BytesIO(png))
                w, h = im.size
                if max(w, h) > MAX_EDGE:
                    scale = MAX_EDGE / max(w, h)
                    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
                if im.mode == "RGBA":
                    bg = Image.new("RGB", im.size, (245, 244, 238))
                    bg.paste(im, mask=im.split()[-1])
                    im = bg
                buf = io.BytesIO()
                im.save(buf, "PNG", optimize=True, compress_level=PNG_COMPRESS)
                dest.write_bytes(buf.getvalue())
                key = (
                    f"c{spec['component']}|{spec['cmap']}|{spec['lo']}-{spec['hi']}"
                    if spec["component"] is not None
                    else f"rgb{spec['r']}{spec['g']}{spec['b']}|{spec['cmap']}|{spec['lo']}-{spec['hi']}"
                )
                entries[key] = {
                    "file": "renders/" + name,
                    "meta": meta,
                    "w": im.size[0],
                    "h": im.size[1],
                }
                out.append(name)
                print(f"  render {name} {dest.stat().st_size // 1024} KB")
            except Exception as exc:  # noqa: BLE001
                print(f"  render failed {path} {spec}: {exc}")

        # Fallbacks onto matplotlib figures when present
        stem = Path(path).stem
        fig = docs / "files" / "outputs" / f"{stem}.png"
        if not fig.exists() and stem.startswith("mf_"):
            fig = docs / "files" / "outputs" / f"{stem}.png"
        if fig.exists():
            entries.setdefault("figure", {"file": "files/outputs/" + fig.name})
        if "pca_scores" in path and (docs / "files" / "outputs" / "pca_rgb.png").exists():
            entries.setdefault("figure", {"file": "files/outputs/pca_rgb.png"})
        manifest[_rel(path)] = entries

    _dump(docs / "api" / "renders.json", manifest)
    return out


def maybe_spectra(docs: Path, step: int = 80) -> None:
    """Optional coarse grid of pixel values (no full cube spectra in quick mode)."""
    try:
        md = data._maps_dir("outputs")
        lab_p, ang_p = md / "sam_labels.npy", md / "sam_angles.npy"
        if not lab_p.exists():
            return
        import numpy as np

        lab = np.asarray(data.load_npy(lab_p))
        H, W = lab.shape[:2]
        rows = list(range(0, H, step))
        cols = list(range(0, W, step))
        # Store a compact grid plus names so hover can snap to nearest sample.
        names = []
        try:
            names = list(data.library(True).keys())
        except Exception:  # noqa: BLE001
            pass
        grid = {
            "step": step,
            "shape": [H, W],
            "rows": rows,
            "cols": cols,
            "names": names,
            "labels": lab[::step, ::step].astype("int16").tolist(),
        }
        if ang_p.exists():
            ang = np.asarray(data.load_npy(ang_p))
            grid["angles"] = np.where(np.isfinite(ang[::step, ::step]), ang[::step, ::step], None).tolist()
        _dump(docs / "api" / "pixel-grid.json", grid)
        print(f"  pixel grid step={step} {H}x{W}")
    except Exception as exc:  # noqa: BLE001
        print("pixel grid failed:", exc)


def summarize(docs: Path) -> None:
    files = [p for p in docs.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    big = sorted(files, key=lambda p: -p.stat().st_size)[:12]
    print(f"\nSite size: {total / 1e6:.2f} MB in {len(files)} files")
    for p in big:
        print(f"  {p.stat().st_size / 1e6:.2f} MB  {p.relative_to(docs)}")
    if total > 100e6:
        print("WARNING: docs/ exceeds 100 MB")
    over = [p for p in files if p.stat().st_size >= 25e6]
    if over:
        print("WARNING: files >= 25 MB:", [str(p.relative_to(docs)) for p in over])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--quick", action="store_true", help="Fewer stretch/cmap variants; faster first publish")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--dirs", default="outputs,outputs/subset")
    args = ap.parse_args()

    docs = (ROOT / args.out).resolve()
    dirs = [d.strip().replace("\\", "/") for d in args.dirs.split(",") if d.strip()]

    print("Exporting static site ->", docs)
    if docs.exists():
        # Remove only our generated trees; leave unknown files alone in case
        # another agent wrote something here (none expected).
        for name in ("index.html", ".nojekyll", "static", "api", "files", "renders"):
            p = docs / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    docs.mkdir(parents=True, exist_ok=True)

    print("Frontend…")
    copy_frontend(docs)
    print("JSON…")
    export_json(docs, dirs)
    print("Figures…")
    copy_figures(docs, dirs)
    if not args.skip_render:
        print("Renders…")
        render_maps(docs, quick=args.quick)
    maybe_spectra(docs, step=80)
    summarize(docs)
    print("Done.")


if __name__ == "__main__":
    main()
