"""FastAPI server for the EMIT Oman-ophiolite dashboard. Binds 127.0.0.1 only."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from webapp import data, jobs  # noqa: E402

app = FastAPI(title="Oman Ophiolite · EMIT Mineral Mapping")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def _err(exc: Exception, code=400):
    raise HTTPException(code, str(exc))


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    try:
        sc = data.scene_public()
    except Exception as exc:  # noqa: BLE001
        sc = {"error": str(exc)}
    cur = jobs.MANAGER.current
    return {"root": str(ROOT), "scene": sc, "outputs": data.output_dirs(),
            "l2b_exists": data.L2B.exists(), "boxes": len(data.boxes()["boxes"]),
            "job": cur.record() if cur else None, "python": sys.version.split()[0]}


@app.get("/api/pipeline/flags")
def flags():
    return jobs.pipeline_flags()


@app.post("/api/pipeline/preview")
def preview(body: dict = Body(...)):
    try:
        cmd, out = jobs.build_command(body.get("kind"), body.get("params") or {})
        return {"command": jobs.display(cmd), "out_dir": out}
    except Exception as exc:  # noqa: BLE001
        return {"command": None, "error": str(exc)}


@app.post("/api/jobs")
def start_job(body: dict = Body(...)):
    try:
        job = jobs.MANAGER.start(body.get("kind"), body.get("params") or {})
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc))
    return job.record()


@app.get("/api/jobs/current")
def current():
    cur = jobs.MANAGER.current
    return cur.record() if cur else None


@app.post("/api/jobs/{job_id}/stop")
def stop(job_id: str):
    job = jobs.MANAGER.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such live job")
    job.stop()
    return {"ok": True}


@app.get("/api/jobs/{job_id}/stream")
async def stream(job_id: str, since: int = 0):
    job = jobs.MANAGER.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such live job")

    async def gen():
        i = since
        last = 0.0
        while True:
            n = len(job.lines)
            if i < n:
                batch = job.lines[i:n]
                yield f"event: lines\ndata: {json.dumps({'from': i, 'lines': batch})}\n\n"
                i = n
            now = time.time()
            if now - last > 0.5 or job.state != "running":
                yield f"event: status\ndata: {json.dumps(job.record())}\n\n"
                last = now
            if job.state != "running" and i >= len(job.lines):
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(0.2)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs")
def runs():
    out = jobs.MANAGER.history()
    for r in out:
        r["images"] = data.list_images(r.get("out_dir") or "outputs")
    return out


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str):
    p = jobs.RUNS / f"{Path(run_id).name}.log"
    if not p.exists():
        raise HTTPException(404, "no log")
    return Response(p.read_text(encoding="utf-8", errors="replace"), media_type="text/plain; charset=utf-8")


@app.get("/api/outputs/dirs")
def dirs():
    return data.output_dirs()


@app.get("/api/layers")
def layers(dir: str = "outputs"):
    try:
        return data.list_layers(dir)
    except FileNotFoundError:
        return {"dir": dir, "layers": [], "texts": []}
    except Exception as exc:  # noqa: BLE001
        _err(exc)


@app.get("/api/render")
def render(path: str, r: int = 0, g: int = 1, b: int = 2, lo: float = 2, hi: float = 98,
           cmap: str = "viridis", component: int | None = None):
    try:
        png, meta = data.render(path, (r, g, b), lo, hi, cmap, component)
    except Exception as exc:  # noqa: BLE001
        _err(exc)
    return Response(png, media_type="image/png", headers={"X-Meta": json.dumps(meta),
                                                          "Access-Control-Expose-Headers": "X-Meta"})


@app.get("/api/pixel")
def pixel(dir: str, row: int, col: int):
    try:
        return data.pixel(dir, row, col)
    except Exception as exc:  # noqa: BLE001
        _err(exc)


@app.get("/api/spectrum")
def spectrum(row: int, col: int):
    try:
        return data.spectrum(row, col)
    except Exception as exc:  # noqa: BLE001
        _err(exc)


@app.get("/api/library")
def library_names():
    try:
        return {"names": list(data.library(True).keys())}
    except Exception as exc:  # noqa: BLE001
        return {"names": [], "error": str(exc)}


@app.get("/api/pipeline/facts")
def facts(dir: str = "outputs"):
    try:
        return data.pipeline_facts(dir)
    except Exception as exc:  # noqa: BLE001
        _err(exc)


@app.get("/api/stats/mf")
def stats_mf(dir: str = "outputs"):
    try:
        return data.mf_stats(dir)
    except Exception as exc:  # noqa: BLE001
        _err(exc)


@app.get("/api/stats/sam")
def stats_sam(dir: str = "outputs"):
    try:
        return data.sam_stats(dir)
    except Exception as exc:  # noqa: BLE001
        _err(exc)


@app.get("/api/boxes")
def boxes():
    return data.boxes()


@app.get("/api/file")
def file(path: str, download: bool = False):
    try:
        p = data.safe_path(path)
    except Exception as exc:  # noqa: BLE001
        _err(exc, 404)
    return FileResponse(p, filename=p.name if download else None,
                        content_disposition_type="attachment" if download else "inline",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/l2b/summary")
def l2b_summary():
    try:
        return data.l2b()["summary"]
    except Exception as exc:  # noqa: BLE001
        _err(exc, 500)


@app.get("/api/l2b/ids.bin")
def l2b_ids():
    ids = data.l2b()["ids"]
    return Response(ids.astype("<i2").tobytes(), media_type="application/octet-stream")


def main():
    import uvicorn
    port = 8000
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"EMIT dashboard -> http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
