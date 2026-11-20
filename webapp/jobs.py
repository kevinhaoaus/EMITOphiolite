"""One-at-a-time subprocess jobs with live log capture and run history."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(__file__).resolve().parent / "runs"
RUNS.mkdir(exist_ok=True)
OUT = ROOT / "outputs"
PY = sys.executable
REFL_REL = r"data\EMIT_L2A_RFL_001_20260220T072251_2605105_013.nc"
L2B_REL = r"data\EMIT_L2B_MIN_001_20260326T084318_2608506_030.nc"
BOXES_REL = r"data\manual_mask_boxes.json"

_CHUNK = re.compile(r"(?i)\bchunk\D{0,20}?(\d+)\s*(?:/|of)\s*(\d+)")
_ROWS = re.compile(r"(?i)\brows?\s+(\d+)\s*[:\-]\s*(\d+)\s*(?:/|of)\s*(\d+)")
_PCT = re.compile(r"\[\s*(\d{1,3})%\]")
_PASS = re.compile(r"(?i)\b(pass\s*\d[^\n]{0,60})")
_SUBSET_RANGE = ("--rows", "--row-range", "--subset-rows", "--subset")
_SUBSET_STOP = ("--max-rows", "--row-stop", "--row-end", "--n-rows", "--rows-stop")


def _env():
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return env


# ----------------------------------------------------------------- CLI discovery
_help_cache: dict = {}


def parse_help(text: str) -> list[dict]:
    flags: list[dict] = []
    cur = None
    opt = re.compile(r"^\s{2}(?:-\w[^,\s]*(?:\s\S+)?,\s+)?(--[\w-]+)(?:[ =](\S+))?(?:\s{2,}(.*))?$")
    for line in text.splitlines():
        m = opt.match(line)
        if m:
            cur = {"flag": m.group(1), "metavar": m.group(2), "help": (m.group(3) or "").strip()}
            flags.append(cur)
            continue
        if cur is not None and re.match(r"^\s{6,}\S", line):
            cur["help"] = (cur["help"] + " " + line.strip()).strip()
        elif not line.strip():
            cur = None
    for f in flags:
        f["is_bool"] = f["metavar"] is None
        d = re.search(r"\(default:?\s*([^)]+)\)", f["help"]) or re.search(r"(?i)default\s+([\w.\-:]+)", f["help"])
        f["default"] = d.group(1).strip() if d else None
    return [f for f in flags if f["flag"] != "--help"]


def pipeline_flags() -> dict:
    p = ROOT / "pipeline.py"
    key = p.stat().st_mtime if p.exists() else None
    if _help_cache.get("key") == key and "val" in _help_cache:
        return _help_cache["val"]
    text, err = "", None
    try:
        r = subprocess.run([PY, str(p), "--help"], cwd=ROOT, capture_output=True, text=True,
                           timeout=120, env=_env())
        text = r.stdout or r.stderr
        if r.returncode != 0:
            err = (r.stderr or "")[-800:]
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    flags = parse_help(text)
    names = {f["flag"] for f in flags}
    subset = next((f for f in _SUBSET_RANGE if f in names), None) or next((f for f in _SUBSET_STOP if f in names), None)
    val = {"flags": flags, "help": text, "error": err, "subset_flag": subset}
    _help_cache.update(key=key, val=val)
    return val


# ----------------------------------------------------------------- command build
def safe_out(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    o = OUT.resolve()
    if p != o and o not in p.parents:
        raise ValueError("output dir must be inside outputs\\")
    return p


def build_command(kind: str, params: dict) -> tuple[list[str], str]:
    params = params or {}
    if kind == "tests":
        return [PY, "-m", "pytest", "tests", "-q", "--color=no"], "outputs"
    if kind == "quicklook":
        out = safe_out(params.get("out") or "outputs")
        dest = out / "emit_l2b_mineral_map.png"
        return [PY, "-u", r"scripts\mineral_quicklook.py", "--input", L2B_REL, "--output",
                str(dest.relative_to(ROOT))], str(out.relative_to(ROOT))
    if kind not in ("smoke", "full"):
        raise ValueError(f"unknown job kind {kind}")
    info = pipeline_flags()
    names = {f["flag"]: f for f in info["flags"]}
    default_out = r"outputs\subset" if kind == "smoke" else "outputs"
    out = safe_out(params.get("out") or default_out)
    rel_out = str(out.relative_to(ROOT))
    cmd = [PY, "-u", "pipeline.py", "--reflectance", REFL_REL, "--usgs", "usgs", "--out", rel_out]
    if params.get("components"):
        cmd += ["--components", str(int(params["components"]))]
    for key, flag in (("max_fit_pixels", "--max-fit-pixels"), ("chunk_rows", "--chunk-rows")):
        if params.get(key) and flag in names:
            cmd += [flag, str(int(params[key]))]
    if params.get("mask_boxes", kind == "full"):
        cmd += ["--mask-boxes", BOXES_REL]
    if kind == "smoke":
        rows = int(params.get("smoke_rows") or 256)
        sf = info.get("subset_flag")
        if sf is None:
            raise ValueError("pipeline.py has no row-subset flag yet (looked for --rows / --max-rows ...). "
                             "Run again once the pipeline rewrite lands.")
        if sf in _SUBSET_RANGE:
            mv = (names[sf].get("metavar") or "").upper()
            cmd += [sf, f"0:{rows}"] if (":" in mv or "A" in mv or mv in ("ROWS", "RANGE")) else [sf, str(rows)]
        else:
            if "--row-start" in names:
                cmd += ["--row-start", "0"]
            cmd += [sf, str(rows)]
    reserved = {c for c in cmd if c.startswith("--")}
    for flag, value in (params.get("extra") or {}).items():
        if flag not in names or flag in reserved or value in (None, "", False):
            continue
        cmd += [flag] if names[flag]["is_bool"] else [flag, str(value)]
    return cmd, rel_out


def display(cmd: list[str]) -> str:
    parts = ["py -3.11" if c == PY else (f'"{c}"' if " " in c else c) for c in cmd]
    return " ".join(parts)


# ----------------------------------------------------------------- jobs
class Job:
    def __init__(self, kind, cmd, out_dir, params):
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{kind}"
        self.kind, self.cmd, self.out_dir, self.params = kind, cmd, out_dir, params
        self.lines: list[dict] = []
        self.state = "running"
        self.exit_code = None
        self.progress = None
        self.phase = ""
        self.started = time.time()
        self.ended = None
        self.log_path = RUNS / f"{self.id}.log"
        self.proc = None
        self.stopped = False

    def record(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "command": display(self.cmd), "args": self.cmd[1:],
            "params": self.params, "out_dir": self.out_dir, "state": self.state,
            "exit_code": self.exit_code, "started": self.started, "ended": self.ended,
            "runtime": (self.ended or time.time()) - self.started, "progress": self.progress,
            "phase": self.phase, "log": str(self.log_path.relative_to(ROOT)), "n_lines": len(self.lines),
        }

    def save(self):
        (RUNS / f"{self.id}.json").write_text(json.dumps(self.record(), indent=2), encoding="utf-8")

    def _parse(self, text):
        m = _ROWS.search(text)
        if m and int(m.group(3)):
            self.progress = min(1.0, int(m.group(2)) / int(m.group(3)))
        m = _CHUNK.search(text)
        if m and int(m.group(2)):
            self.progress = min(1.0, int(m.group(1)) / int(m.group(2)))
        m = _PCT.search(text)
        if m:
            self.progress = int(m.group(1)) / 100.0
        m = _PASS.search(text)
        if m:
            self.phase = m.group(1).strip()

    def run(self):
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with open(self.log_path, "w", encoding="utf-8") as log:
            log.write(f"$ {display(self.cmd)}\n")
            try:
                self.proc = subprocess.Popen(self.cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding="utf-8", errors="replace", bufsize=1,
                                             env=_env(), creationflags=flags)
                for raw in self.proc.stdout:
                    text = raw.rstrip("\r\n")
                    self.lines.append({"t": time.time() - self.started, "text": text})
                    log.write(text + "\n")
                    log.flush()
                    self._parse(text)
                self.exit_code = self.proc.wait()
            except Exception as exc:  # noqa: BLE001
                self.lines.append({"t": time.time() - self.started, "text": f"[webapp] failed to start: {exc}"})
                self.exit_code = -1
        self.ended = time.time()
        self.state = "stopped" if self.stopped else ("success" if self.exit_code == 0 else "failed")
        if self.state == "success":
            self.progress = 1.0
        self.save()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.stopped = True
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
            else:
                self.proc.kill()


class JobManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.current: Job | None = None
        self.jobs: dict[str, Job] = {}
        for p in RUNS.glob("*.json"):  # mark runs orphaned by a server restart
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                if rec.get("state") == "running":
                    rec["state"] = "interrupted"
                    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass

    def start(self, kind, params):
        with self.lock:
            if self.current and self.current.state == "running":
                raise RuntimeError(f"job {self.current.id} is still running")
            cmd, out_dir = build_command(kind, params)
            job = Job(kind, cmd, out_dir, params)
            job.save()
            self.current = job
            self.jobs[job.id] = job
            threading.Thread(target=job.run, daemon=True).start()
            return job

    def history(self):
        recs = []
        for p in sorted(RUNS.glob("*.json"), reverse=True):
            try:
                recs.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        live = {j.id: j.record() for j in self.jobs.values() if j.state == "running"}
        return [live.get(r["id"], r) for r in recs]


MANAGER = JobManager()
