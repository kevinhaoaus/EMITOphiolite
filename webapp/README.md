# EMIT Oman-ophiolite dashboard (local web app)

A local FastAPI + plain HTML/JS dashboard for the pipeline in this repo. No CDN,
no build step. It binds to **127.0.0.1 only**.

## Start

```powershell
cd "C:\Users\Default.DESKTOP-NUBLEQ1\Desktop\KernelHackthon-Geology+"
py -3.11 -m pip install -r webapp\requirements.txt
py -3.11 -m webapp            # or: py -3.11 webapp\app.py   (add --port 8001 to change port)
```

Open http://127.0.0.1:8000 . Or double-click `webapp\start.bat`.

## Pages

* **Run**: buttons for *Run tests* (`py -3.11 -m pytest tests`), *Smoke run* (first N rows via
  `--rows 0:N` into `outputs\subset`), *Full 013 scene* (`--mask-boxes data\manual_mask_boxes.json
  --components 8` into `outputs`), and *L2B quicklook* (`scripts\mineral_quicklook.py`).
  The flags come from `py -3.11 pipeline.py --help` at runtime, so new flags show up under
  *Advanced* by themselves. The log streams live over SSE with elapsed time and a progress bar
  that reads `chunk i/n`, `rows a:b / H` and pytest `[NN%]`. *Stop* kills the process tree
  (`taskkill /T /F`). Only one job runs at a time; a second start gets HTTP 409.
* **Pipeline**: run KPIs (runtime, peak memory, tests passed, spectra mapped, fit sample, chunks).
  A clickable SVG diagram of the pipeline: click a step to see what it does, its equation, its
  parameters and the real numbers, all parsed live from `outputs\full_run.log`,
  `pca_explained.txt`, `mnf_eigenvalues.txt` and the newest test log. Charts (plain SVG, hover
  for values): PCA explained variance with a cumulative line, MNF scree, matched-filter score
  histograms per mineral, SAM label area, SAM angle histogram (0.12 rad marked), and prepared
  pixels per pass-2 chunk. The histograms are computed on the server from subsampled
  `maps\*.npy` memmaps.
* **Scores**: each PCA component in grayscale (PC1 to PC8) next to any matched-filter score map,
  with a shared percentile stretch, an MF colormap, and thumbnail grids of every PC and every
  mineral.
* **Map explorer**: choose `outputs` or `outputs\subset`. Layers are the `.npy` score maps
  (`maps\*.npy`) plus the PNG figures. `.npy` maps are rendered on the server with an
  adjustable percentile stretch and a colormap, and you can pick which components go to RGB
  for (H,W,k) maps. Scroll to zoom, drag to pan, double-click to fit. There is a minimap
  and a scale bar (60 m/pixel). Hovering shows row/col and the value under the cursor.
  Clicking pins the pixel, shows its values from every map, and plots its good-band spectrum
  read straight from the L2A `.nc` (raw or Savitzky–Golay + continuum removed with
  `src.preprocess`), next to the USGS library from `src.library.load_library`, ranked by
  spectral angle. You can switch on an overlay of the 22 exclude boxes.
* **L2B minerals**: `group_1_mineral_id` drawn in the browser. Names are joined where
  `mineral_metadata/index == id`, never by array position. Id 0 is "unclassified" and hidden
  by default. The fill value -9999 is never drawn. Click legend chips to toggle classes
  (All / Top 12 / None). There is a pixel-count table for every id present.
* **Runs**: every UI-launched run is saved as `webapp\runs\<id>.json` (command, args, exit
  code, runtime, output dir) plus `<id>.log`. You can open logs, browse output thumbnails and
  download them, and compare the same image from two runs or output folders with a swipe or
  side-by-side view.

## Notes / gaps

* The run gallery shows the files that are in a run's output folder *now*. Outputs are not
  snapshotted per run.
* Label colours for `sam_labels` / `mf_best_labels` assume label *i* = the *i*-th library
  mineral in `load_library` order.
* PNG figures are matplotlib figures with axes, so pixel hover/spectra only work on `.npy` layers.
