# Landmine Detection, GPR + Magnetometer Sensor Fusion

Automated detection pipeline for subsurface objects (landmines/UXO) using Ground Penetrating Radar (GPR) and magnetometer data. Developed as part of the Helen Parkhurst Meesterproef project in collaboration with TNO and the Defensie Expertise Centrum MILENG.

---

## What it does

The system consists of three independent components that can be run separately or as a full end-to-end pipeline:

| Component | Script | What it does |
|---|---|---|
| GPR detector | `main.py` | Detects hyperbolic reflections in GPR B-scans |
| Magnetometer detector | `mag_upload_AS.py` | Locates ferromagnetic objects via Analytic Signal (THG) |
| Sensor fusion | `sensor_fusion.py` + `integrated_workflow.py` | Combines both detections via AND-gate matching |

---

## Project structure

```
.
├── main.py                  # GPR hyperbola detection algorithm
├── mag_upload_AS.py         # Magnetometer detection (Analytic Signal)
├── sensor_fusion.py         # Fusion logic: coordinate transform + AND-gate matching
├── integrated_workflow.py   # End-to-end pipeline: GPR → MAG → Fusion
├── helpers.py               # Hyperbola fitting, filter functions
├── models.py                # Candidate dataclass
├── test_setup.py            # Pre-flight check: verifies all files + imports
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata
└── gprdata/                 # Place your .out / .mat / .npz data files here
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `h5py`

> For synthetic magnetometer data generation, SimPEG is also required:
> ```bash
> pip install SimPEG discretize
> ```
> This is optional, detection works without it if you supply your own `.npz` data.

---

## Quick start

### Check your setup first
```bash
python test_setup.py
```
This verifies all files are present and all imports work before you run anything.

### Run GPR detection only
```bash
python main.py
```
Set your file path and parameters in the `SETTINGS` dict at the bottom of `main.py`.

### Run magnetometer detection only
```bash
python mag_upload_AS.py
```
Point `mijn_bestand` to your `.npz` magnetometer file. If no file is found, a synthetic test dataset is generated automatically (requires SimPEG).

### Run the full pipeline (GPR → Magnetometer → Fusion)
```bash
python integrated_workflow.py
```
Configure `GPR_FILE`, `MAG_FILE` and `FUSION_MAX_DISTANCE` at the bottom of the script.

---

## GPR detection parameters

Configured via the `SETTINGS` dict in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `sigma` | `0.050` | Detection threshold multiplier, lower = more candidates, higher = stricter |
| `min_trace` | `300` | Minimum depth (trace index) to search |
| `max_trace` | `1800` | Maximum depth (trace index) to search |
| `intensity_T1` | `None` | Amplitude threshold, `None` = auto (70th percentile of candidates) |
| `min_trace_width` | `150` | Minimum blob width in traces (removes vertical clutter) |
| `max_fit_error` | `150` | Maximum allowed hyperbola fit error (MAE) |
| `ground_search_fraction` | `0.25` | Fraction of traces to scan for ground reflection |
| `ground_margin` | `5` | Extra traces to skip after detected ground |

### Output
- Console log with per-candidate accept/reject reasoning
- `visual_gpr_data.png`, 4-panel figure: raw data, binary mask, candidates, validated reflections with fitted hyperbolas

---

## Sensor fusion parameters

Configured in `integrated_workflow.py`:

| Parameter | Default | Description |
|---|---|---|
| `FUSION_MAX_DISTANCE` | `0.5` | Max distance (meters) between GPR and MAG detection to count as a match |
| `TRACE_SPACING` | `0.1` | Physical distance per trace index (meters) |
| `SAMPLE_SPACING` | `0.05` | Physical distance per sample index (meters) |

### How it works
1. GPR trace/sample indices are converted to world coordinates (meters) using `trace_spacing` and `sample_spacing`
2. For each GPR detection, the nearest magnetometer detection is found (Euclidean distance)
3. If the distance is within `max_distance`, it's a match (AND-gate)
4. A fusion score is computed: `0.6 × distance_score + 0.4 × amplitude_score`
5. The fused object location is the midpoint between the two detections

### Output
- `sensor_fusion_result.png`, map of GPR, MAG and fused detections + match quality chart
- `fusion_results.csv`, table of verified objects with coordinates and scores

---

## Data format

| Sensor | Expected format | Notes |
|---|---|---|
| GPR | `.out` (gprMax HDF5) or `.mat` | Field `['rxs']['rx1']['Ey']` |
| Magnetometer | `.npz` | Keys: `rx_locs` (Nx3 array), `dpred` (N array of TMI values in nT) |

Place all data files in the `gprdata/` folder.

---

## Known limitation: coordinate registration

The AND-gate fusion requires both sensors to have scanned **the same physical area**. When running on the included example data (Chelton GPR field scan + SimPEG synthetic magnetometer data), no matches are found because the two datasets cover different spatial areas. This is expected behaviour, the fusion logic is correct, but end-to-end validation requires co-registered field measurements from both sensors simultaneously.

---

## Pipeline overview

```
GPR .out file                    MAG .npz file
      │                                │
      ▼                                ▼
Background subtraction          Grid interpolation
Gain correction                 Gradient computation
Ground cutoff detection         Analytic Signal (THG)
Blob extraction                 Peak detection
Hyperbola fitting               │
Amplitude filter                │
      │                         │
      ▼                         ▼
  GPR detections           MAG detections
  (world coords)           (world coords)
      │                         │
      └──────────┬──────────────┘
                 ▼
         AND-gate matching
         Fusion score
                 │
                 ▼
        fusion_results.csv
        sensor_fusion_result.png
```
