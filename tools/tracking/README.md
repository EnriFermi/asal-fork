# Flow-Lenia Tracking Tools

Utilities in this package are for video-tracking experiments and baselines. They are separate from the core optimization and MSPD experiment code.

Main entrypoints:

| File | Purpose |
|---|---|
| `flowlenia_classic_tracker.py` | Classic HSV/CC Flow-Lenia tracker. |
| `sam2_pipeline.py` | SAM2-assisted tracking pipeline. |
| `baseline_trackpy.py` | Trackpy baseline. |
| `baseline_ultrack.py` | Ultrack baseline. |
| `baseline_btrack.py` | BayesianTracker baseline. |
| `baseline_trackmate.py` | TrackMate export/import helper. |
| `import_trackmate_csv.py` | Converts TrackMate CSV outputs to repo tracking artifacts. |

Use `scripts/bench.py` to run multiple baselines on one video.

