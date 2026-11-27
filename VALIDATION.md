# Validation

## Dynamic replay update — 2026-11-27

- All 39 published Python sources parsed successfully.
- Ran `tools/smoke_test.py` and `tools/check_dynamic.py` from an isolated copy of the public source with CUDA hidden: both passed. Checks include geometry conventions, trajectory limits, candidate restrictions, CapacityHead(96) forward execution and archived result/source hashes.
- Confirmed environment overrides for data, BEVFusion, trajectory inputs and the road-head directory without accessing those data paths.
- Runtime: Python 3.8.20 / PyTorch 2.1.0 / NumPy 1.24.4 / Pillow 10.4.0.
- The underlying sensor replay completed fresh inference on 40 frames. Four shared historical anchors had road mask IoU >= 0.999659 and identical ground/blocked masks; original source inputs remained unchanged. These are the recorded experiment checks, not a second GPU benchmark of the publication copy.
- Verified the 40-frame GIF, publication hashes, local report links and representative rendered frames. Playback start, frame advance, slider, wraparound, speed and pause were checked with a JavaScript DOM/timer harness; this is not a full browser compatibility test.
- Public result JSON replaces machine paths with placeholders. Weights, raw sensor files, large array caches and per-frame input manifests are excluded.

The source packaging changes paths/cache defaults/font configuration only. No training or full validation benchmark was repeated for this release.

## Initial release

Release checks on 2026-11-25:

- CPU smoke checks passed: empty point cloud, ROI exclusion, road-head forward and backward.
- Runtime: Python 3.8.20 / PyTorch 2.1.0 / NumPy 1.24.4.
- Command: `python tools/smoke_test.py`.
- Tests executed from an isolated copy of the release source with CPU-only visibility, not the original working directory.
- All published Python sources compiled successfully without running training.
- Source/path screening excluded dataset payloads, weights, private server addresses and credentials.

These checks do not reproduce the historical dataset benchmark or a clean installation of the complete GPU training stack. Full training, geospatial data preparation and end-to-end inference remain dependent on the documented data and environment prerequisites.
