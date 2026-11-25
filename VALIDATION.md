# Validation

Release checks on 2026-11-25:

- CPU smoke checks passed: empty point cloud, ROI exclusion, road-head forward and backward.
- Runtime: Python 3.8.20 / PyTorch 2.1.0 / NumPy 1.24.4.
- Command: `python tools/smoke_test.py`.
- Tests executed from an isolated copy of the release source with CPU-only visibility, not the original working directory.
- All published Python sources compiled successfully without running training.
- Source/path screening excluded dataset payloads, weights, private server addresses and credentials.

These checks do not reproduce the historical dataset benchmark or a clean installation of the complete GPU training stack. Full training, geospatial data preparation and end-to-end inference remain dependent on the documented data and environment prerequisites.
