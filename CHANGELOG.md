# Release preparation

## 2026-11-27 — Continuous road replay and dynamic occupancy

- Published per-frame BEVFusion / CapacityHead inference and timestamp-aligned car occupancy replay, with a 40-frame / 19.50-second demonstration and observed ego trajectory.
- Added causal occupancy primitives, the locked no-training evaluation pipeline, independent verification, historical summaries and interactive viewers.
- Preserved the distinction between a single-anchor horizon viewer and actual continuous sensor frames, and between road segmentation and geometry-filtered release.
- Added configurable data, trajectory, capacity-head and font paths. Published results retain historical source hashes; machine paths are replaced with placeholders.
- Original experiment inputs and weights were not modified. Release checks do not rerun training or the full benchmark.

- Selected the project entry points and their local source dependencies from the research tree.
- Added Chinese README, source navigation, input requirements and test runners.
- Replaced machine-specific examples with generic paths and kept upstream notices.
- Kept the original algorithm and historical results unless an explicit fix is listed below.
- Original server projects were not modified. Full benchmark runs were not repeated.

- Added environment overrides for data, maps, work directory, BEVFusion project and feature cache.
