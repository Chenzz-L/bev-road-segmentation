from road import MAP_ROOT
"""Attribute paired fusion/LiDAR road-mask errors to the nuScenes walkway layer."""
import json
from collections import Counter, defaultdict

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.prepared import prep
from shapely.vectorized import contains

from road import ROOT, centers
from road_obstacle_check import ROI
from road_shared_features import OUT
from road_full_validation import WORK


class Layer:
    def __init__(self, polygons):
        polygons = [p for p in polygons if not p.is_empty]
        self.bounds = np.asarray([p.bounds for p in polygons])
        self.polygons = [prep(p) for p in polygons]

    def lookup(self, x, y):
        left, bottom, right, top = self.bounds.T
        nearby = (left <= x.max()) & (right >= x.min()) & (bottom <= y.max()) & (top >= y.min())
        inside = np.zeros(x.shape, bool)
        for i in np.flatnonzero(nearby):
            use = ~inside & (x >= left[i]) & (x <= right[i]) & (y >= bottom[i]) & (y <= top[i])
            if use.any():
                inside[use] = contains(self.polygons[i], x[use], y[use])
        return inside


def layers(region):
    api = NuScenesMap(dataroot=str(MAP_ROOT), map_name=region)
    road_tokens = dict.fromkeys(t for r in api.drivable_area for t in r["polygon_tokens"])
    roads = [api.extract_polygon(t) for t in road_tokens]
    walks = [api.extract_polygon(t) for t in dict.fromkeys(r["polygon_token"] for r in api.walkway)]
    return Layer(roads), Layer(walks), Layer([p.buffer(.5) for p in roads])


def main():
    rows = [r for r in json.loads((OUT / "manifest.json").read_text()) if r["split"] == "val"]
    tokens = [r["token"] for r in rows]
    def arrays(file, keys):
        with np.load(OUT / file) as data:
            assert list(data["tokens"]) == tokens
            return [data[k] for k in keys]
    fusion, = arrays("fusion_masks.npz", ["masks"])
    lidar, = arrays("lidar_masks.npz", ["masks"])
    ground, blocked, labels = arrays("references.npz", ["ground", "blocked", "label"])
    with np.load(WORK / "val_full.npz") as d:
        regions = dict(zip(d["token"], d["region"]))
    poses = json.loads((WORK / "vector_labels/poses.json").read_text())
    examples = {c["token"]: c for c in json.loads((OUT / "visible_surface/cases.json").read_text())
                if c["id"] in ("P19", "P59")}
    case_results = {}
    maps = {}
    groups = defaultdict(lambda: defaultdict(Counter))
    for i, r in enumerate(rows):
        token = r["token"]
        region = regions[token]
        if region not in maps:
            maps[region] = layers(region)
        roads, walks, buffered = maps[region]
        tx, ty, yaw = poses[token]
        c, s = np.cos(yaw), np.sin(yaw)
        x = centers[:, :, 0] * c - centers[:, :, 1] * s + tx
        y = centers[:, :, 0] * s + centers[:, :, 1] * c + ty
        valid = labels[i] != 255
        assert np.array_equal(roads.lookup(x, y)[valid], labels[i][valid] == 1)
        walk, near_road = walks.lookup(x, y), buffered.lookup(x, y)
        add = fusion[i] & ~lidar[i] & (labels[i] == 0)
        remove = lidar[i] & ~fusion[i] & (labels[i] == 0)
        free = ground[i] & ~blocked[i]
        masks = dict(added_free=add & free, removed_free=remove & free, added_raw=add,
                     added_unsampled=add & ROI & ~ground[i] & ~blocked[i],
                     lidar_free=lidar[i] & free & (labels[i] == 0),
                     fusion_free=fusion[i] & free & (labels[i] == 0))
        for k, mask in masks.items():
            groups[k][r["scene"]].update(total=int(mask.sum()), walkway=int((mask & walk).sum()),
                                        walkway_within_half_m_of_road=int((mask & walk & near_road).sum()))
        if token in examples:
            ex = examples[token]
            point = (ex["row"], ex["col"])
            case_results[ex["id"]] = dict(map_drivable=int(labels[i][point]), walkway=bool(walk[point]),
                                          added_raw=bool(add[point]))
    summary = {}
    for k, scenes in groups.items():
        counts = sum(scenes.values(), Counter())
        pairs = np.array([[v["walkway"], v["total"]] for v in scenes.values()])
        draws = np.random.default_rng(17).multinomial(len(pairs), np.full(len(pairs), 1/len(pairs)), size=4000)
        boot = draws @ pairs
        summary[k] = dict(**counts, walkway_pct=100*counts["walkway"]/counts["total"],
                          scenes_with_walkway=int((pairs[:, 0] > 0).sum()),
                          top5_scene_walkway=int(np.sort(pairs[:, 0])[-5:].sum()),
                          scene_bootstrap_pct=np.percentile(100*boot[:, 0]/boot[:, 1], [2.5, 97.5]).tolist())
    before = json.loads((OUT / "summary.json").read_text())["counts"]["fusion"]
    assert summary["added_free"]["total"] == before["added_offroad_vs_lidar"]
    assert summary["removed_free"]["total"] == before["removed_offroad_vs_lidar"]
    for k in ("total", "walkway"):
        assert summary["added_free"][k]-summary["removed_free"][k] == summary["fusion_free"][k]-summary["lidar_free"][k]
    out = dict(frames=len(rows), scenes=len(groups["added_free"]), groups=summary, examples=case_results)
    (OUT / "walkway_errors.json").write_text(json.dumps(dict(**out, by_scene=groups), indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
