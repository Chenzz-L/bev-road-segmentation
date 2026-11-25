"""Paired single-sweep occupancy check on four fixed keyframes per nuScenes val scene."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.ndimage import binary_dilation
import torch

from road import DATA, N, CELL, limit, centers, gx, gy, grid_points
from road_full_validation import WORK
from road_scale_train import new_model


VEC = WORK / "vector_labels"
OUT = WORK / "obstacle_check"
# Exclude the roof sensor's blind area and the ego vehicle in all reference regions.
ROI = ~((abs(gx) < 3.0) & (abs(gy) < 1.5))


def occupancy(xyz, features):
    # Fit local road grade to low returns. Start near ego-ground height, then
    # repeatedly fit only cells within 25 cm of the provisional ground plane.
    seen = features[0] > 0
    low = features[1].astype(float) * 3
    design = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    plane = np.zeros(3)
    for tolerance in (0.55, 0.25, 0.25):
        keep = (seen & ROI).ravel() & (abs(low.ravel() - (design @ plane)) < tolerance)
        if keep.sum() >= 30:
            plane = np.linalg.lstsq(design[keep], low.ravel()[keep], rcond=None)[0]
    ij = np.floor((xyz[:, :2] + limit) / CELL).astype(int)
    inside = ((ij >= 0) & (ij < N)).all(1)
    xy, z = xyz[inside, :2], xyz[inside, 2]
    col, row = ij[inside].T
    h = z - xy @ plane[:2] - plane[2]
    def raster(mask):
        m = np.zeros((N, N), bool)
        m[row[mask], col[mask]] = True
        return m & ROI
    elevated = raster((h >= 0.35) & (h <= 2.2))
    return elevated, binary_dilation(elevated) & ROI, raster(abs(h) <= 0.20)


def references(nusc, sd, road, padding=CELL):
    # LIDAR_TOP keyframes and their annotations share the sample timestamp
    # convention of the devkit. Transform global boxes into the LiDAR-time ego frame.
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    inverse = Quaternion(pose["rotation"]).inverse
    vehicle = np.zeros_like(road)
    padded = np.zeros_like(road)
    for box in nusc.get_boxes(sd["token"]):
        box.translate(-np.asarray(pose["translation"]))
        box.rotate(inverse)
        corners = box.corners()
        if corners[2].min() > 2.2 or corners[2].max() < -0.5:
            continue
        x, y = box.center[:2]
        width, length = box.wlh[:2]
        if max(abs(x), abs(y)) > limit + np.hypot(width, length)/2 + 1:
            continue
        yaw = box.orientation.yaw_pitch_roll[0]
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = gx-x, gy-y
        along, across = abs(c*dx+s*dy), abs(-s*dx+c*dy)
        footprint = (along <= length/2) & (across <= width/2)
        padded |= (along <= length/2 + padding) & (across <= width/2 + padding)
        if box.name.startswith("vehicle."):
            vehicle |= footprint
    return vehicle & road & ROI, road & ~padded & ROI


def plot(case, path):
    scene, token, seen, pred, vehicle, blocked, ground, free_ref = case
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.65))
    # Rows are ego-y, columns ego-x. Transpose so forward is up and left is left.
    state = np.zeros(pred.shape, np.uint8)
    state[pred] = 1
    state[pred & ground] = 2
    state[pred & blocked] = 3
    error = np.zeros(pred.shape, np.uint8)
    error[free_ref & pred & blocked] = 1
    error[vehicle & pred & ~blocked] = 2
    error[vehicle & pred & blocked] = 3
    for ax, im, cmap, title in zip(axes, [pred, state, error], [
            ["#263949", "#e1c67b"], ["#263949","#72849a","#70c9a7","#e47755"],
            ["#263949","#bc85cf","#f3cb74","#5bc8b8"]], [
            "Road prediction + vehicle reference", "Occupancy and visibility on predicted road",
            "Where the obstacle veto acts"]):
        ax.imshow(im.T, origin="lower", extent=(-limit,limit,-limit,limit),
                  cmap=ListedColormap(cmap), vmin=0, vmax=len(cmap)-1, interpolation="nearest")
        if vehicle.any():
            ax.contour(gy.T, gx.T, vehicle.T, levels=[0.5], colors="white", linewidths=.65)
        ax.invert_xaxis()
        ax.set_xlabel("Ego left (m)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Ego forward (m)")
    fig.suptitle(f"{scene}  |  {token[:12]}  |  white outlines: annotated on-road vehicle cells")
    fig.text(.5,.08,"Left: gold = predicted road.   Middle: green = observed free, orange = blocked, grey-blue = unknown.",ha="center",fontsize=9)
    fig.text(.5,.035,"Right: teal = vehicle blocked, yellow = vehicle missed by veto, purple = potential false closure on observed road.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.94))
    fig.savefig(path,dpi=160)
    plt.close(fig)


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    nusc = NuScenes("v1.0-trainval", str(DATA), verbose=False)
    labels = np.load(VEC/"full_y.npy", mmap_mode="r")
    with np.load(WORK/"val_full.npz") as d:
        scenes, tokens = d["scene"], d["token"]
    ids = []
    for scene in np.unique(scenes):
        group = np.flatnonzero(scenes == scene)
        ids.extend(group[np.linspace(0,len(group)-1,4,dtype=int)])
    assert len(set(ids)) == 600 and len(labels) == len(tokens) == 6019
    cp = torch.load(VEC/"segformer_thin_17.pt",map_location="cpu")
    model = new_model(cp["model"]).eval().cuda()
    model.load_state_dict(cp["weights"])
    coords = np.stack([gx, gy])/limit
    counts, rows, examples = Counter(), [], []
    for i in ids:
        token, scene = str(tokens[i]), str(scenes[i])
        sd = nusc.get("sample_data",token)
        assert nusc.get("scene",nusc.get("sample",sd["sample_token"])["scene_token"])["name"] == scene
        calib = nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
        raw = np.fromfile(DATA/sd["filename"],np.float32).reshape(-1,5)
        xyz = raw[:,:3] @ Quaternion(calib["rotation"]).rotation_matrix.T + calib["translation"]
        feat = grid_points(xyz, raw[:,3])
        inp = np.concatenate([feat,coords]).astype(np.float32)
        pred = model(torch.from_numpy(inp[None]).cuda())[0,0].sigmoid().cpu().numpy() >= cp["threshold"]
        direct, blocked, ground = occupancy(xyz, feat)
        vehicle, outside = references(nusc, sd, labels[i] == 1)
        seen = feat[0] > 0
        free_ref = outside & seen  # proxy only: unannotated obstacles may be present
        if not vehicle.any():
            continue
        unknown = pred & ~blocked & ~ground
        free = pred & ~blocked & ground
        stats = {}
        for key, ref in (("vehicle",vehicle),("outside_observed",free_ref),("outside_ground",outside & ground)):
            for label, mask in (("total",np.ones_like(pred)),("baseline",pred),("raw_veto",pred & direct),
                                ("veto",pred & blocked),("unknown",unknown),("free",free)):
                stats[key+"_"+label] = int((ref & mask).sum())
            assert stats[key+"_baseline"] == sum(stats[key+"_"+k] for k in ("veto","unknown","free"))
        counts.update(stats)
        rows.append(dict(scene=scene,lidar_token=token,**stats))
        if stats["vehicle_baseline"] >= 30:
            examples.append((stats["vehicle_veto"]/stats["vehicle_baseline"],
                             (scene,token,seen,pred,vehicle,blocked,ground,free_ref)))
    frac = counts["vehicle_veto"]/counts["vehicle_baseline"]
    _, example = min(examples,key=lambda item:abs(item[0]-frac))
    plot(example, OUT/"example.png")
    report = dict(sampled=len(ids),sampled_scenes=len(set(scenes[ids])),with_vehicles=len(rows),
                  with_vehicle_scenes=len({r["scene"] for r in rows}),seed=17,
                  road_threshold=float(cp["threshold"]),example=example[:2],counts=counts)
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    (OUT/"frames.json").write_text(json.dumps(rows))
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
