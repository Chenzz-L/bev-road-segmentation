"""Compare LiDAR expansion and BEVFusion footprint completion on frozen frames."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from scipy.ndimage import distance_transform_edt
import torch

from road import DATA, CELL, limit, gx, gy, grid_points
from road_obstacle_check import OUT as PRIOR, VEC, ROI, occupancy, references
from road_full_validation import WORK
from road_scale_train import new_model


OUT = WORK / "obstacle_boxes"
DETECTIONS = DATA.parent / "BEVFusion/full_val/predictions/pred_instances_3d/results_nusc.json"
VEHICLES = {"car", "bus", "truck", "trailer", "construction_vehicle", "bicycle", "motorcycle"}
SCORE = 0.30


def footprint(detections, pose):
    mask = np.zeros_like(ROI)
    inv = Quaternion(pose["rotation"]).inverse
    for d in detections:
        if d["detection_name"] not in VEHICLES or d["detection_score"] < SCORE:
            continue
        box = Box(d["translation"], d["size"], Quaternion(d["rotation"]))
        box.translate(-np.asarray(pose["translation"]))
        box.rotate(inv)
        x, y = box.center[:2]
        width, length = box.wlh[:2]
        if max(abs(x), abs(y)) > limit + np.hypot(width, length)/2:
            continue
        z = box.corners()[2]
        if z.min() > 2.2 or z.max() < -0.5:
            continue
        yaw = box.orientation.yaw_pitch_roll[0]
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = gx-x, gy-y
        mask |= (abs(c*dx+s*dy) <= length/2) & (abs(-s*dx+c*dy) <= width/2)
    return mask & ROI


def draw(scene, road, vehicle, old, new, outside):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.45))
    for ax, m, title in zip(axes[:2], (old, new), ("LiDAR: expand one cell", "LiDAR hits + BEVFusion boxes")):
        img = np.zeros_like(road, dtype=np.uint8)
        img[road] = 1
        img[road & m] = 2
        ax.imshow(img.T, origin="lower", extent=(-limit,limit,-limit,limit), interpolation="nearest",
                  cmap=ListedColormap(["#1d2f3d","#768897","#ec8d64"]),vmin=0,vmax=2)
        ax.set_title(title, fontsize=11)
    delta = np.zeros_like(road,dtype=np.uint8)
    for idx, mask in enumerate((vehicle & new & ~old, outside & old & ~new,
                                vehicle & old & ~new, outside & new & ~old),1):
        delta[road & mask] = idx
    axes[2].imshow(delta.T,origin="lower",extent=(-limit,limit,-limit,limit),interpolation="nearest",
                   cmap=ListedColormap(["#1d2f3d","#55c5a4","#8dbaff","#ffce71","#ed8baa"]),vmin=0,vmax=4)
    axes[2].set_title("Changed cells on reference regions",fontsize=11)
    for ax in axes:
        ax.contour(gy.T,gx.T,vehicle.T,levels=[.5],colors="white",linewidths=.6)
        ax.invert_xaxis()
        ax.set_xlabel("Ego left (m)")
    axes[0].set_ylabel("Ego forward (m)")
    fig.suptitle(scene+"  |  white outlines: on-road vehicle reference")
    fig.text(.5,.075,"First two panels: orange = blocked on predicted road.   Right: green = additional vehicle coverage, blue = road released,",ha="center",fontsize=9)
    fig.text(.5,.03,"yellow = lost vehicle coverage, pink = newly blocked outside observed road cells.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.065,right=.985,bottom=.26,top=.81,wspace=.23)
    fig.savefig(OUT/"paired_example.png",dpi=170)
    plt.close(fig)


@torch.inference_mode()
def main(example_only=False):
    OUT.mkdir(exist_ok=True)
    prior = json.loads((PRIOR/"frames.json").read_text())
    prior_report = json.loads((PRIOR/"summary.json").read_text())
    if example_only:
        prior = [row for row in prior if row["lidar_token"] == prior_report["example"][1]]
        assert len(prior) == 1
    nusc = NuScenes("v1.0-trainval", str(DATA), verbose=False)
    with np.load(WORK/"val_full.npz") as d:
        index = {str(t):i for i,t in enumerate(d["token"])}
    labels = np.load(VEC/"full_y.npy",mmap_mode="r")
    raw_predictions = json.loads(DETECTIONS.read_bytes())
    meta = raw_predictions["meta"]
    predictions = raw_predictions["results"]
    cp = torch.load(VEC/"segformer_thin_17.pt",map_location="cpu")
    model = new_model(cp["model"]).cuda().eval()
    model.load_state_dict(cp["weights"])
    coords = np.stack([gx,gy])/limit
    totals, rows = Counter(), []
    for j, row in enumerate(prior):
        token = row["lidar_token"]
        sd = nusc.get("sample_data",token)
        sample = sd["sample_token"]
        dets = predictions[sample]
        assert all(d["sample_token"] == sample for d in dets)
        calib = nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
        raw = np.fromfile(DATA/sd["filename"],np.float32).reshape(-1,5)
        xyz = raw[:,:3] @ Quaternion(calib["rotation"]).rotation_matrix.T + calib["translation"]
        feat = grid_points(xyz,raw[:,3])
        inp = torch.from_numpy(np.concatenate([feat,coords]).astype(np.float32)[None]).cuda()
        road = model(inp)[0,0].sigmoid().cpu().numpy() >= cp["threshold"]
        direct, expanded, ground = occupancy(xyz,feat)
        boxes = footprint(dets,nusc.get("ego_pose",sd["ego_pose_token"]))
        vehicle, outside = references(nusc,sd,labels[index[token]] == 1)
        outside &= feat[0] > 0
        distance = distance_transform_edt(~vehicle,sampling=CELL)
        near = outside & (distance <= 2)
        _, outside_unpadded = references(nusc,sd,labels[index[token]] == 1,padding=0)
        # Direct four-neighbours of a vehicle reference cell, excluding every
        # annotated footprint but imposing no buffer around it.
        adjacent = outside_unpadded & (feat[0] > 0) & (distance <= CELL)
        refs = {"vehicle":vehicle,"outside":outside,"near":near,"adjacent":adjacent}
        methods = {"hits":direct,"expanded":expanded,"hits_boxes":direct|boxes,"expanded_boxes":expanded|boxes}
        stats = Counter()
        for region, ref in refs.items():
            eligible = ref & road
            stats[region+"_eligible"] = int(eligible.sum())
            for name, mask in methods.items():
                stats[region+"_"+name] = int((eligible & mask).sum())
                stats[region+"_"+name+"_unknown"] = int((eligible & ~mask & ~ground).sum())
                stats[region+"_"+name+"_free"] = int((eligible & ~mask & ground).sum())
            for key, mask in (("added",methods["hits_boxes"] & ~expanded),("removed",expanded & ~methods["hits_boxes"])):
                stats[region+"_"+key] = int((eligible & mask).sum())
        for region, old in (("vehicle","vehicle"),("outside","outside_observed")):
            for key, prev in (("eligible","baseline"),("hits","raw_veto"),("expanded","veto"),("expanded_unknown","unknown"),("expanded_free","free")):
                assert stats[region+"_"+key] == row[old+"_"+prev], (token,key)
        totals.update(stats)
        rows.append(dict(scene=row["scene"],lidar_token=token,sample_token=sample,**stats))
        if token == prior_report["example"][1]:
            draw(row["scene"],road,vehicle,expanded,methods["hits_boxes"],outside)
            if example_only:
                return
        if (j+1) % 150 == 0:
            print("Verified paired frames:",j+1,flush=True)
    report = dict(frames=len(rows),scenes=len({r["scene"] for r in rows}),score=SCORE,
                  classes=sorted(VEHICLES),seed=17,detector_meta=meta,example=prior_report["example"],
                  counts=totals)
    (OUT/"frames.json").write_text(json.dumps(rows))
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
