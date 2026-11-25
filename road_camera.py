"""Compare surround-camera, LiDAR, and fused inputs for BEV road segmentation."""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from torch import nn

from road import DATA, OUT, UNet, counts, gx, gy, limit, metric


WORK = OUT.parent / "road_fusion"
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
           "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
SPLITS = ("train", "dev", "val")
MODES = ("camera", "lidar", "fusion")


def extract_images(nusc):
    cache = WORK / "rgb"
    cache.mkdir(exist_ok=True)
    wanted = {}
    for split in ("train", "dev"):
        for token in np.load(OUT / (split + ".npz"))["token"]:
            lidar = nusc.get("sample_data", str(token))
            sample = nusc.get("sample", lidar["sample_token"])
            for name in CAMERAS:
                filename = nusc.get("sample_data", sample["data"][name])["filename"]
                dest = cache / Path(filename).name
                if not dest.exists() and not (DATA / filename).exists():
                    wanted[filename] = dest
    if wanted:
        proc = subprocess.Popen(["pigz", "-dc", str(DATA.parent / "v1.0-trainval01_blobs.tgz")],
                                stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|", bufsize=1024*1024) as archive:
                for entry in archive:
                    name = entry.name[2:] if entry.name.startswith("./") else entry.name
                    if name not in wanted:
                        continue
                    dest = wanted.pop(name)
                    temp = dest.with_suffix(".part")
                    temp.write_bytes(archive.extractfile(entry).read())
                    assert temp.stat().st_size == entry.size
                    temp.replace(dest)
                    if not wanted:
                        break
        finally:
            proc.stdout.close()
            proc.terminate()
            proc.wait()
        if wanted:
            raise RuntimeError(f"{len(wanted)} images missing after reading shard")
    print("Training images ready", flush=True)
    return cache


def pose(rec):
    return transform_matrix(rec["translation"], Quaternion(rec["rotation"]))


def prepare():
    cv2.setNumThreads(1)
    nusc = NuScenes("v1.0-trainval", str(DATA), verbose=False)
    cache = extract_images(nusc)
    ground = np.stack((gx, gy, np.zeros_like(gx), np.ones_like(gx))).reshape(4, -1)
    for split in SPLITS:
        tokens = np.load(OUT / (split + ".npz"))["token"]
        features = []
        for token in tokens:
            lidar = nusc.get("sample_data", str(token))
            sample = nusc.get("sample", lidar["sample_token"])
            world = pose(nusc.get("ego_pose", lidar["ego_pose_token"])) @ ground
            image_bev = np.zeros((*gx.shape, 4), np.uint8)
            best = np.zeros(gx.shape)
            for name in CAMERAS:
                sd = nusc.get("sample_data", sample["data"][name])
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                cam_pose = pose(nusc.get("ego_pose", sd["ego_pose_token"])) @ pose(cs)
                xyz = np.linalg.solve(cam_pose, world)[:3]
                uvh = np.array(cs["camera_intrinsic"]) @ xyz
                u, v = (uvh[:2] / np.maximum(uvh[2:], 1e-4)).reshape(2, *gx.shape)
                h, w = sd["height"], sd["width"]
                # In overlaps use the view closest to its optical axis.
                cosine = (xyz[2] / np.linalg.norm(xyz, axis=0)).reshape(gx.shape)
                use = (u>=1) & (u<w-1) & (v>=1) & (v<h-1) & (cosine>best)
                path = cache / Path(sd["filename"]).name
                if split=="val" or not path.exists():
                    path = DATA / sd["filename"]
                img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
                if img is None:
                    raise FileNotFoundError(path)
                pixels = cv2.remap(img, ((u+.5)*img.shape[1]/w-.5).astype("float32"),
                                   ((v+.5)*img.shape[0]/h-.5).astype("float32"),
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                image_bev[use,:3] = pixels[use,::-1]
                image_bev[use,3] = 255
                best[use] = cosine[use]
            features.append(image_bev.transpose(2,0,1))
        np.savez_compressed(WORK / (split + ".npz"), features=features, token=tokens)
        print("Projected", split, len(tokens), flush=True)


def train(mode):
    torch.manual_seed(17)
    sets = {}
    for split in SPLITS:
        data = np.load(OUT / (split + ".npz"))
        rgb = np.load(WORK / (split + ".npz"))
        assert np.array_equal(data["token"], rgb["token"])
        camera = rgb["features"].astype("float32")/255
        lidar = data["x"].astype("float32")
        if mode=="lidar":
            camera.fill(0)
        if mode=="camera":
            lidar.fill(0)
        xy = np.broadcast_to(np.stack((gx,gy))/limit, (len(lidar),2,*gx.shape))
        x = np.concatenate((lidar,camera,xy),axis=1).astype("float32")
        sets[split] = (torch.from_numpy(x).cuda(), torch.from_numpy(data["y"]).cuda(), data["scene"])
    model = UNet()
    model.a[0] = nn.Conv2d(10,16,3,padding=1)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    x,y,_ = sets["train"]
    best = -1
    for epoch in range(18):
        model.train()
        for idx in torch.randperm(len(x), device="cuda").split(20):
            pred = model(x[idx])[:,0]
            target = y[idx]
            valid = target!=255
            loss = F.binary_cross_entropy_with_logits(pred[valid],target[valid].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dx,dy,_ = sets["dev"]
            pred = torch.cat([model(t)[:,0].sigmoid() for t in dx.split(20)])
            for threshold in (.35,.4,.45,.5,.55,.6,.65):
                score = metric(counts(pred>=threshold,dy))["mIoU"]
                if score>best:
                    best = score
                    torch.save(dict(weights=model.state_dict(),epoch=epoch+1,threshold=threshold),
                               WORK / (mode+".pt"))
    ckpt = torch.load(WORK / (mode+".pt"))
    model.load_state_dict(ckpt["weights"])
    with torch.no_grad():
        vx,vy,scenes = sets["val"]
        masks = torch.cat([model(t)[:,0].sigmoid()>=ckpt["threshold"] for t in vx.split(20)])
        c = counts(masks,vy)
    report = dict(mode=mode, epoch=ckpt["epoch"], threshold=ckpt["threshold"],
                  dev_mIoU=best,val=metric(c))
    np.savez_compressed(WORK / (mode+"_pred.npz"),masks=masks.cpu().numpy(),scene=scenes,counts=c.cpu().numpy())
    (WORK / (mode+".json")).write_text(json.dumps(report,indent=2))
    print(report)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job",choices=("prepare",*MODES))
    job = parser.parse_args().job
    WORK.mkdir(exist_ok=True)
    if job=="prepare":
        prepare()
    else:
        train(job)
