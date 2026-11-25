"""Test direct surround RGB on a frozen LiDAR road model with identical obstacle masks."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.ndimage import binary_dilation
import torch
from torch import nn
from torch.nn import functional as F

from road import DATA, OUT as ROAD, gx, gy, limit
from road_camera import WORK as CAMERA
from road_full_validation import WORK
from road_scale_data import DEST, SHARDS
from road_scale_train import new_model
from road_obstacle_check import VEC, ROI, occupancy, references
from road_obstacle_boxes import DETECTIONS, OUT as BOX_RUN, footprint


OUT = WORK / "direct_rgb"


def vector_targets(split,tokens):
    if split == "train":
        paths = [DEST/f"part{s}.npz" for s in SHARDS]
        label = "train"
    elif split == "val":
        paths, label = [WORK/"val_full.npz"], "full"
    else:
        paths, label = [ROAD/"dev.npz"], "dev"
    source = []
    for path in paths:
        with np.load(path) as d:
            source.extend(d["token"].tolist())
    lookup = {t:i for i,t in enumerate(source)}
    labels = np.load(VEC/(label+"_y.npy"),mmap_mode="r")
    assert len(labels) == len(source) == len(lookup)
    return labels[[lookup[t] for t in tokens]]


def load(split,network):
    with np.load(ROAD/f"{split}.npz") as d, np.load(CAMERA/f"{split}.npz") as r:
        tokens, scenes, x = d["token"], d["scene"], d["x"]
        assert np.array_equal(tokens,r["token"])
        rgb = torch.from_numpy(r["features"]).cuda().float()/255
    xy = np.broadcast_to(np.stack([gx,gy])/limit,(len(x),2,*gx.shape))
    inp = torch.from_numpy(np.concatenate([x,xy],axis=1).astype(np.float32)).cuda()
    with torch.no_grad():
        logits = torch.cat([network(batch) for batch in inp.split(1 if split=="val" else 16)])
    y = torch.from_numpy(vector_targets(split,tokens)).cuda()
    return dict(logits=logits,rgb=rgb,y=y,tokens=tokens,scenes=scenes,x=x)


def features(data,use_rgb):
    rgb = data["rgb"].clone()
    if not use_rgb:
        rgb[:,:3] = 0
    return torch.cat([data["logits"].clamp(-8,8),rgb],dim=1)


def corrected(head, features, logits):
    return logits + 1.5*head(features).tanh()


def fit(train,dev,use_rgb,threshold):
    torch.manual_seed(17)
    head = nn.Sequential(nn.Conv2d(5,16,3,padding=1),nn.ReLU(),
                         nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1)).cuda()
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    opt = torch.optim.AdamW(head.parameters(),lr=.001)
    xt, xd = features(train,use_rgb), features(dev,use_rgb)
    best = -1
    for epoch in range(1,13):
        for idx in torch.randperm(len(xt),device="cuda").split(24):
            l = corrected(head,xt[idx],train["logits"][idx])[:,0]
            y = train["y"][idx]
            valid = y != 255
            loss = F.binary_cross_entropy_with_logits(l[valid],y[valid].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = corrected(head,xd,dev["logits"])[:,0].sigmoid() >= threshold
            y = dev["y"]
            score = ((p & (y==1)).sum()/(((p | (y==1)) & (y!=255)).sum())).item()
            if score > best:
                best, selected = score, epoch
                weights = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(weights)
    torch.save(dict(weights=weights,epoch=selected,dev_road_iou=best),OUT/("rgb.pt" if use_rgb else "control.pt"))
    return head,selected


def show(rgb,baseline,enhanced,gt,blocked,ground,example):
    fig,axes = plt.subplots(1,4,figsize=(12.8,3.8))
    delta = np.full((*gx.shape,3),[.16,.21,.26])
    changed = (baseline!=enhanced) & (gt!=255)
    delta[changed & (enhanced==(gt==1))] = [.25,.84,.63]
    delta[changed & (enhanced!=(gt==1))] = [.95,.48,.37]
    def state(mask):
        im = np.full((*gx.shape,3),[.16,.21,.26])
        im[mask & ROI] = [.46,.55,.63]
        im[mask & ground] = [.51,.81,.69]
        im[mask & blocked] = [.95,.60,.37]
        return im
    for ax,im,title in zip(axes,[rgb,state(baseline),state(enhanced),delta],
            ["Six cameras on ground plane","No RGB control","With direct RGB","Road changes vs map"]):
        ax.imshow(im.transpose(1,0,2),origin="lower",interpolation="nearest")
        ax.invert_xaxis()
        ax.set(xticks=[],yticks=[],title=title)
    fig.suptitle(example+"  |  fixed BEVFusion detections and LiDAR occupancy")
    fig.text(.5,.06,"Middle: orange = blocked, green = observed free, grey = unknown on predicted road.   Right: green = corrected, coral = introduced error.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.02,right=.99,top=.84,bottom=.16,wspace=.06)
    fig.savefig(OUT/"paired.png",dpi=155)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    cp = torch.load(VEC/"segformer_thin_17.pt",map_location="cpu")
    threshold = cp["threshold"]
    net = new_model(cp["model"]).cuda().eval()
    net.load_state_dict(cp["weights"])
    train,dev,val = [load(split,net) for split in ("train","dev","val")]
    assert all(set(a["scenes"]).isdisjoint(b["scenes"]) for a,b in [(train,dev),(dev,val),(train,val)])
    models = {"original":(val["logits"][:,0].sigmoid()>=threshold).cpu().numpy()}
    epochs = {}
    for name,use in (("control",False),("rgb",True)):
        head,epochs[name] = fit(train,dev,use,threshold)
        with torch.no_grad():
            x = features(val,use)
            masks = [corrected(head,x[i:i+24],val["logits"][i:i+24])[:,0].sigmoid()>=threshold
                     for i in range(0,len(x),24)]
            models[name] = torch.cat(masks).cpu().numpy()
        print(name,epochs[name],flush=True)
    prior = json.loads((BOX_RUN/"frames.json").read_text())
    example = json.loads((BOX_RUN/"summary.json").read_text())["example"]
    predictions = json.loads(DETECTIONS.read_bytes())["results"]
    nusc = NuScenes("v1.0-trainval",str(DATA),verbose=False)
    lookup = {str(t):i for i,t in enumerate(val["tokens"])}
    by_model = {m:Counter() for m in models}
    rows = []
    for row in prior:
        token = row["lidar_token"]
        i = lookup[token]
        sd = nusc.get("sample_data",token)
        calib = nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
        raw = np.fromfile(DATA/sd["filename"],np.float32).reshape(-1,5)
        xyz = raw[:,:3] @ Quaternion(calib["rotation"]).rotation_matrix.T + calib["translation"]
        direct, _, ground = occupancy(xyz,val["x"][i])
        blocked = direct | footprint(predictions[sd["sample_token"]],nusc.get("ego_pose",sd["ego_pose_token"]))
        y = val["y"][i].cpu().numpy()
        vehicle,outside = references(nusc,sd,y==1)
        outside &= val["x"][i,0]>0
        _,outside0 = references(nusc,sd,y==1,padding=0)
        adj = binary_dilation(vehicle) & outside0 & (val["x"][i,0]>0)
        per = {}
        for name,masks in models.items():
            m = masks[i]
            free = m & ~blocked & ground
            stats = dict(road_tp=int((m & (y==1)).sum()),road_union=int(((m | (y==1)) & (y!=255)).sum()),
                         outside_n=int(outside.sum()),outside_free=int((free & outside).sum()),
                         vehicle_n=int(vehicle.sum()),vehicle_free=int((free & vehicle).sum()),
                         adjacent_n=int(adj.sum()),adjacent_blocked=int((m & blocked & adj).sum()),
                         adjacent_free=int((free & adj).sum()),
                         offroad_free=int((free & (y==0)).sum()))
            if name=="original":
                assert stats["outside_free"]==row["outside_hits_boxes_free"] and stats["vehicle_free"]==row["vehicle_hits_boxes_free"]
                assert stats["adjacent_blocked"]==row["adjacent_hits_boxes"]
            per[name]=stats
            by_model[name].update(stats)
        rows.append(dict(scene=row["scene"],token=token,models=per))
        if token==example[1]:
            show(val["rgb"][i,:3].cpu().numpy().transpose(1,2,0),models["control"][i],models["rgb"][i],
                 y,blocked,ground,example[0])
    report = dict(frames=len(rows),scenes=len(set(r["scene"] for r in rows)),train=len(train["x"]),dev=len(dev["x"]),
                  selected_epochs=epochs,threshold=threshold,counts=by_model)
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    (OUT/"frames.json").write_text(json.dumps(rows))
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
