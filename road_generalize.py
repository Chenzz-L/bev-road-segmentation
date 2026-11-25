"""Matched-budget LiDAR road segmentation experiments."""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from audit_road_density import boundary_band
from road import OUT, UNet, counts, gx, gy, limit, metric
from road_scene_data import SHARDS, WORK, digest


KEYS = ("x","y","scene","region","token")
POLICIES = ("original","spread","full")


def read(path):
    with np.load(path) as src:
        return {k:src[k] for k in KEYS}


def subset(data, idx):
    return {k:v[idx] for k,v in data.items()}


def merge(segments):
    return {k:np.concatenate([d[k] for d in segments]) for k in KEYS}


def build_splits():
    train = read(OUT/"train.npz")
    extra,fresh = [],[]
    for shard in SHARDS:
        path = WORK/("shard"+shard+".npz")
        with np.load(path) as archive:
            d = {k:archive[k] for k in KEYS}
            for split,items in (("extra",extra),("fresh",fresh)):
                keep = archive["split"]==split
                if np.any(keep):
                    items.append(subset(d,keep))
    full = merge([train]+extra)
    confirmation = merge(fresh)
    scene_table = json.loads((OUT.parent.parent/"nuscenes_trainval_val/v1.0-trainval/scene.json").read_text())
    logs = {s["name"]:s["log_token"] for s in scene_table}
    def logset(data):
        return {logs[s] for s in set(data["scene"])}
    for existing in (full,read(OUT/"dev.npz"),read(OUT/"val.npz")):
        assert not logset(confirmation)&logset(existing)
    assert len(set(full["token"]))==len(full["token"])
    # At each round, take one frame from each scene before repeating scenes, and
    # interleave regions to avoid a single region consuming a partial round.
    streams = {}
    for region in sorted(set(full["region"])):
        names = sorted(set(full["scene"][full["region"]==region]),key=digest)
        scene_frames = [sorted(np.flatnonzero(full["scene"]==s),key=lambda i:digest(full["token"][i])) for s in names]
        streams[region] = [indices[j] for j in range(max(map(len,scene_frames))) for indices in scene_frames if j<len(indices)]
    chosen = [v[j] for j in range(max(map(len,streams.values()))) for v in streams.values() if j<len(v)][:len(train["x"])]
    spread = subset(full,chosen)
    for name,d in (("spread",spread),("full",full),("fresh",confirmation)):
        np.savez_compressed(WORK/(name+".npz"),**d)
        print(name,len(d["scene"]),len(set(d["scene"])),len(logset(d)),
              {r:len(set(d["scene"][d["region"]==r])) for r in set(d["region"])},flush=True)


def tensors(d):
    x = torch.from_numpy(d["x"]).cuda().float()
    xy = torch.from_numpy(np.stack((gx,gy))/limit).cuda().float().expand(len(x),-1,-1,-1)
    return torch.cat((x,xy),1),torch.from_numpy(d["y"]).cuda()


def network(context):
    model = UNet()
    if context:
        # Broaden the receptive field at the coarsest scale; preserve weights,
        # parameter count, spatial output and the original decoder.
        for layer,rate in ((model.c[0],2),(model.c[3],3)):
            layer.dilation = (rate,rate)
            layer.padding = (rate,rate)
    return model


def fitting(policy,variant,seed,model_factory=None,tag_prefix=""):
    torch.manual_seed(seed)
    old = policy=="original"
    source = OUT/"train.npz" if old else WORK/(policy+".npz")
    d = read(source)
    edge = variant=="edge"
    weights = torch.from_numpy(boundary_band(d["y"])).cuda() if edge else None
    x,y = tensors(d)
    dx,dy = tensors(read(OUT/"dev.npz"))
    model = (network(variant=="context") if model_factory is None else model_factory()).cuda()
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    tag = f"{tag_prefix}{policy}_{variant}_{seed}"
    best = -1
    ids = torch.empty(0,device="cuda",dtype=torch.long)
    for epoch in range(18):
        while len(ids)<624:
            ids = torch.cat((ids,torch.randperm(len(x),device="cuda")))
        training,ids = ids[:624],ids[624:]
        model.train()
        for idx in training.split(20):
            logits = model(x[idx])[:,0]
            target = y[idx]
            valid = target!=255
            loss = F.binary_cross_entropy_with_logits(logits[valid],target[valid].float(),reduction="none")
            if edge:
                w = 1 + 2 * weights[idx][valid].float()
                loss = (loss*w).sum()/w.sum()
            else:
                loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.cat([model(t)[:,0].sigmoid() for t in dx.split(20)])
            for threshold in (.35,.4,.45,.5,.55,.6,.65):
                score = metric(counts(probs>=threshold,dy))["mIoU"]
                if score>best:
                    best=score
                    torch.save(dict(weights=model.state_dict(),threshold=threshold,dev=best,epoch=epoch+1),WORK/(tag+".pt"))
    print(tag,"dev",best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("policy",choices=("prepare",*POLICIES))
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--edge",action="store_true")
    p.add_argument("--context",action="store_true")
    a=p.parse_args()
    if a.policy=="prepare":
        build_splits()
    else:
        assert not (a.edge and a.context)
        fitting(a.policy,"edge" if a.edge else "context" if a.context else "plain",a.seed)
