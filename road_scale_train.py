"""Full-frame, matched training schedule for U-Net and SegFormer-B0."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from audit_road_density import boundary_band
from road import OUT, UNet, counts, gx, gy, limit, metric
from road_generalize import read, tensors
from road_scale_data import DEST, SHARDS
from road_segformer import RoadSegFormer


BATCH = 128
EPOCHS = 18
CACHE = Path("/dev/shm") / ("road-scale-19664-"+str(os.getuid()))


def cache_data():
    inventory = json.loads((DEST/"inventory.json").read_text())
    n = inventory["frames"]
    CACHE.mkdir(mode=0o700,exist_ok=True)
    arrays = {}
    for key,dtype,tail in (("x",np.float16,(4,96,96)),("y",np.uint8,(96,96)),("edge",np.bool_,(96,96))):
        arrays[key] = np.lib.format.open_memmap(CACHE/(key+".npy"),mode="w+",dtype=dtype,shape=(n,*tail))
    start=0
    for shard in SHARDS:
        with np.load(DEST/("part"+shard+".npz")) as d:
            y=d["y"]
            end=start+len(y)
            arrays["x"][start:end]=d["x"]
            arrays["y"][start:end]=y
            arrays["edge"][start:end]=boundary_band(y)
            start=end
    assert start==n
    for a in arrays.values():a.flush()
    print("Ready",n,flush=True)


def new_model(name):
    return UNet() if name=="unet" else RoadSegFormer()


def fit(name,loss_name,seed,small_control=False,vector_labels=False):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    # Keep the same sample order for both architectures and both losses.
    order_rng=torch.Generator(device="cpu").manual_seed(seed)
    output_dir=DEST
    label_cache=CACHE
    if vector_labels:
        from road_full_validation import WORK as refinement
        output_dir=refinement/"vector_labels"
        label_cache=output_dir
        assert not small_control
    if small_control:
        from road_scene_data import WORK
        d=read(WORK/"full.npz")
        x=torch.from_numpy(d["x"]).cuda()
        y=torch.from_numpy(d["y"]).cuda()
        edge=torch.from_numpy(boundary_band(d["y"])).cuda() if loss_name=="edge" else None
    else:
        x=torch.from_numpy(np.load(CACHE/"x.npy",mmap_mode="c")).cuda()
        label_file=label_cache/("train_y.npy" if vector_labels else "y.npy")
        edge_file=label_cache/("train_edge.npy" if vector_labels else "edge.npy")
        y=torch.from_numpy(np.load(label_file,mmap_mode="c")).cuda()
        edge=torch.from_numpy(np.load(edge_file,mmap_mode="c")).cuda() if loss_name in ("edge","thin") else None
    thin=None
    if loss_name=="thin":
        assert vector_labels
        thin=torch.from_numpy(np.load(label_cache/"train_thin.npy",mmap_mode="c")).cuda()
        n0=int((thin & (y==0)).sum());n1=int((thin & (y==1)).sum())
        w0,w1=2*(n0+n1)/n0,2*(n0+n1)/n1
    xy=torch.from_numpy(np.stack((gx,gy))/limit).cuda().half().unsqueeze(0)
    dx,dy=tensors(read(OUT/"dev.npz"))
    if vector_labels:
        dy=torch.from_numpy(np.load(label_cache/"dev_y.npy")).cuda()
    dx=dx.contiguous(memory_format=torch.channels_last)
    model=new_model(name).cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01,fused=True)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024)
    tag=("small_control_" if small_control else "")+f"{name}_{loss_name}_{seed}"
    epoch_size=19664
    steps=math.ceil(epoch_size/BATCH)
    total=steps*EPOCHS
    best=-1.
    history=[]
    skipped=0
    started=time.monotonic()
    unused=torch.empty(0,dtype=torch.long)
    for epoch in range(EPOCHS):
        model.train()
        losses=torch.zeros((),device="cuda")
        while len(unused)<epoch_size:
            unused=torch.cat((unused,torch.randperm(len(x),generator=order_rng)))
        perm,unused=unused[:epoch_size].cuda(),unused[epoch_size:]
        for j,ix in enumerate(perm.split(BATCH)):
            step=epoch*steps+j+1
            warmup=steps
            factor=step/warmup if step<=warmup else .05+.95*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))
            opt.param_groups[0]["lr"]=.001*factor
            features=torch.cat([x[ix],xy.expand(len(ix),-1,-1,-1)],1).contiguous(memory_format=torch.channels_last)
            target=y[ix]
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(features)[:,0]
                per_pixel=F.binary_cross_entropy_with_logits(logits,(target==1).float(),reduction="none")
                w=(target!=255).float()
                if edge is not None:w=w*(1.+2.*edge[ix])
                if thin is not None:w=w+thin[ix]*torch.where(target==1,w1,w0)
                loss=(w*per_pixel).sum()/w.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            prev=scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped+=int(scaler.get_scale()<prev)
            losses+=loss.detach()
        model.eval()
        with torch.no_grad():
            prob=torch.cat([model(t)[:,0].sigmoid() for t in dx.split(32)])
            opts=[(metric(counts(prob>=th,dy))["mIoU"],th) for th in (.35,.4,.45,.5,.55,.6,.65)]
            score,threshold=max(opts,key=lambda pair:pair[0])
        row=dict(epoch=epoch+1,train_loss=losses.item()/steps,dev_mIoU=score,threshold=threshold,
                 lr=opt.param_groups[0]["lr"],skipped_updates=skipped)
        assert np.isfinite(row["train_loss"]) and score>0
        history.append(row)
        if score>best:
            best=score
            checkpoint=output_dir/(tag+".pt")
            temp=checkpoint.with_suffix(".tmp")
            torch.save(dict(model=name,loss=loss_name,seed=seed,weights=model.state_dict(),
                            threshold=threshold,epoch=epoch+1,dev=score),temp)
            temp.replace(checkpoint)
        print(tag,epoch+1,round(row["train_loss"],4),round(score,4),round(time.monotonic()-started,1),flush=True)
        (output_dir/(tag+"_curve.json")).write_text(json.dumps(history,indent=2))
    print("Done",tag,total,"skipped",skipped,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",choices=("cache","unet","segformer"))
    p.add_argument("--loss",choices=("plain","edge","thin"),default="plain")
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--small-control",action="store_true")
    p.add_argument("--vector-labels",action="store_true")
    a=p.parse_args()
    if a.model=="cache":
        cache_data()
    else:
        fit(a.model,a.loss,a.seed,a.small_control,a.vector_labels)
