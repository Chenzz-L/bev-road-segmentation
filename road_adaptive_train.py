"""Controlled equal/adaptive modality fusion against the frozen 200-scene head."""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from road_adaptive_data import DEST, MODALITIES, manifests
from road_shared_features import OUT, CACHE
from road_200_data import DEST as TRAIN200
from road_shared_heads import RoadHead
from road_obstacle_check import ROI


class AdaptiveHead(nn.Module):
    def __init__(self, adaptive):
        super().__init__()
        def project(n):return nn.Sequential(nn.Conv2d(n,16,1),nn.GroupNorm(4,16),nn.ReLU())
        self.camera=project(80);self.lidar=project(256)
        self.gate=nn.Sequential(nn.Conv2d(40,16,1),nn.ReLU(),nn.Conv2d(16,2,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        self.residual=RoadHead(56)
        nn.init.zeros_(self.residual.output.weight);nn.init.zeros_(self.residual.output.bias)
        self.adaptive=adaptive

    def forward(self, modal, geometry, base_small, base_hidden, original, force_equal=False):
        camera=self.camera(modal[:,:80]);lidar=self.lidar(modal[:,80:])
        g=self.gate(torch.cat([camera,lidar,geometry,base_small.sigmoid()],1)).softmax(1)
        if not self.adaptive or force_equal:g=torch.full_like(g,.5)
        mixed=g[:,:1]*camera+g[:,1:]*lidar
        z=self.residual(torch.cat([mixed,geometry,base_small,base_hidden],1))
        return original+1.5*z.tanh(),g[:,0]


def load_split(split,base):
    rows=manifests()[split]
    with np.load(DEST/f"{split}_geometry.npz") as ref:
        assert list(ref["tokens"])==[r["token"] for r in rows]
        d={k:ref[k] for k in ("y","ground","blocked","walk")}
        d["geometry"]=torch.tensor(ref["geometry"],device="cuda")
    d["target"]=torch.tensor(d["y"],device="cuda")
    n=len(rows)
    d["fused"]=torch.empty((n,256,64,64),device="cuda",dtype=torch.float16)
    d["modal"]=torch.empty((n,336,64,64),device="cuda",dtype=torch.float16)
    for i,r in enumerate(rows):
        token=r["token"];old=CACHE/(token+".npy")
        if old.exists():
            arr=np.load(old,mmap_mode="r")
            d["modal"][i]=torch.from_numpy(arr[:336].copy())
            d["fused"][i]=torch.from_numpy(arr[336:592].copy())
        else:
            folder="eval27_features" if split=="reserved" else "features"
            d["modal"][i]=torch.from_numpy(np.load(MODALITIES/(token+".npy")))
            d["fused"][i]=torch.from_numpy(np.load(TRAIN200/folder/(token+".npy")))
    with torch.no_grad():
        hidden=[];original=[]
        for x in d["fused"].split(24):
            x=x.float().contiguous(memory_format=torch.channels_last)
            hidden.append(base.project(x).half());original.append(base(x))
        d["base_hidden"]=torch.cat(hidden)
        d["original"]=torch.cat(original)
        d["base_small"]=F.interpolate(d["original"][:,None],size=(64,64),mode="bilinear",align_corners=False).half()
    d["rows"]=rows
    print("Loaded",split,n,flush=True)
    return d


def forward(net,mode,d,idx,force_equal=False):
    if mode=="continue":return net(d["fused"][idx].float().contiguous(memory_format=torch.channels_last)),None
    args=[d[k][idx].float().contiguous(memory_format=torch.channels_last) for k in ("modal","geometry","base_small","base_hidden")]
    return net(*args,d["original"][idx],force_equal=force_equal)


@torch.no_grad()
def predict(net,mode,d,force_equal=False):
    net.eval();logits=[];gates=[]
    for start in range(0,len(d["rows"]),24):
        z,g=forward(net,mode,d,slice(start,start+24),force_equal)
        logits.append(z.cpu().numpy())
        if g is not None:gates.append(g.cpu().numpy())
    return np.concatenate(logits),np.concatenate(gates) if gates else None


def metrics(mask,d):
    y=d["y"];walk=d["walk"];gate=d["ground"]&~d["blocked"]
    tp=int((mask&(y==1)).sum());union=int(((mask|(y==1))&(y!=255)).sum())
    missing=~d["ground"]&~d["blocked"]&(y==1)&ROI
    return dict(iou=tp/union,recall=tp/int((y==1).sum()),tp=tp,union=union,
        walkway_raw_fp=int((mask&walk).sum()),walkway_pass_fp=int((mask&walk&gate).sum()),
        offroad_pass_fp=int((mask&(y==0)&gate).sum()),missing_recall=float(mask[missing].mean()))


def new_model(mode,seed,baseline_weights):
    torch.manual_seed(seed)
    net=RoadHead(256) if mode=="continue" else AdaptiveHead(mode=="adaptive")
    if mode=="continue":net.load_state_dict(baseline_weights)
    return net.cuda().to(memory_format=torch.channels_last)


def fit(mode,seed,tr,dev,weights):
    net=new_model(mode,seed,weights)
    opt=torch.optim.AdamW(net.parameters(),lr=.0003)
    order=torch.Generator().manual_seed(seed)
    ref=metrics((dev["original"].sigmoid()>=.60).cpu().numpy(),dev)
    best=None;history=[]
    for epoch in range(1,13):
        net.train()
        for idx in torch.randperm(len(tr["rows"]),generator=order).split(24):
            z,_=forward(net,mode,tr,idx)
            y=tr["target"][idx];valid=y!=255
            loss=F.binary_cross_entropy_with_logits(z[valid],y[valid].float())
            opt.zero_grad();loss.backward();opt.step()
        z,_=predict(net,mode,dev)
        stats=metrics(z>=np.log(.6/.4),dev)
        feasible=stats["walkway_pass_fp"]<=ref["walkway_pass_fp"]
        key=(feasible,stats["iou"])
        record=dict(epoch=epoch,feasible=feasible,**stats)
        history.append(record)
        if best is None or key>best[0]:
            best=(key,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},record)
        print(seed,mode,epoch,round(stats["iou"],5),stats["walkway_pass_fp"],feasible,flush=True)
    torch.save(dict(weights=best[1],selected=best[2],history=history),DEST/f"{mode}_{seed}.pt")
    return dict(selected=best[2],history=history,dev_baseline=ref)


def paired(a,b,d):
    scenes,index=np.unique([r["scene"] for r in d["rows"]],return_inverse=True)
    draws=np.random.default_rng(17).multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),size=4000)
    def agg(v):return np.bincount(index,weights=v,minlength=len(scenes))
    y=d["y"];free=d["ground"]&~d["blocked"]&d["walk"]
    at=draws@agg((a&(y==1)).sum((1,2)));au=draws@agg(((a|(y==1))&(y!=255)).sum((1,2)))
    bt=draws@agg((b&(y==1)).sum((1,2)));bu=draws@agg(((b|(y==1))&(y!=255)).sum((1,2)))
    wd=draws@agg((a&free).sum((1,2))-(b&free).sum((1,2)))
    return dict(iou_delta_pp_ci=np.percentile(100*(at/au-bt/bu),[2.5,97.5]).tolist(),
                walkway_pass_delta_ci=np.percentile(wd,[2.5,97.5]).tolist())


def main(seed):
    allrows=sum(manifests().values(),[])
    while True:
        missing=sum(not (CACHE/(r["token"]+".npy")).exists() and not (MODALITIES/(r["token"]+".npy")).exists() for r in allrows)
        ready=all((DEST/f"{s}_geometry.npz").exists() for s in manifests())
        if not missing and ready:break
        print("Waiting",missing,"geometry",ready,flush=True);time.sleep(20)
    weights=torch.load(TRAIN200/"baseline_v1/road_head.pt",map_location="cpu")["weights"]
    base=RoadHead(256).cuda().to(memory_format=torch.channels_last).eval()
    base.load_state_dict(weights);base.requires_grad_(False)
    tr=load_split("train",base);dev=load_split("dev",base)
    selection={}
    for mode in ("continue","equal","adaptive"):
        selection[mode]=fit(mode,seed,tr,dev,weights)
        (DEST/f"selection_{seed}.json").write_text(json.dumps(selection,indent=2))
    del tr,dev
    report=dict(seed=seed,selection={k:v["selected"] for k,v in selection.items()},evaluation={})
    for split in ("val","reserved"):
        d=load_split(split,base)
        baseline=(d["original"].sigmoid()>=.6).cpu().numpy()
        results={"baseline":metrics(baseline,d)};predictions={}
        for mode in selection:
            net=new_model(mode,seed,weights)
            net.load_state_dict(torch.load(DEST/f"{mode}_{seed}.pt",map_location="cpu")["weights"])
            z,g=predict(net,mode,d);m=z>=np.log(.6/.4)
            results[mode]=metrics(m,d);predictions[mode]=m
            if g is not None:
                np.savez_compressed(DEST/f"{split}_{mode}_{seed}_gates.npz",camera_weight=g.astype(np.float16))
                results[mode]["camera_weight_mean"]=float(g.mean())
                results[mode]["camera_weight_std"]=float(g.std())
            if mode=="adaptive":
                ablation,_=predict(net,mode,d,force_equal=True)
                results["adaptive_gate_disabled"]=metrics(ablation>=np.log(.6/.4),d)
        np.savez_compressed(DEST/f"{split}_{seed}_masks.npz",tokens=[r["token"] for r in d["rows"]],baseline=baseline,**predictions)
        report["evaluation"][split]=dict(metrics=results,paired={m:paired(p,baseline,d) for m,p in predictions.items()},
            adaptive_vs_equal=paired(predictions["adaptive"],predictions["equal"],d))
        del d
    (DEST/f"results_{seed}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=17)
    main(p.parse_args().seed)
