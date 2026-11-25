"""Same-head scene-expansion experiment with an equal-update training control."""
import argparse
import json
import math
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from road import centers,OUT as ROAD
from road_full_validation import WORK
from road_shared_heads import RoadHead
from road_shared_features import OUT,CACHE
from road_rgb_ablation import vector_targets
from road_walkway_errors import layers
from road_obstacle_check import ROI
from road_200_data import DEST,FEATURES


def load_data(rows,split):
    tokens=[r["token"] for r in rows]
    y=torch.tensor(vector_targets(split,tokens),device="cuda")
    x=torch.empty((len(rows),256,64,64),device="cuda",dtype=torch.float16)
    for i,t in enumerate(tokens):
        path=FEATURES/(t+".npy")
        arr=np.load(path if path.exists() else CACHE/(t+".npy"),mmap_mode="r")
        x[i]=torch.from_numpy((arr if path.exists() else arr[336:592]).copy())
    return x,y


@torch.inference_mode()
def predict(net,x):
    net.eval()
    return torch.cat([net(b.float().contiguous(memory_format=torch.channels_last)).sigmoid() for b in x.split(24)])


def train(kind,seed,tr,dev):
    torch.manual_seed(seed)
    net=RoadHead(256).cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    order=torch.Generator().manual_seed(seed)
    x,y=tr
    if kind!="expanded":x,y=x[:624],y[:624]
    budget=468 if kind=="original" else 1800
    interval=26 if kind=="original" else 100
    dx,dy=dev
    step=0;best=-1;curve=[]
    while step<budget:
        net.train()
        for idx in torch.randperm(len(x),generator=order).split(24):
            z=net(x[idx].float().contiguous(memory_format=torch.channels_last))
            target=y[idx];valid=target!=255
            loss=F.binary_cross_entropy_with_logits(z[valid],target[valid].float())
            opt.zero_grad();loss.backward();opt.step()
            step+=1
            if step%interval==0:
                prob=predict(net,dx)
                for threshold in (.35,.40,.45,.50,.55,.60,.65):
                    m=prob>=threshold
                    iou=((m&(dy==1)).sum()/(((m|(dy==1))&(dy!=255)).sum())).item()
                    if iou>best:
                        best=iou
                        selected=dict(step=step,threshold=threshold,dev_road_iou=iou)
                        weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                curve.append(dict(step=step,best=best))
                print(seed,kind,"step",step,"dev best",round(best,6),flush=True)
                net.train()
            if step==budget:break
    checkpoint=dict(weights=weights,seed=seed,kind=kind,total_steps=budget,**selected)
    torch.save(checkpoint,DEST/f"{kind}_{seed}.pt")
    net.load_state_dict(weights)
    return net,dict(seed=seed,kind=kind,total_steps=budget,**selected,curve=curve)


def main(seed):
    rows=json.loads((DEST/"manifest.json").read_text())
    original=json.loads((OUT/"manifest.json").read_text())
    dev_rows=[r for r in original if r["split"]=="dev"]
    val_rows=[r for r in original if r["split"]=="val"]
    assert [r["token"] for r in rows[:624]]==[r["token"] for r in original if r["split"]=="train"]
    tr=load_data(rows[:624],"train");dev=load_data(dev_rows,"dev")
    selected={}
    for kind in ("original","longer","expanded"):
        if kind=="expanded":
            del tr
            while True:
                missing=sum(not (FEATURES/(r["token"]+".npy")).is_file() for r in rows[624:])
                if not missing:break
                print(seed,"waiting for features",missing,flush=True)
                time.sleep(20)
            tr=load_data(rows,"train")
        net,info=train(kind,seed,tr,dev)
        selected[kind]=info
        (DEST/f"selection_{seed}.json").write_text(json.dumps(selected,indent=2))
        print(seed,kind,{k:v for k,v in info.items() if k!="curve"},flush=True)
    (DEST/f"selection_{seed}.json").write_text(json.dumps(selected,indent=2))
    del tr,dev
    vx,vy=load_data(val_rows,"val")
    y=vy.cpu().numpy()
    tokens=[r["token"] for r in val_rows]
    with np.load(OUT/"references.npz") as d:
        assert list(d["tokens"])==tokens and np.array_equal(y,d["label"])
        ground,blocked=d["ground"],d["blocked"]
    with np.load(OUT/"fusion_masks.npz") as d:
        assert list(d["tokens"])==tokens
        historic=d["masks"]
    with np.load(ROAD/"val.npz") as d:regions=dict(zip(d["token"],d["region"]))
    poses=json.loads((WORK/"vector_labels/poses.json").read_text())
    maps={};walk=np.zeros_like(ground)
    for i,r in enumerate(val_rows):
        region=regions[r["token"]]
        if region not in maps:maps[region]=layers(region)[1]
        tx,ty,yaw=poses[r["token"]];co,si=np.cos(yaw),np.sin(yaw)
        xx=centers[:,:,0]*co-centers[:,:,1]*si+tx
        yy=centers[:,:,0]*si+centers[:,:,1]*co+ty
        walk[i]=maps[region].lookup(xx,yy)&(y[i]==0)
    free=ground&~blocked
    missing=~ground&~blocked&(y==1)&ROI
    def metrics(m):
        tp=int((m&(y==1)).sum());union=int(((m|(y==1))&(y!=255)).sum())
        return dict(tp=tp,union=union,iou=tp/union,recall=tp/int((y==1).sum()),
                    walkway_raw_fp=int((m&walk).sum()),walkway_pass_fp=int((m&walk&free).sum()),
                    offroad_pass_fp=int((m&(y==0)&free).sum()),missing_recall=float(m[missing].mean()))
    results={"historical":metrics(historic)}
    predictions={}
    for kind in selected:
        ckpt=torch.load(DEST/f"{kind}_{seed}.pt",map_location="cpu")
        net=RoadHead(256).cuda().to(memory_format=torch.channels_last)
        net.load_state_dict(ckpt["weights"])
        m=(predict(net,vx)>=ckpt["threshold"]).cpu().numpy()
        results[kind]=dict(threshold=ckpt["threshold"],**metrics(m))
        predictions[kind]=m
        np.savez_compressed(DEST/f"{kind}_{seed}_masks.npz",tokens=tokens,masks=m)
    scenes,index=np.unique([r["scene"] for r in val_rows],return_inverse=True)
    draws=np.random.default_rng(17).multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),size=4000)
    def agg(v):return np.bincount(index,weights=v,minlength=len(scenes))
    paired={}
    a=predictions["expanded"]
    for name,b in (("original",predictions["original"]),("longer",predictions["longer"]),("historical",historic)):
        atp=draws@agg((a&(y==1)).sum((1,2)))
        au=draws@agg(((a|(y==1))&(y!=255)).sum((1,2)))
        btp=draws@agg((b&(y==1)).sum((1,2)))
        bu=draws@agg(((b|(y==1))&(y!=255)).sum((1,2)))
        delta=draws@agg((a&walk&free).sum((1,2))-(b&walk&free).sum((1,2)))
        paired[name]=dict(iou_delta_pp_ci=np.percentile(100*(atp/au-btp/bu),[2.5,97.5]).tolist(),
                          walkway_delta_ci=np.percentile(delta,[2.5,97.5]).round().astype(int).tolist())
    report=dict(seed=seed,selection={k:{n:v for n,v in s.items() if n!="curve"} for k,s in selected.items()},
                validation=results,paired=paired)
    (DEST/f"results_{seed}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=17)
    main(p.parse_args().seed)
