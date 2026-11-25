"""Fit matched road heads, then score passable cells with the saved obstacle protocol."""
import argparse
from collections import Counter,defaultdict
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes
import torch
from torch import nn
import torch.nn.functional as F

from road import DATA, OUT as ROAD, grid_points
from road_full_validation import WORK
from road_obstacle_boxes import DETECTIONS, OUT as BOX_RUN, footprint
from road_obstacle_check import occupancy,references,ROI
from road_rgb_ablation import vector_targets
from road_shared_features import OUT,CACHE,MODES


CHANNELS = {"camera":slice(0,80),"lidar":slice(80,336),"fusion":slice(336,592)}


class RoadHead(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(channels,32,1),nn.GroupNorm(4,32),nn.ReLU())
        blocks=[]
        for dilation in (1,2,4):
            blocks += [nn.Conv2d(32,32,3,dilation=dilation,padding=dilation),nn.GroupNorm(4,32),nn.ReLU()]
        self.context=nn.Sequential(*blocks)
        self.output=nn.Conv2d(32,1,1)

    def forward(self,x):
        z=self.project(x)
        logits=self.output(z+self.context(z))
        return F.interpolate(logits,size=(96,96),mode="bilinear",align_corners=False)[:,0]


def labels(split,rows):
    with np.load(ROAD/f"{split}.npz") as d:
        tokens=d["token"]
    indices=[r["index"] for r in rows]
    assert list(tokens[indices])==[r["token"] for r in rows]
    return torch.from_numpy(vector_targets(split,tokens)[indices]).cuda()


def fit(mode):
    torch.manual_seed(17)
    rows=json.loads((OUT/"manifest.json").read_text())
    ch=CHANNELS[mode]
    data={}
    for split in ("train","dev","val"):
        part=[r for r in rows if r["split"]==split]
        x=torch.empty((len(part),ch.stop-ch.start,64,64),dtype=torch.float16,device="cuda")
        for i,r in enumerate(part):
            arr=np.load(CACHE/(r["token"]+".npy"),mmap_mode="r")
            x[i]=torch.from_numpy(arr[ch].copy())
        data[split]=(x,labels(split,part),part)
    tr,ty,_=data["train"]
    dx,dy,_=data["dev"]
    network=RoadHead(ch.stop-ch.start).cuda().to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(network.parameters(),lr=.001)
    best=-1
    order=torch.Generator().manual_seed(17)
    def predict(x):
        return torch.cat([network(t.float().contiguous(memory_format=torch.channels_last)).sigmoid()
                          for t in x.split(24)])
    for epoch in range(1,19):
        network.train()
        for idx in torch.randperm(len(tr),generator=order).split(24):
            l=network(tr[idx].float().contiguous(memory_format=torch.channels_last))
            y=ty[idx]
            valid=y!=255
            loss=F.binary_cross_entropy_with_logits(l[valid],y[valid].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        network.eval()
        with torch.no_grad():
            prob=predict(dx)
            for threshold in (.35,.40,.45,.50,.55,.60,.65):
                m=prob>=threshold
                iou=((m & (dy==1)).sum()/(((m | (dy==1)) & (dy!=255)).sum())).item()
                if iou>best:
                    best=iou
                    selected=dict(epoch=epoch,threshold=threshold,dev_road_iou=iou)
                    weights={k:v.cpu().clone() for k,v in network.state_dict().items()}
    network.load_state_dict(weights)
    vx,_,part=data["val"]
    with torch.no_grad():
        masks=(predict(vx)>=selected["threshold"]).cpu().numpy()
    np.savez_compressed(OUT/(mode+"_masks.npz"),masks=masks,tokens=[r["token"] for r in part])
    torch.save(dict(weights=weights,**selected),OUT/(mode+"_head.pt"))
    (OUT/(mode+"_dev.json")).write_text(json.dumps(selected))
    print(mode,selected,flush=True)


def plot(scene,y,pred,ground,blocked):
    imgs=[np.where(y==1,1,0)]
    for name in MODES:
        m=pred[name] & ROI
        state=m.astype("int32")
        state[m&ground]=2
        state[m&blocked]=3
        imgs.append(state)
    colors=np.array([[35,48,58],[118,137,149],[107,209,167],[248,156,108]])/255
    fig,axes=plt.subplots(1,4,figsize=(12,3.7))
    for ax,im,title in zip(axes,imgs,("Map road reference","Camera road head","LiDAR road head","Fused road head")):
        ax.imshow(colors[im.T],origin="lower",interpolation="nearest")
        ax.invert_xaxis()
        ax.set(xticks=[],yticks=[],title=title)
    fig.suptitle(scene+"  |  same frame, same obstacle mask")
    fig.text(.5,.055,"Heads: green = observed free road, orange = blocked, grey = road with insufficient evidence. Ego forward is up.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.01,right=.99,top=.83,bottom=.16,wspace=.07)
    fig.savefig(OUT/"paired.png",dpi=165)
    plt.close(fig)


def prepare_references():
    rows=[r for r in json.loads((OUT/"manifest.json").read_text()) if r["split"]=="val"]
    with np.load(ROAD/"val.npz") as d:
        gt=vector_targets("val",d["token"])
    nusc=NuScenes("v1.0-trainval",str(DATA),verbose=False)
    boxes=json.loads(DETECTIONS.read_bytes())["results"]
    old={r["lidar_token"]:r for r in json.loads((BOX_RUN/"frames.json").read_text())}
    assert set(old)=={r["token"] for r in rows}
    grids=defaultdict(list)
    for r in rows:
        token=r["token"]
        y=gt[r["index"]]
        sd=nusc.get("sample_data",token)
        raw=np.fromfile(r["lidar"],np.float32).reshape(-1,5)
        calib=np.asarray(r["ego_from_lidar"])
        xyz=raw[:,:3]@calib[:3,:3].T+calib[:3,3]
        feat=grid_points(xyz,raw[:,3])
        direct,_,ground=occupancy(xyz,feat)
        b=footprint(boxes[r["sample"]],nusc.get("ego_pose",sd["ego_pose_token"]))
        blocked=direct|b
        vehicle,outside=references(nusc,sd,y==1)
        outside &= feat[0]>0
        for k,v in (("blocked",blocked),("ground",ground),("vehicle",vehicle),("outside",outside),("label",y)):
            grids[k].append(v)
    np.savez_compressed(OUT/"references.npz",tokens=[r["token"] for r in rows],**grids)
    print("Verified reference frames",len(rows),flush=True)


def evaluate():
    rows=[r for r in json.loads((OUT/"manifest.json").read_text()) if r["split"]=="val"]
    tokens=[r["token"] for r in rows]
    masks={}
    for m in MODES:
        with np.load(OUT/(m+"_masks.npz")) as d:
            assert list(d["tokens"])==tokens
            masks[m]=d["masks"]
    with np.load(OUT/"references.npz") as d:
        assert list(d["tokens"])==tokens
        refs={k:d[k] for k in ("blocked","ground","vehicle","outside","label")}
    example=json.loads((BOX_RUN/"summary.json").read_text())["example"]
    totals=defaultdict(Counter)
    per=[]
    for j,r in enumerate(rows):
        token=r["token"]
        blocked,ground,vehicle,outside,y=(v[j] for v in refs.values())
        pred={m:masks[m][j] for m in MODES}
        frees={m:p&ground&~blocked for m,p in pred.items()}
        stats={}
        for m,p in pred.items():
            free=frees[m]
            stats[m]=dict(road_tp=int((p&(y==1)).sum()),road_union=int(((p|(y==1))&(y!=255)).sum()),
                          outside=int(outside.sum()),outside_free=int((outside&free).sum()),
                          offroad_free=int(((y==0)&free).sum()),vehicle_free=int((vehicle&free).sum()),
                          added_good_vs_lidar=int((outside&free&~frees["lidar"]).sum()),
                          lost_good_vs_lidar=int((outside&~free&frees["lidar"]).sum()),
                          added_offroad_vs_lidar=int(((y==0)&free&~frees["lidar"]).sum()),
                          removed_offroad_vs_lidar=int(((y==0)&~free&frees["lidar"]).sum()))
            totals[m].update(stats[m])
        per.append(dict(token=token,scene=r["scene"],metrics=stats))
        if token==example[1]: plot(r["scene"],y,pred,ground,blocked)
    original=json.loads((WORK/"direct_rgb/summary.json").read_text())["counts"]["original"]
    assert totals["lidar"]["outside"]==original["outside_n"]
    scene_counts=defaultdict(lambda:defaultdict(Counter))
    for r in per:
        for m,s in r["metrics"].items(): scene_counts[r["scene"]][m].update(s)
    n=len(scene_counts)
    draws=np.random.default_rng(17).multinomial(n,np.full(n,1/n),size=4000)
    ious={}
    for m in MODES:
        c=draws@np.array([[s[m]["road_tp"],s[m]["road_union"]] for s in scene_counts.values()])
        ious[m]=100*c[:,0]/c[:,1]
    report=dict(frames=len(rows),scenes=n,train=624,dev=120,seed=17,counts=totals,
                selection={m:json.loads((OUT/(m+"_dev.json")).read_text()) for m in MODES},
                fusion_minus_lidar_iou_pp_ci=np.percentile(ious["fusion"]-ious["lidar"],[2.5,97.5]).tolist(),
                previous_segformer=original)
    (OUT/"summary.json").write_text(json.dumps(report,indent=2))
    (OUT/"frames.json").write_text(json.dumps(per))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("job",choices=(*MODES,"references","evaluate"))
    job=p.parse_args().job
    if job=="references": prepare_references()
    elif job=="evaluate": evaluate()
    else: fit(job)
