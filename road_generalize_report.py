"""Evaluate frozen checkpoints on old validation and new log-disjoint scenes."""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from audit_road_density import boundary_band
from road import OUT
from road_generalize import POLICIES, WORK, network, read, tensors


SEEDS = (17,29,43)
MODES = [s+"_"+loss for s in POLICIES for loss in ("plain","edge")] + ["original_context","full_context"]
METRICS = ("road_IoU","mIoU","boundary_F1","boundary_precision","boundary_recall","band_mIoU")
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
YY,XX = np.mgrid[-2:3,-2:3]
DISK = (XX**2+YY**2<=4).astype("uint8")


def spread(mask,kernel):
    return cv2.dilate(mask.astype("uint8"),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0


def contour(mask,valid):
    return mask & valid & spread(~mask & valid,CROSS)


def confusion(p,y,valid):
    return [(valid & a).sum((1,2)) for a in (p&(y==1),p&(y==0),~p&(y==1),~p&(y==0))]


def frame_counts(pred,y,band,gt):
    boundaries = []
    for mask,truth,row,valid in zip(pred,gt,y,y!=255):
        detected = contour(mask,valid)
        matched = (detected & spread(truth,DISK)).sum()
        recovered = (truth & spread(detected,DISK)).sum()
        boundaries.append([matched,detected.sum(),recovered,truth.sum()])
    return np.column_stack([*confusion(pred,y,y!=255),np.array(boundaries),*confusion(pred,y,band)])


def metrics(c):
    tp,fp,fn,tn,pm,pn,rm,rn,etp,efp,efn,etn = np.moveaxis(c,-1,0)
    div = lambda a,b: a/np.maximum(b,1e-12)
    iou = div(tp,tp+fp+fn)
    p,r = div(pm,pn),div(rm,rn)
    return np.stack([iou,(iou+div(tn,tn+fp+fn))/2,div(2*p*r,p+r),p,r,
                     .5*(div(etp,etp+efp+efn)+div(etn,etn+efp+efn))],axis=-1)


def as_dict(values):
    return {k:float(v) for k,v in zip(METRICS,values)}


def region_info(d):
    return {r:{"scenes":len(set(d["scene"][d["region"]==r])),
               "frames":int(np.sum(d["region"]==r))} for r in sorted(set(d["region"]))}


def examples(split,data,y,base,new,frame_delta):
    # Visualize the 25th and 75th percentile of frame-level road IoU change, seed 17.
    order = np.argsort(frame_delta,kind="stable")
    picks = [order[int((len(order)-1)*q)] for q in (.25,.75)]
    fig,axes = plt.subplots(2,5,figsize=(12,5))
    colors=ListedColormap(["#213441","#efbc62"])
    changes=ListedColormap(["#dde2e3","#137454","#b64332"])
    for row,i in enumerate(picks):
        valid = y[i]!=255
        change = np.zeros(y.shape[1:],np.uint8)
        change[(base[i]!=y[i]) & (new[i]==y[i]) & valid]=1
        change[(base[i]==y[i]) & (new[i]!=y[i]) & valid]=2
        imgs=[data["x"][i,0],np.ma.masked_where(~valid,y[i]),base[i],new[i],change]
        for j,(ax,img) in enumerate(zip(axes[row],imgs)):
            ax.imshow(img.T,origin="lower",cmap="gray" if j==0 else changes if j==4 else colors,
                      vmin=0,vmax=2 if j==4 else 1)
            ax.invert_xaxis()
            ax.set(xticks=[],yticks=[])
        axes[row,0].set_ylabel(f'{data["scene"][i]}\n{frame_delta[i]*100:+.1f} pp')
    for ax,title in zip(axes[0],("LiDAR density","Map target","Original","Selected model","Fixed / newly wrong")):
        ax.set_title(title,fontsize=10)
    fig.suptitle("New logs  |  Gold = road, green = fixed, red = newly wrong  |  Ego forward up")
    fig.tight_layout()
    fig.savefig(WORK/(split+"_cases.jpg"),dpi=150)
    plt.close(fig)
    return [{"scene":str(data["scene"][i]),"token":str(data["token"][i]),
             "region":str(data["region"][i]),"delta_road_IoU_pp":float(frame_delta[i]*100)} for i in picks]


def main():
    cv2.setNumThreads(1)
    ckpts = {(m,s):torch.load(WORK/(f"{m}_{s}.pt"),map_location="cpu") for m in MODES for s in SEEDS}
    dev = {m:[ckpts[m,s]["dev"] for s in SEEDS] for m in MODES}
    selected = max(MODES,key=lambda m:np.mean(dev[m]))
    print("development selection",selected,dev,flush=True)
    report = {"seeds":SEEDS,"dev_mIoU":dev,"selected":selected,"training_data":{}}
    for name,path in (("original",OUT/"train.npz"),("spread",WORK/"spread.npz"),("full",WORK/"full.npz")):
        report["training_data"][name]=region_info(read(path))
    scenes = json.loads((OUT.parent.parent/"nuscenes_trainval_val/v1.0-trainval/scene.json").read_text())
    logs = {s["name"]:s["log_token"] for s in scenes}
    for split,path in (("val",OUT/"val.npz"),("fresh",WORK/"fresh.npz")):
        data = read(path)
        x,_ = tensors(data)
        y = data["y"]
        band = boundary_band(y)
        gt = [contour(row==1,row!=255) for row in y]
        saved = {}
        groups,idx = np.unique([logs[s] for s in data["scene"]],return_inverse=True)
        draws = np.random.default_rng(37).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
        boots = {}
        totals = {}
        by_frame = {}
        for m in MODES:
            model = network(m.endswith("_context")).cuda().eval()
            raw = []
            errors = {}
            for seed in SEEDS:
                cp = ckpts[m,seed]
                model.load_state_dict(cp["weights"])
                with torch.no_grad():
                    mask = torch.cat([model(t)[:,0].sigmoid()>=cp["threshold"] for t in x.split(20)]).cpu().numpy()
                raw.append(frame_counts(mask,y,band,gt))
                if m in {"original_plain","spread_plain","full_plain","full_context",selected}:
                    for key,area in (("band",band),("empty_band",band & (data["x"][:,0]==0)),
                                     ("outside_band",(y!=255) & ~band)):
                        tp,fp,fn,tn = (a.sum() for a in confusion(mask,y,area))
                        errors.setdefault(key,[]).append([fn/max(tp+fn,1),fp/max(tn+fp,1)])
                if split=="fresh" and seed==SEEDS[0] and m in ("original_plain",selected):
                    saved[m]=mask
            raw=np.array(raw)
            by_frame[m]=raw
            per_log=np.zeros((len(SEEDS),len(groups),raw.shape[-1]))
            for i in range(len(SEEDS)):
                np.add.at(per_log[i],idx,raw[i])
            boots[m]=metrics(draws@per_log).mean(axis=0)
            seed_metrics=metrics(raw.sum(axis=1))
            totals[m]=dict(mean=as_dict(seed_metrics.mean(axis=0)),
                           per_seed=[as_dict(a) for a in seed_metrics],
                           error_rates={k:dict(road_miss=float(np.mean(v,0)[0]),background_false_road=float(np.mean(v,0)[1])) for k,v in errors.items()},
                           regions={r:as_dict(metrics(raw[:,data["region"]==r].sum(1)).mean(0)) for r in set(data["region"])})
        pairs=[(m,"original_plain") for m in MODES if m!="original_plain"]
        pairs.extend([("spread_edge","spread_plain"),("full_edge","full_plain"),("full_plain","spread_plain")])
        pairs.append(("full_context","full_plain"))
        stats={}
        for a,b in pairs:
            lo,hi=np.percentile((boots[a]-boots[b])*100,[2.5,97.5],axis=0)
            stats[a+" minus "+b] = {k:[float(l),float(h)] for k,l,h in zip(METRICS,lo,hi)}
        result=dict(frames=len(y),scenes=len(set(data["scene"])),logs=len(groups),regions=region_info(data),
                    methods=totals,delta_95_pp=stats)
        if split=="fresh":
            delta=metrics(by_frame[selected][0])[:,0]-metrics(by_frame["original_plain"][0])[:,0]
            result["examples"]=examples(split,data,y,saved["original_plain"],saved[selected],delta)
        report[split]=result
        print(split,{m:[round(totals[m]["mean"][k]*100,2) for k in METRICS[:3]] for m in MODES},flush=True)
    (WORK/"results.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
