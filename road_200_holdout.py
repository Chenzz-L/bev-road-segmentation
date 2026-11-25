"""Evaluate selected heads on the reserved 27 scenes without any retuning."""
import json
import time
from collections import Counter

import numpy as np
import torch

from road_200_data import DEST
from road_shared_features import OUT
from road_shared_heads import RoadHead
from road_200_train import predict
from road_frozen_evaluate import count
from road_obstacle_check import ROI


@torch.inference_mode()
def main():
    frozen=OUT/"passable_walkway/frozen"
    rows=json.loads((frozen/"new_manifest.json").read_text())
    paths=[DEST/"eval27_features"/(r["token"]+".npy") for r in rows]
    required=paths+[DEST/f"results_{seed}.json" for seed in (17,29,43)]
    while not all(p.is_file() for p in required):time.sleep(20)
    x=torch.tensor(np.stack([np.load(p) for p in paths]),device="cuda")
    refs=[]
    for r in rows:
        with np.load(frozen/"frames"/(r["token"]+".npz")) as d:
            refs.append({k:d[k] for k in ("y","walk","ground","blocked","original","s17")})
    y,walk,ground,blocked,old,corrected=[np.stack([r[k] for r in refs]) for k in ("y","walk","ground","blocked","original","s17")]
    gate=ground&~blocked
    missing=~ground&~blocked&(y==1)&ROI
    def metrics(m):
        c=count(m,y,walk,gate,missing)
        return dict(**c,iou=c["tp"]/c["union"],recall=c["tp"]/c["road_n"],missing_recall=c["missing_tp"]/c["missing_n"])
    scenes,index=np.unique([r["scene"] for r in rows],return_inverse=True)
    draws=np.random.default_rng(17).multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),size=4000)
    def agg(v):return np.bincount(index,weights=v,minlength=len(scenes))
    def paired(a,b):
        atp=draws@agg((a&(y==1)).sum((1,2)))
        au=draws@agg(((a|(y==1))&(y!=255)).sum((1,2)))
        btp=draws@agg((b&(y==1)).sum((1,2)))
        bu=draws@agg(((b|(y==1))&(y!=255)).sum((1,2)))
        delta=draws@agg((a&walk&gate).sum((1,2))-(b&walk&gate).sum((1,2)))
        return dict(iou_delta_pp_ci=np.percentile(100*(atp/au-btp/bu),[2.5,97.5]).tolist(),
                    walkway_delta_ci=np.percentile(delta,[2.5,97.5]).tolist())
    # Check that re-extraction preserves the historical baseline on this set.
    net=RoadHead(256).cuda().to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(frozen/"fusion_head.pt",map_location="cpu")["weights"])
    replay=(predict(net,x)>=.45).cpu().numpy()
    disagreement=int((replay!=old).sum())
    assert disagreement/old.size<.001,disagreement
    report=dict(frames=len(rows),scenes=len(scenes),replay_changed_cells=disagreement,
                historical=metrics(old),previous_correction=metrics(corrected),seeds={})
    for seed in (17,29,43):
        masks={};result={};fixed={}
        for kind in ("original","longer","expanded"):
            ckpt=torch.load(DEST/f"{kind}_{seed}.pt",map_location="cpu")
            net.load_state_dict(ckpt["weights"])
            prob=predict(net,x).cpu().numpy()
            masks[kind]=prob>=ckpt["threshold"]
            result[kind]=dict(threshold=ckpt["threshold"],**metrics(masks[kind]))
            fixed[kind]=metrics(prob>=.45)
        regions={}
        for region in sorted({r["region"] for r in rows}):
            select=np.array([r["region"]==region for r in rows])
            regions[region]={}
            for kind,m in masks.items():
                c=count(m[select],y[select],walk[select],gate[select],missing[select])
                regions[region][kind]=dict(**c,iou=c["tp"]/c["union"])
        report["seeds"][str(seed)]=dict(metrics=result,fixed_threshold_045=fixed,by_region=regions,
            paired={kind:paired(masks["expanded"],masks[kind]) for kind in ("original","longer")},
            versus_historical=paired(masks["expanded"],old))
        np.savez_compressed(DEST/f"eval27_{seed}_masks.npz",tokens=[r["token"] for r in rows],**masks)
    (DEST/"eval27_results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
