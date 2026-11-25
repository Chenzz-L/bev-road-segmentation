from road import MAP_ROOT
"""Evaluate frozen correction checkpoints on previously unused road-project scenes."""
import argparse
import hashlib
import json
import time
from collections import Counter,defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from pyquaternion import Quaternion
from nuscenes.eval.detection.config import config_factory
from nuscenes.map_expansion.map_api import NuScenesMap

from road import DATA,ROOT,centers,grid_points
from road_camera import pose
from road_obstacle_check import occupancy,ROI
from road_obstacle_boxes import footprint
from road_shared_features import OUT,PROJECT,sampling_grid
from road_shared_heads import RoadHead
from road_walkway_errors import layers


DEST=OUT/"passable_walkway/frozen"
CLASSES=("car","truck","construction_vehicle","bus","trailer","barrier","motorcycle","bicycle","pedestrian","traffic_cone")


def count(mask,y,walk,gate,missing):
    return dict(tp=int((mask&(y==1)).sum()),road_n=int((y==1).sum()),
                union=int(((mask|(y==1))&(y!=255)).sum()),walkway_fp=int((mask&walk).sum()),
                walkway_pass_fp=int((mask&walk&gate).sum()),offroad_pass_fp=int((mask&(y==0)&gate).sum()),
                missing_tp=int((mask&missing).sum()),missing_n=int(missing.sum()))


@torch.inference_mode()
def run():
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet3d.apis.inference import convert_SyncBN
    from mmdet3d.registry import MODELS
    from mmdet3d.structures import LiDARInstance3DBoxes
    from mmdet3d.utils import register_all_modules
    from mmdet3d.evaluation.metrics.nuscenes_metric import output_to_nusc_box,lidar_nusc_box_to_global
    import projects.BEVFusion.bevfusion

    frozen=json.loads((DEST/"freeze.json").read_text())
    for file,digest in frozen["sha256"].items():
        assert hashlib.sha256((DEST/file).read_bytes()).hexdigest()==digest
    rows=json.loads((DEST/"new_manifest.json").read_text())
    ep={r["token"]:r for r in json.loads((DATA/"v1.0-trainval/ego_pose.json").read_text())}
    cs={r["token"]:r for r in json.loads((DATA/"v1.0-trainval/calibrated_sensor.json").read_text())}
    sd={r["token"]:r for r in json.loads((DATA/"v1.0-trainval/sample_data.json").read_text())}
    cfg=Config.fromfile(str(PROJECT/"bevfusion_full_val.py"))
    register_all_modules()
    init_default_scope("mmdet3d")
    convert_SyncBN(cfg.model)
    model=MODELS.build(cfg.model)
    load_checkpoint(model,str(PROJECT/"bevfusion-pretrained.pth"),map_location="cpu")
    model.cuda().eval()
    single=Compose([p for p in cfg.test_pipeline if p.type!="LoadPointsFromMultiSweeps"])
    multiple=Compose(cfg.test_pipeline)
    heads={}
    for name,channels,file in [("original",256,"fusion_head.pt")]+[(f"s{s}",261,f"s{s}_w2.pt") for s in (17,29,43)]:
        head=RoadHead(channels).cuda().eval().to(memory_format=torch.channels_last)
        head.load_state_dict(torch.load(DEST/file,map_location="cpu")["weights"])
        heads[name]=head
    maps={}
    records=[]
    masks=defaultdict(list)
    reference=[]
    checkpoints=DEST/"frames"
    checkpoints.mkdir(exist_ok=True)
    for i,r in enumerate(rows):
        saved=checkpoints/(r["token"]+".npz")
        info=checkpoints/(r["token"]+".json")
        if saved.exists() and info.exists():
            records.append(json.loads(info.read_text()))
            with np.load(saved) as data:
                for name in heads:masks[name].append(data[name])
                reference.append(data["y"])
            continue
        current=sd[r["token"]]
        paths=[r["lidar"]]+[v["img_path"] for v in r["images"].values()]
        paths += [str(DATA/s["filename"]) for s in r["sweeps"]]
        from pathlib import Path
        while not all(Path(p).is_file() for p in paths):
            time.sleep(10)
        calib=np.array(r["ego_from_lidar"])
        ego=np.array(r["ego_to_global"])
        world=ego@calib
        sweeps=[]
        for old in r["sweeps"]:
            relative=np.linalg.solve(world,pose(ep[old["ego_pose_token"]])@pose(cs[old["calibrated_sensor_token"]]))
            converter=np.eye(4)
            converter[:3,:3]=relative[:3,:3].T
            converter[:3,3]=-relative[:3,3]
            sweeps.append(dict(timestamp=old["timestamp"]/1e6,
                lidar_points=dict(lidar_path=str(DATA/old["filename"]),lidar2sensor=converter.tolist())))
        entry=dict(images=r["images"],lidar_points=dict(lidar_path=r["lidar"],num_pts_feats=5,lidar2ego=calib.tolist()),
                   ego2global=ego.tolist(),timestamp=current["timestamp"]/1e6,
                   sample_idx=r["sample"],box_type_3d=LiDARInstance3DBoxes,lidar_sweeps=sweeps)
        captured=[]
        hook=model.fusion_layer.register_forward_hook(lambda module,args,output:captured.append(output))
        inp=model.data_preprocessor(pseudo_collate([single(entry.copy())]),training=False)
        inp["inputs"]["points"][0][:,4]=0
        model.extract_feat(inp["inputs"],[s.metainfo for s in inp["data_samples"]])
        hook.remove()
        assert len(captured)==1
        fused=F.grid_sample(captured[0],sampling_grid(calib),align_corners=False).half()
        before=heads["original"](fused.float().contiguous(memory_format=torch.channels_last))
        del captured
        full=model.data_preprocessor(pseudo_collate([multiple(entry.copy())]),training=False)
        prediction=model.predict(full["inputs"],full["data_samples"])[0].pred_instances_3d
        detection={key:getattr(prediction,key).cpu() for key in ("bboxes_3d","scores_3d","labels_3d")}
        boxes,_=output_to_nusc_box(detection)
        boxes=lidar_nusc_box_to_global(entry,boxes,CLASSES,config_factory("detection_cvpr_2019"))
        det=[dict(detection_name=CLASSES[b.label],detection_score=float(b.score),translation=b.center.tolist(),
                  size=b.wlh.tolist(),rotation=b.orientation.elements.tolist()) for b in boxes]
        raw=np.fromfile(r["lidar"],np.float32).reshape(-1,5)
        xyz=raw[:,:3]@calib[:3,:3].T+calib[:3,3]
        geometry=grid_points(xyz,raw[:,3])
        direct,_,ground=occupancy(xyz,geometry)
        current_pose=ep[current["ego_pose_token"]]
        blocked=direct|footprint(det,current_pose)
        gate=ground&~blocked
        region=r["region"]
        if region not in maps:
            api=NuScenesMap(dataroot=str(MAP_ROOT),map_name=region)
            maps[region]=(*layers(region)[:2],api.canvas_edge)
        roads,walks,edge=maps[region]
        yaw=np.arctan2(ego[1,0],ego[0,0])
        co,si=np.cos(yaw),np.sin(yaw)
        x=centers[:,:,0]*co-centers[:,:,1]*si+ego[0,3]
        ycoord=centers[:,:,0]*si+centers[:,:,1]*co+ego[1,3]
        y=roads.lookup(x,ycoord).astype(np.uint8)
        y[(x<0)|(ycoord<0)|(x>=edge[0])|(ycoord>=edge[1])]=255
        walk=walks.lookup(x,ycoord)&(y==0)
        missing=~ground&~blocked&(y==1)&ROI
        geom=torch.tensor(geometry[None],device="cuda",dtype=torch.float32)
        gx=F.interpolate(geom,size=(64,64),mode="bilinear",align_corners=False).half()
        bx=F.interpolate(before[:,None],size=(64,64),mode="bilinear",align_corners=False).half()
        inp_head=torch.cat([fused,gx,bx],dim=1).float().contiguous(memory_format=torch.channels_last)
        pred={"original":(before.sigmoid()[0]>=frozen["threshold"]).cpu().numpy()}
        for name in ("s17","s29","s43"):
            delta=1.5*heads[name](inp_head).tanh()
            output=before+torch.tensor(gate,device="cuda")*delta
            pred[name]=(output.sigmoid()[0]>=frozen["threshold"]).cpu().numpy()
            assert not ((pred[name]!=pred["original"])&~gate).any()
        row=dict(token=r["token"],scene=r["scene"],log=r["log"],region=region,
                 metrics={k:count(v,y,walk,gate,missing) for k,v in pred.items()},sweeps=len(sweeps))
        np.savez_compressed(saved,**pred,y=y,ground=ground,blocked=blocked,walk=walk)
        info.write_text(json.dumps(row))
        records.append(row)
        for name in heads:masks[name].append(pred[name])
        reference.append(y)
        if (i+1)%8==0:print("Evaluated",i+1,"/",len(rows),flush=True)
    totals={name:Counter() for name in heads}
    for r in records:
        for name in heads:totals[name].update(r["metrics"][name])
    def rates(c):
        return dict(**c,iou=c["tp"]/c["union"],recall=c["tp"]/c["road_n"],missing_recall=c["missing_tp"]/c["missing_n"])
    scenes=sorted({r["scene"] for r in records})
    draws=np.random.default_rng(17).multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),size=4000)
    confidence={}
    for name in ("s17","s29","s43"):
        data=[]
        for scene in scenes:
            b=Counter();a=Counter()
            for r in records:
                if r["scene"]==scene:b.update(r["metrics"]["original"]);a.update(r["metrics"][name])
            data.append([a["tp"],a["union"],b["tp"],b["union"],a["walkway_pass_fp"]-b["walkway_pass_fp"]])
        boot=draws@np.array(data)
        confidence[name]=dict(iou_delta_pp_ci=np.percentile(100*(boot[:,0]/boot[:,1]-boot[:,2]/boot[:,3]),[2.5,97.5]).tolist(),
                              walkway_pass_delta_ci=np.percentile(boot[:,4],[2.5,97.5]).round().astype(int).tolist())
    by_region={}
    for region in sorted({r["region"] for r in records}):
        c={name:Counter() for name in heads}
        for r in records:
            if r["region"]==region:
                for name in heads:c[name].update(r["metrics"][name])
        by_region[region]={k:rates(v) for k,v in c.items()}
    report=dict(frames=len(records),scenes=len(scenes),logs=len({r["log"] for r in records}),
                metrics={k:rates(v) for k,v in totals.items()},scene_bootstrap=confidence,by_region=by_region)
    (DEST/"new_results.json").write_text(json.dumps(report,indent=2))
    (DEST/"new_frames.json").write_text(json.dumps(records))
    print(json.dumps(report),flush=True)


if __name__=="__main__":run()
