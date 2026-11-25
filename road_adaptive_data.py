"""Prepare aligned modality tensors and observed geometry for a locked experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from road import OUT as ROAD, centers, grid_points
from road_full_validation import WORK
from road_shared_features import OUT, CACHE, PROJECT, sampling_grid
from road_200_data import DEST as TRAIN200
from road_rgb_ablation import vector_targets
from road_obstacle_check import occupancy
from road_walkway_errors import layers

DEST=TRAIN200/"adaptive_v1"
MODALITIES=DEST/"modalities"


def manifests():
    old=json.loads((OUT/"manifest.json").read_text())
    return dict(train=json.loads((TRAIN200/"manifest.json").read_text()),
                dev=[r for r in old if r["split"]=="dev"],
                val=[r for r in old if r["split"]=="val"],
                reserved=json.loads((OUT/"passable_walkway/frozen/new_manifest.json").read_text()))


def prepare():
    DEST.mkdir(exist_ok=True)
    protocol=dict(seeds=[17,29,43],threshold=.60,epochs=12,batch_size=24,lr=.0003,
        modes=["continue","equal","adaptive"],loss="unweighted BCE",
        training="same 2400 frames / 200 scenes; frozen BEVFusion and baseline for residual modes",
        camera="80-channel pure-camera BEV with empty sparse-depth input, as in original camera cache",
        lidar="256-channel LiDAR BEV; current sweep only",
        geometry="point count, min/max height, intensity, seen, ground, elevated return; no map labels as inputs",
        adaptive="spatial softmax gate over camera/LiDAR projections; bounded residual to baseline logits",
        equal="identical residual network with constant 0.5/0.5 gate",
        selection="At fixed threshold, maximize dev IoU subject to walkway pass FP <= baseline; otherwise highest dev IoU and flag infeasible",
        dev_pass="ground and no elevated return",eval_pass="original fixed ground and blocked masks",
        evaluation="150 scenes / 542 frames and reserved 27 scenes / 108 frames, after all models selected")
    (DEST/"protocol.json").write_text(json.dumps(protocol,indent=2))
    poses=json.loads((WORK/"vector_labels/poses.json").read_text())
    regions={}
    for split in ("train","dev","val"):
        with np.load(ROAD/f"{split}.npz") as d:regions.update(zip(d["token"],d["region"]))
    maps={}
    for split,rows in manifests().items():
        path=DEST/f"{split}_geometry.npz"
        if path.exists():continue
        if split!="reserved":y=vector_targets(split,[r["token"] for r in rows])
        else:
            y=np.stack([np.load(OUT/"passable_walkway/frozen/frames"/(r["token"]+".npz"))["y"] for r in rows])
        geometries=[];grounds=[];blocks=[];walk=[]
        for i,r in enumerate(rows):
            raw=np.fromfile(r["lidar"],np.float32).reshape(-1,5)
            t=np.array(r["ego_from_lidar"])
            xyz=raw[:,:3]@t[:3,:3].T+t[:3,3]
            geo=grid_points(xyz,raw[:,3]);direct,_,ground=occupancy(xyz,geo)
            features=np.concatenate([geo,(geo[:1]>0),ground[None],direct[None]],axis=0)
            geometries.append(F.interpolate(torch.tensor(features[None]).float(),size=(64,64),mode="bilinear",align_corners=False)[0].numpy().astype(np.float16))
            grounds.append(ground);blocks.append(direct)
            region=r.get("region",regions.get(r["token"]))
            if region not in maps:maps[region]=layers(region)[1]
            if "ego_to_global" in r:
                ego=np.array(r["ego_to_global"]);tx,ty=ego[:2,3];yaw=np.arctan2(ego[1,0],ego[0,0])
            else:tx,ty,yaw=poses[r["token"]]
            c,s=np.cos(yaw),np.sin(yaw)
            xx=centers[:,:,0]*c-centers[:,:,1]*s+tx
            yy=centers[:,:,0]*s+centers[:,:,1]*c+ty
            walk.append(maps[region].lookup(xx,yy)&(y[i]==0))
            if (i+1)%400==0:print("Geometry",split,i+1,flush=True)
        ground=np.stack(grounds);blocked=np.stack(blocks)
        if split=="val":
            with np.load(OUT/"references.npz") as ref:
                assert list(ref["tokens"])==[r["token"] for r in rows]
                assert np.array_equal(y,ref["label"]) and np.array_equal(ground,ref["ground"])
                blocked=ref["blocked"]
        elif split=="reserved":
            for i,r in enumerate(rows):
                with np.load(OUT/"passable_walkway/frozen/frames"/(r["token"]+".npz")) as ref:
                    assert np.array_equal(ground[i],ref["ground"])
                    assert np.array_equal(walk[i],ref["walk"])
                    blocked[i]=ref["blocked"]
        np.savez_compressed(path,tokens=[r["token"] for r in rows],geometry=np.stack(geometries),y=y,
                            ground=ground,blocked=blocked,walk=np.stack(walk))
        print("Prepared",split,len(rows),flush=True)


@torch.inference_mode()
def extract(rank,workers):
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet3d.apis.inference import convert_SyncBN
    from mmdet3d.registry import MODELS
    from mmdet3d.structures import LiDARInstance3DBoxes
    from mmdet3d.utils import register_all_modules
    import projects.BEVFusion.bevfusion
    MODALITIES.mkdir(parents=True,exist_ok=True)
    cfg=Config.fromfile(str(PROJECT/"bevfusion_full_val.py"))
    register_all_modules();init_default_scope("mmdet3d");convert_SyncBN(cfg.model)
    model=MODELS.build(cfg.model)
    load_checkpoint(model,str(PROJECT/"bevfusion-pretrained.pth"),map_location="cpu")
    model.cuda().eval()
    pipe=Compose([p for p in cfg.test_pipeline if p.type!="LoadPointsFromMultiSweeps"])
    splits=manifests();all_rows=sum(splits.values(),[])
    todo=[r for r in all_rows if not (CACHE/(r["token"]+".npy")).exists()]
    rows=splits["train"][:1]+todo[rank::workers]
    for i,r in enumerate(rows):
        dest=MODALITIES/(r["token"]+".npy")
        if dest.exists():continue
        entry=dict(images=r["images"],lidar_points=dict(lidar_path=r["lidar"],num_pts_feats=5),
                   sample_idx=r["sample"],box_type_3d=LiDARInstance3DBoxes)
        inp=model.data_preprocessor(pseudo_collate([pipe(entry)]),training=False)
        data=inp["inputs"];pts=data["points"];pts[0][:,4]=0
        img=data["imgs"];meta=inp["data_samples"][0].metainfo
        def metadata(key,default=None):return img.new_tensor(np.asarray([meta.get(key,default)]))
        args=(metadata("lidar2img"),metadata("cam2img"),metadata("cam2lidar"),
              metadata("img_aug_matrix"),metadata("lidar_aug_matrix",np.eye(4)),[meta])
        captured=[]
        hook=model.view_transform.register_forward_pre_hook(lambda module,args:captured.append(args))
        model.extract_img_feat(img,[pts[0].clone()],*args);hook.remove()
        cam=model.view_transform(captured.pop()[0],[pts[0][:0]],*args)
        lidar=model.extract_pts_feat(data)
        feature=F.grid_sample(torch.cat([cam,lidar],1),sampling_grid(np.array(r["ego_from_lidar"])),align_corners=False)[0].half().cpu().numpy()
        assert feature.shape==(336,64,64) and np.isfinite(feature).all()
        if i==0:
            old=np.load(CACHE/(r["token"]+".npy"),mmap_mode="r")[:336]
            assert np.allclose(feature,old,atol=.03,rtol=.03)
            print("Replay",rank,float(abs(feature-old).max()),flush=True)
            continue
        tmp=dest.with_suffix(f".{rank}.tmp")
        with tmp.open("wb") as f:np.save(f,feature)
        tmp.replace(dest)
        if i%50==0:print("Extract",rank,i,len(rows)-1,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("job",choices=("prepare","extract"))
    p.add_argument("--rank",type=int,default=0);p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    if a.job=="prepare":prepare()
    else:extract(a.rank,a.workers)
