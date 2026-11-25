"""Extract frozen BEVFusion tensors on a common ego grid from one synchronized sample."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from road import DATA, OUT as ROAD, SHARD
from road_camera import CAMERAS, WORK as CAMERA, pose
from road_full_validation import WORK


OUT = WORK/"shared_bev"
CACHE = Path(os.environ.get("ROAD_FEATURE_CACHE", str(OUT/"feature_cache")))
PROJECT = Path(os.environ.get("BEVFUSION_PROJECT", DATA.parent/"BEVFusion"))
MODES = ("camera","lidar","fusion")


def manifest():
    from nuscenes import NuScenes
    from mmengine import load
    nusc = NuScenes("v1.0-trainval",str(DATA),verbose=False)
    prior = {r["lidar_token"] for r in json.loads((WORK/"obstacle_boxes/frames.json").read_text())}
    out = []
    for split in ("train","dev","val"):
        with np.load(ROAD/f"{split}.npz") as d:
            tokens,scenes = d["token"],d["scene"]
        for i,(token,scene) in enumerate(zip(tokens,scenes)):
            if split=="val" and token not in prior:
                continue
            sd = nusc.get("sample_data",token)
            sample = nusc.get("sample",sd["sample_token"])
            def transform(sd,table):
                return pose(nusc.get(table,sd[table+"_token"]))
            calib = transform(sd,"calibrated_sensor")
            world = transform(sd,"ego_pose") @ calib
            lidar = DATA/sd["filename"]
            if not lidar.is_file():
                lidar = SHARD/lidar.name
            assert lidar.is_file(), lidar
            images = {}
            for name in CAMERAS:
                cam = nusc.get("sample_data",sample["data"][name])
                path = DATA/cam["filename"]
                if not path.is_file():
                    path = CAMERA/"rgb"/path.name
                assert path.is_file(), path
                cam_world = transform(cam,"ego_pose") @ transform(cam,"calibrated_sensor")
                images[name] = dict(img_path=str(path),lidar2cam=np.linalg.solve(cam_world,world).tolist(),
                                    cam2img=nusc.get("calibrated_sensor",cam["calibrated_sensor_token"])["camera_intrinsic"])
            out.append(dict(token=str(token),sample=sample["token"],scene=str(scene),index=i,split=split,
                            lidar=str(lidar),ego_from_lidar=calib.tolist(),images=images))
    assert [sum(r["split"]==s for r in out) for s in ("train","dev","val")] == [624,120,542]
    ref = next(r for r in out if r["split"]=="val")
    entry = next(e for e in load(str(PROJECT/"full_val/nuscenes_infos_val.pkl"))["data_list"] if e["token"]==ref["sample"])
    for name in CAMERAS:
        assert np.allclose(entry["images"][name]["lidar2cam"],ref["images"][name]["lidar2cam"],atol=1e-4)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"manifest.json").write_text(json.dumps(out))
    print("Verified camera transforms and split sizes",len(out),flush=True)


def sampling_grid(ego):
    axis = (np.arange(64)+.5)*.75-24
    xx,yy = np.meshgrid(axis,axis)
    points = np.stack([xx,yy,np.zeros_like(xx),np.ones_like(xx)],axis=-1)
    lidar = points @ np.linalg.inv(ego).T
    # BEVFusion pool stores X as tensor height and Y as width.
    grid = torch.tensor(lidar[:,:,[1,0]]/54,dtype=torch.float32,device="cuda")[None]
    # Verify axis order and metric calibration using an analytic coordinate field.
    a = (torch.arange(180,device="cuda")+.5)*.6-54
    x,y = torch.meshgrid(a,a,indexing="ij")
    field = torch.stack([x,y])[None]
    rec = F.grid_sample(field,grid,align_corners=False)[0].permute(1,2,0).cpu().numpy()
    assert np.max(abs(rec-lidar[:,:,:2]))<2e-5
    return grid


@torch.inference_mode()
def extract(rank,workers,smoke):
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet3d.apis.inference import convert_SyncBN
    from mmdet3d.registry import MODELS
    from mmdet3d.structures import LiDARInstance3DBoxes
    from mmdet3d.utils import register_all_modules
    import projects.BEVFusion.bevfusion
    cfg = Config.fromfile(str(PROJECT/"bevfusion_full_val.py"))
    register_all_modules()
    init_default_scope("mmdet3d")
    convert_SyncBN(cfg.model)
    model = MODELS.build(cfg.model)
    load_checkpoint(model,str(PROJECT/"bevfusion-pretrained.pth"),map_location="cpu")
    model.cuda().eval()
    pipeline = Compose([p for p in cfg.test_pipeline if p.type!="LoadPointsFromMultiSweeps"])
    rows = json.loads((OUT/"manifest.json").read_text())[rank::workers]
    CACHE.mkdir(exist_ok=True)
    grids = {}
    for j,r in enumerate(rows):
        dest = CACHE/(r["token"]+".npy")
        if dest.exists():
            if smoke: break
            continue
        entry = dict(images=r["images"],lidar_points=dict(lidar_path=r["lidar"],num_pts_feats=5),
                     sample_idx=r["sample"],box_type_3d=LiDARInstance3DBoxes)
        inp = model.data_preprocessor(pseudo_collate([pipeline(entry)]),training=False)
        data = inp["inputs"]
        pts = data["points"]
        pts[0][:,4] = 0  # Current sweep time lag; raw column 5 contains ring index.
        img = data["imgs"]
        meta = inp["data_samples"][0].metainfo
        def metadata(key,default=None):
            return img.new_tensor(np.asarray([meta.get(key,default)]))
        args = (metadata("lidar2img"),metadata("cam2img"),metadata("cam2lidar"),
                metadata("img_aug_matrix"),metadata("lidar_aug_matrix",np.eye(4)),[meta])
        # Reuse the image backbone. Pure camera gets an empty sparse-depth input;
        # the normal fusion path receives projected LiDAR depths as in the checkpoint.
        captured = []
        def save_args(module,inputs):
            captured.append(inputs)
        hook = model.view_transform.register_forward_pre_hook(save_args)
        cam_depth = model.extract_img_feat(img,[pts[0].clone()],*args)
        hook.remove()
        img_features = captured.pop()[0]
        cam = model.view_transform(img_features,[pts[0][:0]],*args)
        lidar = model.extract_pts_feat(data)
        fusion = model.fusion_layer([cam_depth,lidar])
        sizes = [f.shape[1] for f in (cam,lidar,fusion)]
        assert sizes==[80,256,256] and all(f.shape[-2:]==(180,180) for f in (cam,lidar,fusion))
        key = tuple(np.asarray(r["ego_from_lidar"]).ravel())
        if key not in grids:
            grids[key] = sampling_grid(np.asarray(r["ego_from_lidar"]))
        feat = F.grid_sample(torch.cat([cam,lidar,fusion],1),grids[key],align_corners=False)[0].half()
        assert torch.isfinite(feat).all() and all(t.abs().max()>0 for t in feat.split(sizes))
        tmp = dest.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.save(f,feat.cpu().numpy())
        tmp.replace(dest)
        if smoke or (j+1)%100==0:
            print("worker",rank,"frames",j+1,"aligned",tuple(feat.shape),flush=True)
        if smoke: break


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("job",choices=("manifest","extract"))
    p.add_argument("--rank",type=int,default=0)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.job=="manifest": manifest()
    else: extract(a.rank,a.workers,a.smoke)
