"""Frozen 52-to-200 scene expansion, sensor recovery, and fusion-only caching."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.nn import functional as F

from road import ROOT,DATA
from road_camera import CAMERAS,pose
from road_shared_features import OUT,CACHE,PROJECT,sampling_grid


DEST=OUT/"train200"
FEATURES=ROOT/"road_refinement/shared_bev/train200/features"


def manifest():
    DEST.mkdir(exist_ok=True)
    old=json.loads((OUT/"manifest.json").read_text())
    base=[r for r in old if r["split"]=="train"]
    originals={r["scene"] for r in base}
    scene_table=json.loads((DATA/"v1.0-trainval/scene.json").read_text())
    scenes={s["name"]:s for s in scene_table}
    scene_names={s["token"]:s["name"] for s in scene_table}
    log_table={s["token"]:s for s in json.loads((DATA/"v1.0-trainval/log.json").read_text())}
    reserved={r["scene"] for r in old if r["split"]!="train"}
    reserved.update(r["scene"] for r in json.loads((OUT/"passable_walkway/frozen/new_manifest.json").read_text()))
    forbidden_logs={scenes[s]["log_token"] for s in reserved}
    pool={r["scene"] for r in json.loads((ROOT/"road_scale/manifest.json").read_text()).values()}
    candidates=pool-originals-reserved
    candidates={s for s in candidates if scenes[s]["log_token"] not in forbidden_logs}
    region_groups=defaultdict(list)
    for s in candidates:
        region=log_table[scenes[s]["log_token"]]["location"]
        region_groups[region].append(s)
    for group in region_groups.values():
        group.sort(key=lambda s:hashlib.sha256(("train200-v1-"+s).encode()).hexdigest())
    # Round robin regions with deterministic scene ordering, before inspecting results.
    chosen=[]
    for i in range(max(map(len,region_groups.values()))):
        for region in sorted(region_groups):
            if i<len(region_groups[region]):chosen.append(region_groups[region][i])
    chosen=chosen[:148]
    assert len(chosen)==148 and len(originals)==52
    selected=set(chosen)|originals
    assert len(selected)==200 and not selected&reserved
    samples=json.loads((DATA/"v1.0-trainval/sample.json").read_text())
    per_scene=defaultdict(list)
    for s in samples:
        name=scene_names[s["scene_token"]]
        if name in chosen:per_scene[name].append(s)
    needed={}
    for name in chosen:
        ordered=sorted(per_scene[name],key=lambda r:r["timestamp"])
        for idx in np.linspace(0,len(ordered)-1,12,dtype=int):
            needed[ordered[idx]["token"]]=name
    sensors={r["token"]:r["channel"] for r in json.loads((DATA/"v1.0-trainval/sensor.json").read_text())}
    calibrations={r["token"]:r for r in json.loads((DATA/"v1.0-trainval/calibrated_sensor.json").read_text())}
    records=defaultdict(dict)
    for sd in json.loads((DATA/"v1.0-trainval/sample_data.json").read_text()):
        if sd["is_key_frame"] and sd["sample_token"] in needed:
            channel=sensors[calibrations[sd["calibrated_sensor_token"]]["sensor_token"]]
            if channel in (*CAMERAS,"LIDAR_TOP"):records[sd["sample_token"]][channel]=sd
    pose_tokens={r["ego_pose_token"] for d in records.values() for r in d.values()}
    poses={r["token"]:r for r in json.loads((DATA/"v1.0-trainval/ego_pose.json").read_text()) if r["token"] in pose_tokens}
    parts={s["name"]:i//85+1 for i,s in enumerate(sorted(scene_table,key=lambda s:s["name"]))}
    extra=[]
    required=defaultdict(set)
    for token,name in sorted(needed.items(),key=lambda p:(p[1],p[0])):
        rec=records[token]
        assert len(rec)==7
        sd=rec["LIDAR_TOP"]
        calib=pose(calibrations[sd["calibrated_sensor_token"]])
        ego=pose(poses[sd["ego_pose_token"]])
        world=ego@calib
        images={}
        for channel in CAMERAS:
            cam=rec[channel]
            cam_world=pose(poses[cam["ego_pose_token"]])@pose(calibrations[cam["calibrated_sensor_token"]])
            images[channel]=dict(img_path=str(DATA/cam["filename"]),lidar2cam=np.linalg.solve(cam_world,world).tolist(),
                                 cam2img=calibrations[cam["calibrated_sensor_token"]]["camera_intrinsic"])
        for record in rec.values():
            if not (DATA/record["filename"]).is_file():required[parts[name]].add(record["filename"])
        extra.append(dict(token=sd["token"],sample=token,scene=name,split="train",lidar=str(DATA/sd["filename"]),
                          region=log_table[scenes[name]["log_token"]]["location"],log=scenes[name]["log_token"],
                          ego_from_lidar=calib.tolist(),ego_to_global=ego.tolist(),images=images))
    assert len(extra)==1776 and len(base)==624
    (DEST/"manifest.json").write_text(json.dumps(base+extra))
    (DEST/"required.json").write_text(json.dumps({k:sorted(v) for k,v in required.items()}))
    report=dict(original_scenes=52,expanded_scenes=200,original_frames=624,expanded_frames=2400,
                added_regions=Counter(r["region"] for r in extra),required_files={k:len(v) for k,v in required.items()},
                added_scene_log_overlap_with_reserved=0,selection="148 extra scenes, region round-robin, deterministic hash ordering; 12 evenly spaced frames per extra scene",
                settings=dict(head="unchanged RoadHead(256)",loss="BCE",epochs=18,batch_size=24,lr=.001,seeds=[17,29,43],
                              thresholds=[.35,.40,.45,.50,.55,.60,.65],selection="highest dev road IoU"))
    (DEST/"protocol.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


def recover():
    required=json.loads((DEST/"required.json").read_text())
    def extract(item):
        part,files=item
        left={f for f in files if not (DATA/f).is_file()}
        if not left:return
        archive=DATA.parent/f"v1.0-trainval{int(part):02}_blobs.tgz"
        replacement=PROJECT/"full_val/archive_recovery"/archive.name
        if replacement.is_file():archive=replacement
        proc=subprocess.Popen(["pigz","-dc",str(archive)],stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout,mode="r|",bufsize=1024*1024) as source:
                for m in source:
                    name=m.name[2:] if m.name.startswith("./") else m.name
                    if name not in left:continue
                    target=DATA/name
                    target.parent.mkdir(exist_ok=True,parents=True)
                    tmp=target.with_suffix(target.suffix+".train200.part")
                    with source.extractfile(m) as reader,tmp.open("wb") as writer:
                        shutil.copyfileobj(reader,writer,1024*1024)
                    assert tmp.stat().st_size==m.size
                    tmp.replace(target);left.remove(name)
                    if not left:break
        finally:
            proc.stdout.close();proc.terminate();proc.wait()
        assert not left,(part,len(left))
        print("Recovered archive",part,len(files),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(extract,required.items()))


@torch.inference_mode()
def features(rank,workers,evaluation=False):
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet3d.apis.inference import convert_SyncBN
    from mmdet3d.registry import MODELS
    from mmdet3d.structures import LiDARInstance3DBoxes
    from mmdet3d.utils import register_all_modules
    import projects.BEVFusion.bevfusion
    destination=DEST/"eval27_features" if evaluation else FEATURES
    destination.mkdir(exist_ok=True)
    cfg=Config.fromfile(str(PROJECT/"bevfusion_full_val.py"))
    register_all_modules();init_default_scope("mmdet3d");convert_SyncBN(cfg.model)
    model=MODELS.build(cfg.model)
    load_checkpoint(model,str(PROJECT/"bevfusion-pretrained.pth"),map_location="cpu")
    model.cuda().eval()
    pipe=Compose([p for p in cfg.test_pipeline if p.type!="LoadPointsFromMultiSweeps"])
    rows=json.loads((DEST/"manifest.json").read_text())
    # Reuse original 624 cached frames; verify the new extraction path on one.
    extra=json.loads((OUT/"passable_walkway/frozen/new_manifest.json").read_text()) if evaluation else rows[624:]
    work=rows[:1]+extra[rank::workers]
    for j,r in enumerate(work):
        path=destination/(r["token"]+".npy")
        if path.exists():continue
        files=[r["lidar"]]+[v["img_path"] for v in r["images"].values()]
        while not all(Path(p).is_file() for p in files):
            time.sleep(10)
        entry=dict(images=r["images"],lidar_points=dict(lidar_path=r["lidar"],num_pts_feats=5),
                   sample_idx=r["sample"],box_type_3d=LiDARInstance3DBoxes)
        inp=model.data_preprocessor(pseudo_collate([pipe(entry)]),training=False)
        data=inp["inputs"];data["points"][0][:,4]=0
        img=data["imgs"];meta=inp["data_samples"][0].metainfo
        def metadata(key,default=None):return img.new_tensor(np.asarray([meta.get(key,default)]))
        args=(metadata("lidar2img"),metadata("cam2img"),metadata("cam2lidar"),
              metadata("img_aug_matrix"),metadata("lidar_aug_matrix",np.eye(4)),[meta])
        cam=model.extract_img_feat(img,[data["points"][0].clone()],*args)
        lidar=model.extract_pts_feat(data)
        fused=model.fusion_layer([cam,lidar])
        feature=F.grid_sample(fused,sampling_grid(np.array(r["ego_from_lidar"])),align_corners=False)[0].half().cpu().numpy()
        assert feature.shape==(256,64,64) and np.isfinite(feature).all()
        if j==0:
            old=np.load(CACHE/(r["token"]+".npy"),mmap_mode="r")[336:592]
            error=float(np.abs(feature.astype(float)-old.astype(float)).max())
            assert np.allclose(feature,old,atol=.02,rtol=.02),error
            print("Feature replay max error",rank,error,flush=True)
            continue
        tmp=path.with_suffix(f".{rank}.tmp")
        with tmp.open("wb") as f:np.save(f,feature)
        tmp.replace(path)
        if j%50==0:print("Features",rank,j,"/",len(work)-1,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("job",choices=("manifest","recover","features"))
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--evaluation",action="store_true")
    a=parser.parse_args()
    if a.job=="manifest":manifest()
    elif a.job=="recover":recover()
    else:features(a.rank,a.workers,a.evaluation)
