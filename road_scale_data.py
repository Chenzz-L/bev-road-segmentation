"""Recover every eligible nuScenes LiDAR keyframe without changing the frozen split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

from road import DATA, OUT, MAP_RES, centers, grid_points, raster
from road_scene_data import WORK


DEST = WORK.parent / "road_scale"
SHARDS = tuple(f"{i:02d}" for i in range(1,11))
KEYS = ("token","scene","log","region")


def make_manifest():
    old = json.loads((WORK/"manifest.json").read_text()).values()
    allowed = set(np.load(OUT/"train.npz")["scene"]) | {r["scene"] for r in old if r["split"]=="extra"}
    nusc = NuScenes("v1.0-trainval",str(DATA),verbose=False)
    records = {}
    for s in nusc.scene:
        if s["name"] not in allowed:
            continue
        log = s["log_token"]
        region = nusc.get("log",log)["location"]
        token = s["first_sample_token"]
        while token:
            sample = nusc.get("sample",token)
            sd = nusc.get("sample_data",sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose",sd["ego_pose_token"])
            cs = nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
            rot = Quaternion(ego["rotation"]).rotation_matrix
            records[sd["filename"]] = dict(token=sd["token"],scene=s["name"],log=log,region=region,
                ego=ego["translation"][:2],yaw=float(np.arctan2(rot[1,0],rot[0,0])),
                translation=cs["translation"],rotation=cs["rotation"])
            token = sample["next"]
    assert len(allowed)==489 and len(records)==19664
    (DEST/"manifest.json").write_text(json.dumps(records))
    print("Manifest",len(records),len(allowed),flush=True)


def scan(shard, dest=DEST):
    records = json.loads((dest/"manifest.json").read_text())
    blob = "v1.0-trainval"+shard+"_blobs.tgz"
    recovery = DATA.parent/"BEVFusion/full_val/archive_recovery"/blob
    path = recovery if recovery.is_file() else DATA.parent/blob
    maps,rows,features,labels = {},[],[],[]
    proc = subprocess.Popen(["pigz","-dc",str(path)],stdout=subprocess.PIPE)
    entered = False
    try:
        with tarfile.open(fileobj=proc.stdout,mode="r|",bufsize=1024*1024) as archive:
            for entry in archive:
                name = entry.name.lstrip("./")
                if entered and not name.startswith("samples/LIDAR_TOP/"):
                    break
                if not name.startswith("samples/LIDAR_TOP/") or not entry.isfile():
                    continue
                entered = True
                r = records.get(name)
                if r is None:
                    continue
                raw = np.frombuffer(archive.extractfile(entry).read(),np.float32).reshape(-1,5)
                assert len(raw)>0 and np.isfinite(raw).all(),name
                xyz = raw[:,:3]@Quaternion(r["rotation"]).rotation_matrix.T + r["translation"]
                x = grid_points(xyz,raw[:,3])
                if r["region"] not in maps:
                    maps[r["region"]] = raster(r["region"])
                m = maps[r["region"]]
                co,si = np.cos(r["yaw"]),np.sin(r["yaw"])
                world = centers @ np.array([[co,si],[-si,co]]) + r["ego"]
                u,v = np.rint(world/MAP_RES).astype(int).transpose(2,0,1)
                valid = (u>=0)&(v>=0)&(u<m.shape[1])&(v<m.shape[0])
                y = np.full(u.shape,255,np.uint8)
                y[valid] = m[v[valid],u[valid]]
                assert np.any(valid) and np.isfinite(x).all()
                rows.append(r); features.append(x); labels.append(y)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    assert entered
    temp = dest/("part"+shard+".tmp.npz")
    np.savez(temp,x=np.asarray(features,dtype=np.float16).reshape(-1,4,96,96),
             y=np.asarray(labels,dtype=np.uint8).reshape(-1,96,96),**{k:[r[k] for r in rows] for k in KEYS})
    temp.replace(dest/("part"+shard+".npz"))
    print(shard,len(rows),str(path),flush=True)


def verify():
    records = json.loads((DEST/"manifest.json").read_text()).values()
    expected = {r["token"] for r in records}
    prior = np.load(WORK/"full.npz")
    old_x,old_y = prior["x"],prior["y"]
    prior_index = {t:i for i,t in enumerate(prior["token"])}
    verified,seen,regions,scenes,logs = set(),set(),Counter(),set(),set()
    for s in SHARDS:
        with np.load(DEST/("part"+s+".npz")) as d:
            tokens = d["token"]
            assert len(tokens)==len(set(tokens)) and not seen.intersection(tokens)
            seen.update(tokens);regions.update(d["region"]);scenes.update(d["scene"]);logs.update(d["log"])
            x,y = d["x"],d["y"]
            for i,t in enumerate(tokens):
                if t in prior_index:
                    j = prior_index[t]
                    assert np.array_equal(x[i],old_x[j]) and np.array_equal(y[i],old_y[j]),t
                    verified.add(t)
    assert seen==expected and len(verified)==len(prior_index), (len(expected-seen),len(seen-expected),len(verified))
    tbl=json.loads((DATA/"v1.0-trainval/scene.json").read_text())
    by_name={s["name"]:s["log_token"] for s in tbl}
    shared={}
    for name,p in (("dev",OUT/"dev.npz"),("val",OUT/"val.npz"),("fresh",WORK/"fresh.npz")):
        with np.load(p) as d:
            assert not scenes.intersection(d["scene"]) and not seen.intersection(d["token"])
            shared[name]=len(logs & {by_name[v] for v in d["scene"]})
            if name!="dev":assert shared[name]==0
    report=dict(frames=len(seen),scenes=len(scenes),logs=len(logs),regions=regions,exact_old_frames=len(verified),shared_logs=shared)
    (DEST/"inventory.json").write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("job",choices=("manifest","verify",*SHARDS))
    job=parser.parse_args().job
    DEST.mkdir(exist_ok=True)
    if job=="manifest":make_manifest()
    elif job=="verify":verify()
    else:scan(job)
