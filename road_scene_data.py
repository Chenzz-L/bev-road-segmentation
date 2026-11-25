"""Build log-disjoint supplemental LiDAR training and confirmation sets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.splits import train
from pyquaternion import Quaternion

from road import DATA, OUT, centers, grid_points, raster, MAP_RES

WORK = OUT.parent / "road_generalization"
SHARDS = ("03","05","07","08","10")


def digest(x):
    return hashlib.sha256(x.encode()).hexdigest()


def manifest():
    nusc = NuScenes("v1.0-trainval",str(DATA),verbose=False)
    sets = {s:set(np.load(OUT/(s+".npz"))["scene"]) for s in ("train","dev","val")}
    blocked = {s["log_token"] for s in nusc.scene if any(s["name"] in v for v in sets.values())}
    candidates = [s for s in nusc.scene if s["name"] in train and s["log_token"] not in blocked]
    logs = {s["log_token"]:nusc.get("log",s["log_token"])["location"] for s in candidates}
    held = set()
    for region in set(logs.values()):
        ordered = sorted((l for l,r in logs.items() if r==region),key=digest)
        held.update(ordered[::4])
    records = {}
    for scene in candidates:
        tok = scene["first_sample_token"]
        tokens = []
        while tok:
            sample = nusc.get("sample",tok)
            tokens.append(sample["data"]["LIDAR_TOP"])
            tok = sample["next"]
        for j in np.linspace(0,len(tokens)-1,8,dtype=int):
            sd = nusc.get("sample_data",tokens[j])
            pose = nusc.get("ego_pose",sd["ego_pose_token"])
            cs = nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
            rot = Quaternion(pose["rotation"]).rotation_matrix
            log = scene["log_token"]
            records[sd["filename"]] = dict(token=sd["token"], scene=scene["name"],log=log,region=logs[log],
                split="fresh" if log in held else "extra",ego=pose["translation"][:2],
                yaw=float(np.arctan2(rot[1,0],rot[0,0])),
                rotation=cs["rotation"],translation=cs["translation"])
    (WORK/"manifest.json").write_text(json.dumps(records))
    print("Candidate samples",len(records),"new logs",len(logs),"confirmation logs",len(held),flush=True)


def scan(shard):
    records = json.loads((WORK/"manifest.json").read_text())
    maps = {}
    samples = []
    proc = subprocess.Popen(["pigz","-dc",str(DATA.parent/("v1.0-trainval"+shard+"_blobs.tgz"))],
                            stdout=subprocess.PIPE)
    entered = False
    seen = 0
    try:
        with tarfile.open(fileobj=proc.stdout,mode="r|",bufsize=1024*1024) as archive:
            for entry in archive:
                path = entry.name.lstrip("./")
                if entered and not path.startswith("samples/LIDAR_TOP/"):
                    break
                if not path.startswith("samples/LIDAR_TOP/") or not entry.isfile():
                    continue
                entered = True
                seen += 1
                if path not in records:
                    continue
                r = records[path]
                raw = np.frombuffer(archive.extractfile(entry).read(),np.float32).reshape(-1,5)
                xyz = raw[:,:3] @ Quaternion(r["rotation"]).rotation_matrix.T + r["translation"]
                x = grid_points(xyz,raw[:,3])
                region = r["region"]
                if region not in maps:
                    maps[region] = raster(region)
                m = maps[region]
                co,si = np.cos(r["yaw"]),np.sin(r["yaw"])
                glob = centers @ np.array([[co,si],[-si,co]]) + r["ego"]
                uv = np.rint(glob/MAP_RES).astype(int)
                u,v = uv[:,:,0],uv[:,:,1]
                valid = (u>=0)&(u<m.shape[1])&(v>=0)&(v<m.shape[0])
                y = np.full(u.shape,255,dtype="uint8")
                y[valid] = m[v[valid],u[valid]]
                samples.append((r,x,y))
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    assert seen>500 and samples,(shard,seen)
    # Confirm that each contributing scene was read in full.
    counts = np.unique([r["scene"] for r,x,y in samples],return_counts=True)[1]
    assert np.all(counts==8),(shard,counts)
    values = {k:[r[k] for r,x,y in samples] for k in ("split","scene","region","log","token")}
    np.savez_compressed(WORK/("shard"+shard+".npz"),x=[x for r,x,y in samples],
                        y=[y for r,x,y in samples],**values)
    print("Finished",shard,len(samples),len(counts),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("job",choices=("manifest",*SHARDS))
    job=parser.parse_args().job
    WORK.mkdir(exist_ok=True)
    if job=="manifest":
        manifest()
    else:
        scan(job)
