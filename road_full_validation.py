"""Build BEV tensors for the official val split and audit errors on narrow regions."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.splits import val
from pyquaternion import Quaternion

from road import DATA, OUT, MAP_RES, centers, encode, raster
from road_scale_data import DEST


WORK = DEST.parent/"road_refinement"


def prepare():
    WORK.mkdir(exist_ok=True)
    nusc=NuScenes("v1.0-trainval",str(DATA),verbose=False)
    scenes=[s for s in nusc.scene if s["name"] in val]
    maps={r:raster(r) for r in {nusc.get("log",s["log_token"])["location"] for s in scenes}}
    records=[]
    for s in sorted(scenes,key=lambda v:v["name"]):
        region=nusc.get("log",s["log_token"])["location"]
        t=s["first_sample_token"]
        while t:
            sample=nusc.get("sample",t)
            sd=nusc.get("sample_data",sample["data"]["LIDAR_TOP"])
            records.append((s["name"],s["log_token"],region,sd))
            t=sample["next"]
    def frame(record):
        _,_,region,sd=record
        c=nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
        x=encode(DATA/sd["filename"],c)
        e=nusc.get("ego_pose",sd["ego_pose_token"])
        rot=Quaternion(e["rotation"]).rotation_matrix
        yaw=np.arctan2(rot[1,0],rot[0,0]);co,si=np.cos(yaw),np.sin(yaw)
        u,v=np.rint((centers@np.array([[co,si],[-si,co]])+e["translation"][:2])/MAP_RES).astype(int).transpose(2,0,1)
        m=maps[region]
        valid=(u>=0)&(v>=0)&(u<m.shape[1])&(v<m.shape[0])
        y=np.full(u.shape,255,np.uint8)
        y[valid]=m[v[valid],u[valid]]
        return x,y
    assert len(records)==6019 and len(scenes)==150
    with ThreadPoolExecutor(max_workers=12) as pool:
        pairs=list(pool.map(frame,records))
    x,y=map(np.stack,zip(*pairs))
    tokens=[v[-1]["token"] for v in records]
    ids={t:i for i,t in enumerate(tokens)}
    assert len(ids)==len(records)
    with np.load(OUT/"val.npz") as prior:
        idx=[ids[t] for t in prior["token"]]
        assert np.array_equal(x[idx],prior["x"]) and np.array_equal(y[idx],prior["y"])
    train_records=json.loads((DEST/"manifest.json").read_text()).values()
    assert {r["log"] for r in train_records}.isdisjoint({r[1] for r in records})
    tmp=WORK/"val_full.tmp.npz"
    np.savez(tmp,x=x,y=y,scene=[r[0] for r in records],log=[r[1] for r in records],
             region=[r[2] for r in records],token=tokens)
    tmp.replace(WORK/"val_full.npz")
    print("Verified",len(records),"frames,",len(scenes),"scenes,",len(idx),"previous tensors identical",flush=True)


if __name__=="__main__":
    prepare()
