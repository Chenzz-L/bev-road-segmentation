from road import MAP_ROOT
"""Label ego-grid centers by exact polygon membership and preserve the original valid mask."""
import argparse
import json
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from shapely.prepared import prep
from shapely.vectorized import contains

from road import DATA, ROOT, OUT, centers
from road_full_validation import WORK
from road_scale_data import DEST, SHARDS
from road_scale_train import CACHE
from road_scene_data import WORK as PREVIOUS
from audit_road_density import boundary_band
from road_fine_metrics import narrow_cells


VECTOR=WORK/"vector_labels"
SPLITS=("dev","full","held","new","train")


def paths(split):
    if split=="train":return [DEST/f"part{s}.npz" for s in SHARDS]
    return [{"dev":OUT/"dev.npz","full":WORK/"val_full.npz",
             "held":PREVIOUS/"fresh.npz","new":WORK/"new_logs/test.npz"}[split]]


def poses():
    tokens=set()
    for split in SPLITS:
        if split=="train":continue
        for p in paths(split):
            if p.exists():tokens.update(np.load(p)["token"])
    # Include planned test tokens even if the extraction worker is still running.
    extra=WORK/"new_logs/manifest.json"
    if extra.exists():tokens.update(r["token"] for r in json.loads(extra.read_text()).values())
    meta=DATA/"v1.0-trainval"
    sd={s["token"]:s["ego_pose_token"] for s in json.loads((meta/"sample_data.json").read_text()) if s["token"] in tokens}
    used=set(sd.values())
    p={s["token"]:s for s in json.loads((meta/"ego_pose.json").read_text()) if s["token"] in used}
    lookup={}
    for token,ego in sd.items():
        v=p[ego]; rot=Quaternion(v["rotation"]).rotation_matrix
        lookup[token]=[*v["translation"][:2],float(np.arctan2(rot[1,0],rot[0,0]))]
    for r in json.loads((DEST/"manifest.json").read_text()).values():lookup[r["token"]]=[*r["ego"],r["yaw"]]
    (VECTOR/"poses.json").write_text(json.dumps(lookup))


class VectorMap:
    def __init__(self,region,map_root=MAP_ROOT):
        api=NuScenesMap(dataroot=str(map_root),map_name=region)
        self.canvas=api.canvas_edge
        # Some map records have a null token that the installed devkit resolves
        # to a real polygon. Preserve exactly the same lookup as record_on_point.
        tokens=dict.fromkeys(t for r in api.drivable_area for t in r["polygon_tokens"])
        polygons=[api.extract_polygon(t) for t in tokens]
        self.bounds=np.array([p.bounds for p in polygons])
        self.prepared=[prep(p) for p in polygons]

    def label(self,pose,valid=None):
        tx,ty,yaw=pose; co,si=np.cos(yaw),np.sin(yaw)
        xx=centers[:,:,0]*co-centers[:,:,1]*si+tx
        yy=centers[:,:,0]*si+centers[:,:,1]*co+ty
        if valid is None:valid=(xx>=0)&(yy>=0)&(xx<self.canvas[0])&(yy<self.canvas[1])
        x0,y0,x1,y1=self.bounds.T
        candidates=(x1>=xx.min())&(x0<=xx.max())&(y1>=yy.min())&(y0<=yy.max())
        road=np.zeros(valid.shape,dtype=bool)
        for i in np.flatnonzero(candidates):
            inside=(xx>=x0[i])&(xx<=x1[i])&(yy>=y0[i])&(yy<=y1[i])&valid&~road
            if inside.any():
                road[inside]=contains(self.prepared[i],xx[inside],yy[inside])
        y=road.astype("uint8");y[~valid]=255
        return y


def generate(split):
    lookup=json.loads((VECTOR/"poses.json").read_text())
    files=paths(split)
    n=sum(len(np.load(p)["token"]) for p in files)
    tmp=VECTOR/(split+"_y.tmp.npy")
    result=np.lib.format.open_memmap(tmp,mode="w+",dtype="uint8",shape=(n,96,96))
    start=0;changed=valid_count=0;maps={}
    for p in files:
        with np.load(p) as d:
            tokens,regions,old=d["token"],d["region"],d["y"]
        for j,(t,r) in enumerate(zip(tokens,regions)):
            if r not in maps:maps[r]=VectorMap(r)
            valid=old[j]!=255
            out=maps[r].label(lookup[t],valid)
            changed+=int((out!=old[j]).sum());valid_count+=int(valid.sum())
            result[start+j]=out
        start+=len(tokens)
    assert start==n
    result.flush(); del result
    tmp.replace(VECTOR/(split+"_y.npy"))
    if split=="train":
        y=np.load(VECTOR/"train_y.npy",mmap_mode="r")
        edge=np.lib.format.open_memmap(VECTOR/"train_edge.npy",mode="w+",dtype=bool,shape=y.shape)
        thin=np.lib.format.open_memmap(VECTOR/"train_thin.npy",mode="w+",dtype=bool,shape=y.shape)
        for i in range(0,len(y),128):
            edge[i:i+128]=boundary_band(y[i:i+128])
            thin[i:i+128]=np.stack([narrow_cells(f) for f in y[i:i+128]])
        edge.flush();thin.flush()
    report=dict(frames=n,valid_cells=valid_count,changed=changed,changed_pct=100*changed/valid_count)
    (VECTOR/(split+"_audit.json")).write_text(json.dumps(report,indent=2))
    print(split,report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("poses",*SPLITS));a=p.parse_args()
    VECTOR.mkdir(exist_ok=True)
    if a.action=="poses":poses()
    else:generate(a.action)
