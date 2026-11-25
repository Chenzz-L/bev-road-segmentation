"""LiDAR-only BEV segmentation of the static nuScenes drivable_area layer."""
import argparse
import os
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes import NuScenes
from nuscenes.utils.splits import train,val
from pyquaternion import Quaternion
from torch import nn
import torch.nn.functional as F

ROOT=Path(os.environ.get('ROAD_WORK_DIR', Path(__file__).resolve().parent))
OUT=ROOT/'road'
SHARD=Path(os.environ.get('ROAD_LIDAR_SHARD', ROOT/'lidar_shard01'))
DATA=Path(os.environ.get('NUSCENES_ROOT', ROOT.parent/'nuscenes_trainval_val'))
MAP_ROOT=Path(os.environ.get('NUSCENES_MAP_ROOT', DATA))
N=96
CELL=.5
MAP_RES=.25
limit=N*CELL/2
axis=(np.arange(N)+.5)*CELL-limit
gx,gy=np.meshgrid(axis,axis)
centers=np.stack([gx,gy],axis=-1)


def raster(location):
    m=json.loads((MAP_ROOT/'maps/expansion'/(location+'.json')).read_text())
    nodes={r['token']:(r['x'],r['y']) for r in m['node']}
    polygons={p['token']:p for p in m['polygon']}
    wh=np.ceil(np.asarray(m['canvas_edge'])/MAP_RES).astype(int)+1
    out=np.zeros(tuple(wh[::-1]),np.uint8)
    for token in {t for d in m['drivable_area'] for t in d['polygon_tokens']}:
        poly=polygons[token]
        rings=[poly['exterior_node_tokens']]+[h['node_tokens'] for h in poly['holes']]
        pts=[np.rint(np.array([nodes[t] for t in ring])/MAP_RES).astype(np.int32) for ring in rings if len(ring)>=3]
        if not pts: continue
        lo=np.maximum(pts[0].min(0),0)
        hi=np.minimum(pts[0].max(0)+1,wh)
        if np.any(hi<=lo): continue
        local=np.zeros(tuple((hi-lo)[::-1]),np.uint8)
        cv2.fillPoly(local,[pts[0]-lo],1)
        for hole in pts[1:]:
            cv2.fillPoly(local,[hole-lo],0)
        out[lo[1]:hi[1],lo[0]:hi[0]] |= local
    return out


def grid_points(xyz,intensities):
    idx=np.floor((xyz[:,:2]+limit)/CELL).astype(int)
    ok=(idx>=0).all(1)&(idx<N).all(1)&(xyz[:,2]>-3)&(xyz[:,2]<6)
    keys=idx[ok,1]*N+idx[ok,0]
    count=np.bincount(keys,minlength=N*N)
    low=np.full(N*N,6.)
    high=np.full(N*N,-3.)
    np.minimum.at(low,keys,xyz[ok,2]); np.maximum.at(high,keys,xyz[ok,2])
    low[count==0]=0; high[count==0]=0
    intensity=np.bincount(keys,weights=intensities[ok],minlength=N*N)/np.maximum(count,1)
    return np.stack((np.log1p(count)/3,low/3,high/3,intensity/255)).reshape(4,N,N).astype(np.float16)


def encode(path,calib):
    raw=np.fromfile(path,np.float32).reshape(-1,5)
    xyz=raw[:,:3]@Quaternion(calib['rotation']).rotation_matrix.T+calib['translation']
    return grid_points(xyz,raw[:,3])


def prepare():
    OUT.mkdir(exist_ok=True)
    assert (SHARD/'READY').exists()
    files={p.name for p in SHARD.glob('*.bin')}
    nusc=NuScenes('v1.0-trainval',str(DATA),verbose=False)
    maps={}
    by_scene={}
    for sc in nusc.scene:
        name=sc['name']; seq=[]; token=sc['first_sample_token']
        while token:
            s=nusc.get('sample',token)
            sd=nusc.get('sample_data',s['data']['LIDAR_TOP'])
            if name in val or (name in train and Path(sd['filename']).name in files):
                seq.append(sd)
            token=s['next']
        if seq:
            by_scene[name]=(nusc.get('log',sc['log_token'])['location'],seq)
    tr=sorted(set(train)&set(by_scene),key=lambda n:hashlib.sha256(n.encode()).hexdigest())
    dev=set(tr[:max(6,len(tr)//6)])
    assert len(tr)>len(dev)+10 and set(val)<=set(by_scene)
    for sp,names in [('train',set(tr)-dev),('dev',dev),('val',set(val))]:
        X=[];Y=[];all_sc=[];regions=[];tokens=[]
        for name in sorted(names):
            loc,seq=by_scene[name]
            if loc not in maps: maps[loc]=raster(loc)
            m=maps[loc]
            for j in np.linspace(0,len(seq)-1,min(len(seq),4 if sp=='val' else 12),dtype=int):
                sd=seq[j]
                path=DATA/sd['filename'] if sp=='val' else SHARD/Path(sd['filename']).name
                c=nusc.get('calibrated_sensor',sd['calibrated_sensor_token'])
                e=nusc.get('ego_pose',sd['ego_pose_token'])
                rot=Quaternion(e['rotation']).rotation_matrix
                yaw=np.arctan2(rot[1,0],rot[0,0]); co,si=np.cos(yaw),np.sin(yaw)
                glob=centers@np.array([[co,si],[-si,co]])+e['translation'][:2]
                uv=np.rint(glob/MAP_RES).astype(int)
                valid=(uv[:,:,0]>=0)&(uv[:,:,1]>=0)&(uv[:,:,0]<m.shape[1])&(uv[:,:,1]<m.shape[0])
                mask=np.full((N,N),255,np.uint8)
                mask[valid]=m[uv[:,:,1][valid],uv[:,:,0][valid]]
                X.append(encode(path,c)); Y.append(mask)
                all_sc.append(name);regions.append(loc);tokens.append(sd['token'])
        np.savez_compressed(OUT/(sp+'.npz'),x=X,y=Y,scene=all_sc,region=regions,token=tokens)
        print(sp,len(names),len(X),flush=True)


def block(i,o):
    return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.GroupNorm(4,o),nn.ReLU(),
                         nn.Conv2d(o,o,3,padding=1),nn.GroupNorm(4,o),nn.ReLU())


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=block(6,16); self.b=block(16,24); self.c=block(24,40)
        self.upb=block(64,24); self.upa=block(40,16); self.out=nn.Conv2d(16,1,1)

    def forward(self,x):
        a=self.a(x); b=self.b(F.max_pool2d(a,2)); c=self.c(F.max_pool2d(b,2))
        b=self.upb(torch.cat((F.interpolate(c,scale_factor=2,mode='nearest'),b),1))
        return self.out(self.upa(torch.cat((F.interpolate(b,scale_factor=2,mode='nearest'),a),1)))


def counts(pred,y):
    return torch.stack([(pred & (y==1)).sum((1,2)),(pred & (y==0)).sum((1,2)),
                        (~pred & (y==1)).sum((1,2)),(~pred & (y==0)).sum((1,2))],1)


def metric(c):
    tp,fp,fn,tn=c.sum(0).tolist()
    road=tp/max(tp+fp+fn,1)
    bg=tn/max(tn+fp+fn,1)
    return dict(road_IoU=road,background_IoU=bg,mIoU=(road+bg)/2)


def train_model(kind):
    torch.manual_seed(17)
    sets={}
    for sp in ('train','dev','val'):
        data=np.load(OUT/(sp+'.npz'))
        input_file=OUT/(sp+'_history.npz')
        source=np.load(input_file)[kind] if kind in ('aligned','unaligned') else data['x']
        x=torch.from_numpy(source).cuda().float()
        coords=torch.from_numpy(np.stack((gx,gy))/limit).cuda().float().expand(len(x),-1,-1,-1)
        sets[sp]=(torch.cat((x,coords),1),torch.from_numpy(data['y']).cuda(),data['scene'],data['region'])
    if kind in ('unet','aligned','unaligned'): net=UNet()
    elif kind=='local': net=nn.Sequential(block(6,24),block(24,24),nn.Conv2d(24,1,1))
    else: net=nn.Sequential(nn.Conv2d(6,32,1),nn.ReLU(),nn.Conv2d(32,32,1),nn.ReLU(),nn.Conv2d(32,1,1))
    net=net.cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    x,y,_,_=sets['train']
    prior=(y==1).sum(0)/(y!=255).sum(0).clamp_min(1)
    best=-1
    for epoch in range(18):
        net.train()
        for ids in torch.randperm(len(x),device='cuda').split(20):
            logits=net(x[ids])[:,0]; target=y[ids]
            valid=target!=255
            loss=F.binary_cross_entropy_with_logits(logits[valid],target[valid].float())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            net.eval()
            dx,dy,_,_=sets['dev']
            scores=torch.cat([net(t)[:,0].sigmoid() for t in dx.split(20)])
            for threshold in (.35,.4,.45,.5,.55,.6,.65):
                value=metric(counts(scores>=threshold,dy))['mIoU']
                if value>best:
                    best=value
                    torch.save(dict(weights=net.state_dict(),epoch=epoch+1,threshold=threshold),OUT/(kind+'.pt'))
        if epoch in (0,5,11,17): print(kind,epoch+1,best,flush=True)
    ckpt=torch.load(OUT/(kind+'.pt'))
    net.load_state_dict(ckpt['weights'])
    vx,vy,scenes,regions=sets['val']
    prior_t=max((.35,.4,.45,.5,.55,.6,.65),key=lambda t:metric(counts(prior>=t,sets['dev'][1]))['mIoU'])
    with torch.no_grad():
        vs=torch.cat([net(t)[:,0].sigmoid() for t in vx.split(20)])
        cn=counts(vs>=ckpt['threshold'],vy)
        report=dict(kind=kind,epoch=ckpt['epoch'],threshold=ckpt['threshold'],gpu=torch.cuda.get_device_name(),
                    splits={k:dict(frames=len(v[0]),scenes=len(set(v[2]))) for k,v in sets.items()},
                    val=metric(cn),dev_mIoU=best,prior=metric(counts(prior>=prior_t,vy)),prior_threshold=prior_t,
                    by_region={r:metric(cn[np.flatnonzero(regions==r)]) for r in sorted(set(regions))})
    np.savez_compressed(OUT/(kind+'_counts.npz'),scene=scenes,counts=cn.cpu().numpy())
    (OUT/(kind+'.json')).write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['prepare','pixel','local','unet','aligned','unaligned'])
    arg=p.parse_args().job
    if arg=='prepare':prepare()
    else:train_model(arg)
