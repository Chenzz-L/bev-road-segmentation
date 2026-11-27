"""Continuous sensor-frame road inference and timestamp-aligned occupancy replay."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import argparse,gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pyquaternion import Quaternion
import road_log_head_control as roadhead
from dynamic_road_core import ROI,predict_occupancy,release,footprint_global,trajectory

ROOT=Path(__file__).resolve().parent
PROJECT=roadhead.PROJECT;DATA=roadhead.DATA
BASE=roadhead.OUT;OUT=BASE/'rolling_dynamic_v1'
TP=Path(os.environ.get('TRAJECTORY_PROJECT', PROJECT/'trajectory_prediction'))
SCENE='scene-0798';SEED=17
CAMERAS=('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):roadhead.save(OUT/n,x)
def sha(p):return roadhead.digest(p)
def pose(r):
    m=np.eye(4);m[:3,:3]=Quaternion(r['rotation']).rotation_matrix;m[:3,3]=r['translation'];return m
def gzrows(p):
    with gzip.open(p,'rt') as f:
        for row in f:yield json.loads(row)

def prepare():
    assert not OUT.exists(),'Do not overwrite';OUT.mkdir()
    meta=DATA/'v1.0-trainval';scene=next(s for s in read(meta/'scene.json') if s['name']==SCENE)
    samples={r['token']:r for r in read(meta/'sample.json') if r['scene_token']==scene['token']}
    ordered=[];token=scene['first_sample_token']
    while token:r=samples[token];ordered.append(r);token=r['next']
    assert len(ordered)==scene['nbr_samples']==40
    sensors={r['token']:r['channel'] for r in read(meta/'sensor.json')};calibs={r['token']:r for r in read(meta/'calibrated_sensor.json')}
    selected=[r for r in read(meta/'sample_data.json') if r['sample_token'] in samples and r['is_key_frame'] and sensors[calibs[r['calibrated_sensor_token']]['sensor_token']] in ('LIDAR_TOP',)+CAMERAS]
    bysample={}
    for r in selected:bysample.setdefault(r['sample_token'],{})[sensors[calibs[r['calibrated_sensor_token']]['sensor_token']]]=r
    needed={r['ego_pose_token'] for r in selected};egos={r['token']:r for r in read(meta/'ego_pose.json') if r['token'] in needed}
    rows=[];filehashes={}
    for i,s in enumerate(ordered):
        channels=bysample[s['token']];sd=channels['LIDAR_TOP'];calib=pose(calibs[sd['calibrated_sensor_token']]);ego=pose(egos[sd['ego_pose_token']]);lidar_world=ego@calib
        assert sd['timestamp']==s['timestamp'];images={}
        for camera in CAMERAS:
            cam=channels[camera];c=calibs[cam['calibrated_sensor_token']];world=pose(egos[cam['ego_pose_token']])@pose(c);path=DATA/cam['filename'];assert path.is_file()
            images[camera]=dict(img_path=str(path),lidar2cam=np.linalg.solve(world,lidar_world).tolist(),cam2img=c['camera_intrinsic'],timestamp_us=cam['timestamp']);filehashes[str(path)]=sha(path)
        path=DATA/sd['filename'];assert path.is_file();filehashes[str(path)]=sha(path)
        rows.append(dict(index=i,scene=SCENE,sample=s['token'],token=sd['token'],timestamp_us=s['timestamp'],elapsed_s=(s['timestamp']-ordered[0]['timestamp'])/1e6,lidar=str(path),ego_from_lidar=calib.tolist(),ego_to_global=ego.tolist(),images=images))
    xy=np.array([r['ego_to_global'] for r in rows])[:,:2,3];distance=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum());assert distance>10
    selection=read(BASE/('selection_%d.json'%SEED))['wide_fusion']
    sources=[PROJECT/'bevfusion-pretrained.pth',BASE/'wide/fusion/17.pt',BASE/'selection_17.json',BASE/'val_wide_fusion_17_prob.npy',BASE/'val_manifest.json',BASE/'val_references.npz',PROJECT/'full_val/predictions/pred_instances_3d/results_nusc.json',TP/'outputs/val_v1/penalty8_states.jsonl.gz',TP/'outputs/val_v1/penalty8_forecasts.jsonl.gz',TP/'outputs/gru_val_v1/forecasts.jsonl.gz',ROOT/'dynamic_road_core.py',ROOT/'road_shared_features.py',ROOT/'road_obstacle_check.py',ROOT/'road.py']
    save('manifest.json',rows);save('sensor_file_hashes.json',filehashes)
    save('protocol.json',dict(scene=SCENE,selection='Same scene as previous dynamic example; selected before rolling inference, no selection by road prediction quality',frames=40,elapsed_seconds=rows[-1]['elapsed_s'],ego_distance_m=distance,
        road='Fresh frozen BEVFusion camera/LiDAR/fused feature inference at every actual keyframe, same current-sweep extraction and ego sampling as old cache; width96 fusion RoadHead seed17',road_threshold=selection['threshold'],
        geometry='Recompute ground/elevated returns from each current point cloud; current detected vehicle footprints at score >=0.3',
        motion='Timestamp-indexed existing causal radar tracking and CV/GRU forecasts; all current car detections score>=0.1, no GT matching filter',
        horizons=[0,1,2,3],render_horizon=2,coordinates='Per-frame current ego grid, plus global trajectory display using observed ego poses only; global viewport extent is offline visualization metadata',
        no_training=True,no_map_label_input=True,no_temporal_road_network=True,no_frame_interpolation=True,playback='Actual approximately 2 Hz keyframes at recorded timestamp intervals; offline replay, not an end-to-end real-time speed claim',
        verification='Compare shared old anchors against existing road probabilities and ground/blocked masks, and prove per-frame inference through unique feature/probability fingerprints',
        protected_sha256={str(p):sha(p) for p in sources},script_sha256=sha(__file__)))
    print('PREPARED',len(rows),'frames',distance,'m',flush=True)

@torch.inference_mode()
def infer():
    protocol=read(OUT/'protocol.json');assert sha(__file__)==protocol['script_sha256'];assert not (OUT/'road_outputs.npz').exists()
    sys.path.insert(0,str(PROJECT/'mmdetection3d-1.4.0'))
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet3d.apis.inference import convert_SyncBN
    from mmdet3d.registry import MODELS
    from mmdet3d.structures import LiDARInstance3DBoxes
    from mmdet3d.utils import register_all_modules
    import projects.BEVFusion.bevfusion
    from road_shared_features import sampling_grid
    from road import grid_points
    from road_obstacle_check import occupancy
    cfg=Config.fromfile(str(PROJECT/'bevfusion_full_val.py'));register_all_modules();init_default_scope('mmdet3d');convert_SyncBN(cfg.model)
    model=MODELS.build(cfg.model);load_checkpoint(model,str(PROJECT/'bevfusion-pretrained.pth'),map_location='cpu');model.cuda().eval()
    pipe=Compose([p for p in cfg.test_pipeline if p.type!='LoadPointsFromMultiSweeps'])
    head=roadhead.CapacityHead(96).cuda().to(memory_format=torch.channels_last);head.load_state_dict(torch.load(BASE/'wide/fusion/17.pt',map_location='cpu')['weights']);head.eval()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    rows=read(OUT/'manifest.json');allboxes=read(PROJECT/'full_val/predictions/pred_instances_3d/results_nusc.json')['results']
    probabilities=[];grounds=[];blocks=[];audit=[];start=time.monotonic()
    vehicles={'car','bus','truck','trailer','construction_vehicle','bicycle','motorcycle'}
    for i,r in enumerate(rows):
        tick=time.monotonic()
        entry=dict(images={c:{k:v for k,v in x.items() if k!='timestamp_us'} for c,x in r['images'].items()},lidar_points=dict(lidar_path=r['lidar'],num_pts_feats=5),sample_idx=r['sample'],box_type_3d=LiDARInstance3DBoxes)
        data=model.data_preprocessor(pseudo_collate([pipe(entry)]),training=False);inp=data['inputs'];inp['points'][0][:,4]=0;img=inp['imgs'];meta=data['data_samples'][0].metainfo
        def md(key,default=None):return img.new_tensor(np.asarray([meta.get(key,default)]))
        args=(md('lidar2img'),md('cam2img'),md('cam2lidar'),md('img_aug_matrix'),md('lidar_aug_matrix',np.eye(4)),[meta])
        # Match original feature extraction defaults; head inference remains FP32/no TF32.
        torch.backends.cudnn.allow_tf32=True
        camera=model.extract_img_feat(img,[inp['points'][0].clone()],*args);lidar=model.extract_pts_feat(inp);fused=model.fusion_layer([camera,lidar])
        feature=F.grid_sample(fused,sampling_grid(np.array(r['ego_from_lidar'])),align_corners=False)[0].half()
        torch.backends.cudnn.allow_tf32=False
        prob=head(feature[None].float().contiguous(memory_format=torch.channels_last)).sigmoid()[0].cpu().numpy();assert np.isfinite(prob).all()
        raw=np.fromfile(r['lidar'],np.float32).reshape(-1,5);t=np.array(r['ego_from_lidar']);xyz=raw[:,:3]@t[:3,:3].T+t[:3,3]
        geo=grid_points(xyz,raw[:,3]);direct,_,ground=occupancy(xyz,geo);e=np.array(r['ego_to_global']);blocked=direct.copy()
        for b in allboxes[r['sample']]:
            if b['detection_name'] in vehicles and b['detection_score']>=.3:blocked |= footprint_global(b['translation'],b['size'],Quaternion(b['rotation']).rotation_matrix,e[:3,3],e[:3,:3])
        probabilities.append(prob);grounds.append(ground);blocks.append(blocked)
        audit.append(dict(frame=i,sample=r['sample'],timestamp_us=r['timestamp_us'],feature_sha256=roadhead.array_hash(feature.cpu().numpy()),probability_sha256=roadhead.array_hash(prob),road_cells=int((prob>=protocol['road_threshold']).sum()),seconds=time.monotonic()-tick))
        print('FRESH_FRAME',i+1,len(rows),'road_cells',audit[-1]['road_cells'],'seconds',round(audit[-1]['seconds'],2),flush=True)
    probs=np.stack(probabilities);ground=np.stack(grounds);blocked=np.stack(blocks)
    np.savez_compressed(OUT/'road_outputs.npz',probability=probs,ground=ground,blocked=blocked,tokens=np.array([r['token'] for r in rows]))
    oldrows=read(BASE/'val_manifest.json');indices={r['token']:i for i,r in enumerate(oldrows)};oldp=np.load(BASE/'val_wide_fusion_17_prob.npy',mmap_mode='r');checks=[]
    with np.load(BASE/'val_references.npz') as z:
        for i,r in enumerate(rows):
            if r['token'] not in indices:continue
            j=indices[r['token']];diff=abs(probs[i]-oldp[j]);m=probs[i]>=protocol['road_threshold'];n=oldp[j]>=protocol['road_threshold'];union=(m|n).sum()
            checks.append(dict(frame=i,max_probability_difference=float(diff.max()),mean_probability_difference=float(diff.mean()),threshold_mask_iou=float((m&n).sum()/max(union,1)),ground_equal=bool(np.array_equal(ground[i],z['ground'][j])),blocked_equal=bool(np.array_equal(blocked[i],z['blocked'][j]))))
    assert len(checks)>=3 and all(x['mean_probability_difference']<.003 and x['threshold_mask_iou']>.99 and x['ground_equal'] and x['blocked_equal'] for x in checks),checks
    assert len({x['feature_sha256'] for x in audit})==len(rows) and len({x['probability_sha256'] for x in audit})==len(rows)
    save('fresh_inference.json',dict(frames=audit,overlap_checks=checks,all_frames_fresh=True,unique_feature_hashes=len(rows),unique_probability_hashes=len(rows),adjacent_mean_abs_probability_change=np.abs(np.diff(probs,axis=0)).mean((1,2)).tolist(),total_seconds=time.monotonic()-start,device=torch.cuda.get_device_name()))
    print('ROAD_INFERENCE_VERIFIED',checks,flush=True)

def motion():
    from nuscenes.eval.common.config import config_factory
    rows=read(OUT/'manifest.json');tokens={r['sample'] for r in rows};protocol=read(OUT/'protocol.json');allowed=config_factory('tracking_nips_2019').class_range
    raw=read(PROJECT/'full_val/predictions/pred_instances_3d/results_nusc.json')['results'];boxes={t:[b for b in raw[t] if b['detection_name'] in allowed and b['detection_score']>=.1] for t in tokens};del raw
    states={(s['sample_token'],s['detection_index']):s for s in gzrows(TP/'outputs/val_v1/penalty8_states.jsonl.gz') if s['sample_token'] in tokens and s['observed']}
    rates={(s['sample_token'],s['detection_index']):s['yaw_rate_rad_s'] for s in gzrows(TP/'outputs/val_v1/penalty8_forecasts.jsonl.gz') if s['sample_token'] in tokens}
    forecasts={(s['sample_token'],s['detection_index']):{k:s[k] for k in ('tracking_id','used_gru','predictions')} for s in gzrows(TP/'outputs/gru_val_v1/forecasts.jsonl.gz') if s['sample_token'] in tokens}
    with np.load(OUT/'road_outputs.npz') as z:road={k:z[k] for k in z.files}
    occ=[];passes=[];objects=[]
    for i,r in enumerate(rows):
        obs=[];ego=np.array(r['ego_to_global']);e=dict(translation=ego[:3,3].tolist(),rotation=ego[:3,:3].tolist())
        for j,b in enumerate(boxes[r['sample']]):
            if b['detection_name']!='car':continue
            key=(r['sample'],j);v=np.array(b['velocity']);v=np.nan_to_num(v);origin=np.array(b['translation'][:2]);res=np.zeros((6,2));rate=0.;tid='det_%d'%j;active=False
            if key in forecasts:
                f=forecasts[key];s=states[key];assert s['timestamp_us']==r['timestamp_us'] and s['tracking_id']==f['tracking_id'];assert np.allclose(origin,s['detection_xy'],atol=1e-8)
                v=np.array(s['state'][2:]);rate=rates[key];res=np.array(f['predictions']['gru'])-np.array(f['predictions']['cv']);tid=f['tracking_id'];active=f['used_gru']
            o=dict(origin=origin.tolist(),z=b['translation'][2],velocity=v.tolist(),yaw_rate=rate,gru_residual=res.tolist(),size=b['size'],rotation=Quaternion(b['rotation']).rotation_matrix.tolist(),tracking_id=tid,score=b['detection_score'],gru_active=active)
            o['future_global']=[trajectory(o,h,'gru').tolist() for h in (0,.5,1,1.5,2,2.5,3)];obs.append(o)
        masks=np.stack([predict_occupancy(obs,e,h,'gru') for h in range(4)])
        releases=np.stack([release(road['probability'][i],protocol['road_threshold'],road['ground'][i],road['blocked'][i],m) for m in masks]);assert not (releases&~road['ground'][i]).any() and not (releases&road['blocked'][i]).any()
        occ.append(masks);passes.append(releases);objects.append(dict(sample=r['sample'],timestamp_us=r['timestamp_us'],objects=obs))
    np.savez_compressed(OUT/'rolling_outputs.npz',**road,occupancy=np.stack(occ),release=np.stack(passes),elapsed_s=np.array([r['elapsed_s'] for r in rows]),ego_to_global=np.array([r['ego_to_global'] for r in rows]))
    save('rolling_objects.json',objects)
    for path,h in protocol['protected_sha256'].items():assert sha(path)==h,path
    summary=dict(status='PASS',scene=SCENE,frames=len(rows),duration_s=rows[-1]['elapsed_s'],ego_distance_m=protocol['ego_distance_m'],road_threshold=protocol['road_threshold'],fresh_road_inference_every_frame=True,tracked_forecasts_updated_every_frame=True,static_road_probability_reused=False,map_labels_used=False,no_training=True,no_future_ego_pose_in_prediction=True,overlap_checks=read(OUT/'fresh_inference.json')['overlap_checks'],sensor_inputs_unchanged=all(sha(path)==h for path,h in read(OUT/'sensor_file_hashes.json').items()),all_original_sources_unchanged=True,
        adjacent_road_mask_changed_cells=np.count_nonzero(np.diff(road['probability']>=protocol['road_threshold'],axis=0),axis=(1,2)).tolist(),limits=['2 Hz offline keyframe replay, not a real-time performance benchmark','Road head remains a single-frame model, no recurrent BEV or temporal smoothing','Radar tracking/trajectory predictions reused from timestamp-indexed causal cache','Only cars receive future motion prediction','Unknown/currently blocked regions remain restricted within each future forecast; next real frame may provide new evidence'])
    save('verification.json',summary);print('ROLLING_VERIFIED',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','infer','motion']);a=p.parse_args();{'prepare':prepare,'infer':infer,'motion':motion}[a.command]()
