"""No-training future occupancy prototype. Lock predictions before GT evaluation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,gzip,hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from dynamic_road_core import METHODS,ROI,predict_occupancy,release,footprint_global,tests,trajectory

ROOT=Path(__file__).resolve().parent
WORK=Path(os.environ.get('ROAD_WORK_DIR', ROOT))
BASE=Path(os.environ.get('ROAD_LOG_HEAD_DIR', WORK/'road_refinement/shared_bev/train200/log_head_capacity_v1'))
PROJECT=Path(os.environ.get('BEVFUSION_PROJECT', WORK.parent/'BEVFusion'))
OUT=BASE/'dynamic_occupancy_v1'
TP=Path(os.environ.get('TRAJECTORY_PROJECT', PROJECT/'trajectory_prediction'))
DATA=Path(os.environ.get('NUSCENES_ROOT', WORK.parent/'nuscenes_trainval_val'))
DETECTIONS=PROJECT/'full_val/predictions/pred_instances_3d/results_nusc.json'
STATES=TP/'outputs/val_v1/penalty8_states.jsonl.gz'
FORECASTS=TP/'outputs/gru_val_v1/forecasts.jsonl.gz'
BASEFORECASTS=TP/'outputs/val_v1/penalty8_forecasts.jsonl.gz'
HORIZONS=(0.,1.,2.,3.)
SEEDS=(17,29,43)
ARMS=('current',)+METHODS
FIELDS=('reference_road','car_risk_domain','all_actor_risk_domain','released','correct_release','car_conflict','actor_conflict','offroad_release','occupied_gt_car','occupied_pred','occupied_tp','occupied_fp','occupied_fn')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def gzrows(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def refdata():
    with np.load(BASE/'val_references.npz') as z:return {k:z[k] for k in z.files}

def prepare():
    assert not OUT.exists(),'Refuse overwrite';OUT.mkdir()
    save('unit_tests.json',tests())
    rows=read(BASE/'val_manifest.json');split=read(TP/'gru_data_v2/split.json');scenes={r['scene'] for r in rows}
    assert len(rows)==542 and len(scenes)==150 and scenes<=set(split['official_val_scenes'])
    assert not scenes&set(split['train_scenes']+split['dev_scenes'])
    excluded={}
    for s in ('dev','reserved'):
        ss={r['scene'] for r in read(BASE/(s+'_manifest.json'))}
        excluded[s]={k:len(ss&set(split[k])) for k in ('train_scenes','dev_scenes','official_val_scenes')}
    protected={str(p):sha(p) for p in [BASE/'val_manifest.json',BASE/'val_references.npz',DETECTIONS,STATES,FORECASTS,BASEFORECASTS,TP/'gru_run_v1/best.pt',TP/'gru_data_v2/split.json',TP/'outputs/gru_val_v1/summary.json']}
    thresholds={}
    for seed in SEEDS:
        sel=read(BASE/('selection_%d.json'%seed))['wide_fusion'];thresholds[str(seed)]=sel['threshold']
        for p in (BASE/('selection_%d.json'%seed),BASE/('val_wide_fusion_%d_prob.npy'%seed),BASE/('wide/fusion/%d.pt'%seed)):
            protected[str(p)]=sha(p)
    save('protocol.json',dict(date='2026-11-27',status='LOCKED_BEFORE_DYNAMIC_EVALUATION',no_training=True,
        task='Future car occupancy and anticipatory restriction of current evidence-supported road candidates; not full future freespace reconstruction',
        anchors='All original 542 frames / 150 official validation scenes; retain anchors with no detections, false detections and unmatched forecasts',
        exclude_other_road_sets_due_gru_overlap=excluded,road_seeds=list(SEEDS),road_thresholds=thresholds,
        future_queries='Nearest keyframe to nominal 1/2/3 s within 0.1 s and 0.75 s maximum intermediate gap, per horizon independently; use actual dt for every predictor; absent future gets explicit exclusion',
        coordinates='All outputs in current LIDAR timestamp ego frame, fixed 48m x 48m / 0.5m grid; no future ego pose input',
        arms=list(ARMS),scope='Car trajectory model only. Other obstacles and current car occupancy remain in original current blocked mask. Future GT entrants and missed cars are included, without GT matching gates.',
        prediction='Reuse all cached causal forecasts regardless of labeled/instance/error fields, strip GT metadata. Verify current detection indices and positions against causal states. Missing forecasts use current detector velocity CV for all moving models.',
        heading='Current detected box rotation and dimensions held fixed for every model; only predicted translation differs. GRU does not predict yaw.',
        interpolation='CV/CTRV queried analytically at actual dt. GRU global XY residual interpolated at 0,.5,...,3s; for 3<dt<=3.1 hold final residual and continue CV. No future measurement enters predictors.',
        uncertainty='Separate non-probabilistic margin sensitivity: each side +0.25+0.25*dt meters for hold/CV/CTRV/GRU equally. No outcome-based margin tuning.',
        release='Original road probability threshold AND current ground AND NOT current blocked AND NOT predicted future occupancy. Unknown/currently blocked areas never become released merely because a car is predicted to leave.',
        primary='Per horizon, future car-conflicting released grid cells versus hold, with road coverage and false blocking cost; GRU versus CTRV on identical anchors and footprints',
        secondary='Future all-annotated-actor conflicts, offroad release, occupancy IoU and recall, current-absent entering GT cars, per-log distribution; annotations are incomplete freespace evidence',
        bootstrap='1000 paired log draws; three fixed road seeds averaged; no significance claim or independence of frames',
        success='Report tradeoff, not simply fewer conflicts from smaller release area. Only claim no-coverage-loss improvement when car conflicts decrease and evaluable road coverage does not decrease.',
        demo_selection='Top three distinct logs by predicted GRU-at-3s intrusion into seed17 current candidates; input-only example selection, not metric/GT cherry-picking',
        limits=['Previously inspected validation set, not blind','GRU trained on 560 scenes, selected on 140; tracker penalty previously tuned on val','Road task log isolation does not remove detection pretraining exposure','Current car-blocked/unknown regions remain blocked/unknown even if cars leave','Instantaneous occupancy, not swept path, ego planning, collision avoidance or safety certification','Constant box yaw can misrepresent turning vehicles','Only cars receive motion prediction; other future actors evaluated as an additional limitation','Free means no annotated actor plus current ground evidence, not verified future physical free space'],
        protected_sha256=protected,script_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'dynamic_road_core.py']}))
    print('PROTOCOL_LOCKED',flush=True)

def infer():
    p=read(OUT/'protocol.json');assert not (OUT/'predictions.locked.json').exists()
    for name,h in p['script_sha256'].items():assert sha(ROOT/name)==h
    rows=read(BASE/'val_manifest.json');bytoken={r['sample']:i for i,r in enumerate(rows)};refs=refdata()
    assert refs['tokens'].tolist()==[r['token'] for r in rows]
    # Only pose/sample metadata are used here, never sample_annotation.json.
    md=DATA/'v1.0-trainval'
    samples={r['token']:r for r in read(md/'sample.json')}
    sd={r['token']:r for r in read(md/'sample_data.json')}
    poses={r['token']:r for r in read(md/'ego_pose.json')}
    allowed=config_factory('tracking_nips_2019').class_range
    raw=read(DETECTIONS)['results'];detections={t:[b for b in raw[t] if b['detection_name'] in allowed and b['detection_score']>=.1] for t in bytoken};del raw
    states={}
    for s in gzrows(STATES):
        if s['sample_token'] in bytoken and s['observed']:
            key=(s['sample_token'],s['detection_index']);assert key not in states;states[key]=s
    bases={}
    for f in gzrows(BASEFORECASTS):
        if f['sample_token'] in bytoken:bases[(f['sample_token'],f['detection_index'])]=dict(predictions=f['predictions'],rate=f['yaw_rate_rad_s'])
    forecasts={}
    for f in gzrows(FORECASTS):
        if f['sample_token'] in bytoken:
            # Explicit allowlist: never copy label, target, matching ID or error fields.
            forecasts[(f['sample_token'],f['detection_index'])]=dict(tracking_id=f['tracking_id'],used_gru=f['used_gru'],predictions=f['predictions'])
    occupancies=np.zeros((len(rows),4,len(METHODS),96,96),bool);valid=np.zeros((len(rows),4),bool);times=np.zeros((len(rows),4));targets=np.full((len(rows),4),'',dtype='U32')
    inputs=[];audit=Counter();egos=[];evaluable=[]
    for i,row in enumerate(rows):
        token=row['sample'];sample=samples[token];lidar=sd[row['token']]
        assert lidar['sample_token']==token
        pose=poses[lidar['ego_pose_token']];ego=dict(translation=pose['translation'],rotation=Quaternion(pose['rotation']).rotation_matrix.tolist());egos.append(ego)
        audit['max_anchor_sample_lidar_skew_us']=max(audit['max_anchor_sample_lidar_skew_us'],abs(sample['timestamp']-lidar['timestamp']))
        objects=[]
        for j,box in enumerate(detections[token]):
            if box['detection_name']!='car':continue
            key=(token,j);audit['car_detections']+=1;origin=np.array(box['translation'][:2]);v=np.array(box['velocity'],float);rate=0.;residual=np.zeros((6,2));mode='detector_cv_fallback'
            if not np.isfinite(v).all():v=np.zeros(2);audit['nonfinite_velocity_held']+=1
            if key in forecasts:
                f=forecasts[key];st=states[key];base=bases[key]
                assert st['timestamp_us']==sample['timestamp'] and st['tracking_id']==f['tracking_id']
                assert np.allclose(origin,st['detection_xy'],atol=1e-8,rtol=0)
                v=np.array(st['state'][2:]);rate=base['rate'];cv=np.asarray(f['predictions']['cv']);gr=np.asarray(f['predictions']['gru'])
                assert np.allclose(cv,origin+np.arange(1,7)[:,None]*.5*v,atol=2e-8,rtol=0)
                assert np.array_equal(cv,np.asarray(base['predictions']['cv']))
                residual=gr-cv;mode='gru' if f['used_gru'] else 'short_history_cv'
                if not f['used_gru']:assert not np.any(residual)
            audit[mode]+=1
            objects.append(dict(detection_index=j,origin=origin.tolist(),z=box['translation'][2],velocity=v.tolist(),yaw_rate=rate,gru_residual=residual.tolist(),size=box['size'],rotation=Quaternion(box['rotation']).rotation_matrix.tolist(),score=box['detection_score'],mode=mode))
        inputs.append(dict(sample=token,timestamp_us=sample['timestamp'],ego=ego,objects=objects))
        chain=[sample];x=sample
        while x['next'] and (x['timestamp']-sample['timestamp'])/1e6<3.2:
            nxt=samples[x['next']]
            if not 0<nxt['timestamp']-x['timestamp']<=750000:break
            assert nxt['scene_token']==sample['scene_token'];chain.append(nxt);x=nxt
        for hi,h in enumerate(HORIZONS):
            target=min(chain,key=lambda x:abs((x['timestamp']-sample['timestamp'])/1e6-h));dt=(target['timestamp']-sample['timestamp'])/1e6
            if abs(dt-h)>.1:
                audit['missing_future_%ds'%h]+=1;continue
            valid[i,hi]=True;times[i,hi]=dt;targets[i,hi]=target['token']
            if dt>3:audit['residual_endpoint_hold_queries']+=1
            for mi,m in enumerate(METHODS):occupancies[i,hi,mi]=predict_occupancy(objects,ego,dt,m)
        if (i+1)%100==0:print('INFER',i+1,flush=True)
    assert audit['max_anchor_sample_lidar_skew_us']<=100000
    for mi in (1,2,3):assert np.array_equal(occupancies[:,0,0],occupancies[:,0,mi])
    for mi in range(4):assert not (occupancies[:,:,mi]&~occupancies[:,:,mi+4]).any()
    assert not read(OUT/'protocol.json').get('evaluation_started',False)
    np.savez_compressed(OUT/'prediction_masks.npz',masks=occupancies,valid=valid,dt=times,targets=targets,tokens=refs['tokens'],samples=np.array([r['sample'] for r in rows]),methods=np.array(METHODS))
    save('causal_inputs.json',inputs);save('input_audit.json',dict(counts=dict(audit),max_time_error_seconds=float(np.max(np.abs(times-np.array(HORIZONS))[valid])),frames_by_horizon=valid.sum(0).tolist(),all_car_forecasts_used_without_gt_match_filter=True,annotations_loaded=False))
    probabilities=np.load(BASE/'val_wide_fusion_17_prob.npy',mmap_mode='r');scores=[]
    for i in range(len(rows)):
        if valid[i,3]:
            c=release(probabilities[i],p['road_thresholds']['17'],refs['ground'][i],refs['blocked'][i],np.zeros((96,96),bool))
            score=int((c&occupancies[i,3,METHODS.index('gru')]).sum());scores.append((score,i))
    demos=[];usedlogs=set()
    for score,i in sorted(scores,reverse=True):
        log=str(refs['logs'][i])
        if log not in usedlogs:demos.append(dict(index=i,score=score,log=log,scene=rows[i]['scene']));usedlogs.add(log)
        if len(demos)==3:break
    save('demo_selection.json',demos)
    save('predictions.locked.json',dict(status='LOCKED_BEFORE_GT_LOAD',protocol_sha256=sha(OUT/'protocol.json'),files={n:sha(OUT/n) for n in ['prediction_masks.npz','causal_inputs.json','input_audit.json','demo_selection.json']},no_annotations_loaded=True))
    print('PREDICTIONS_LOCKED',dict(audit),flush=True)

def metrics(c):
    d={k:int(v) for k,v in zip(FIELDS,c)}
    def ratio(a,b):return float(a/b) if b else None
    d.update(coverage=ratio(d['correct_release'],d['reference_road']),car_conflict_rate=ratio(d['car_conflict'],d['released']),occupancy_iou=ratio(d['occupied_tp'],d['occupied_tp']+d['occupied_fp']+d['occupied_fn']),occupancy_recall=ratio(d['occupied_tp'],d['occupied_gt_car']))
    return d
def average(ds):return {k:float(np.mean([d[k] for d in ds])) if ds[0][k] is not None else None for k in ds[0]}
def compare(a,b):
    dr=100*(b['coverage']-a['coverage']);reduction=100*(1-b['car_conflict']/a['car_conflict']) if a['car_conflict'] else None
    return dict(car_conflict_reduction_pct=reduction,coverage_delta_pp=dr,additional_correct_cells_blocked=a['correct_release']-b['correct_release'],conflict_cells_avoided=a['car_conflict']-b['car_conflict'],strict_joint_success=bool(dr>=-1e-10 and b['car_conflict']<a['car_conflict']))

def evaluate():
    assert not (OUT/'summary.json').exists();lock=read(OUT/'predictions.locked.json');protocol=read(OUT/'protocol.json')
    assert lock['protocol_sha256']==sha(OUT/'protocol.json') and lock['no_annotations_loaded']
    for n,h in lock['files'].items():assert sha(OUT/n)==h
    rows=read(BASE/'val_manifest.json');refs=refdata();inputs=read(OUT/'causal_inputs.json')
    with np.load(OUT/'prediction_masks.npz') as z:pred={k:z[k] for k in z.files}
    db=NuScenes('v1.0-trainval',str(DATA),verbose=False)
    truth=np.zeros((len(rows),4,3,96,96),bool);gtstats=Counter()
    for i,row in enumerate(rows):
        now=db.get('sample',row['sample']);current_ids={db.get('sample_annotation',a)['instance_token'] for a in now['anns']};ego=inputs[i]['ego']
        for hi,h in enumerate(HORIZONS):
            if not pred['valid'][i,hi]:continue
            future=db.get('sample',str(pred['targets'][i,hi]))
            for at in future['anns']:
                ann=db.get('sample_annotation',at)
                mask=footprint_global(ann['translation'],ann['size'],Quaternion(ann['rotation']).rotation_matrix,ego['translation'],ego['rotation'])
                truth[i,hi,1]|=mask
                if ann['category_name']=='vehicle.car':
                    truth[i,hi,0]|=mask
                    if ann['instance_token'] not in current_ids:truth[i,hi,2]|=mask;gtstats['current_absent_future_car_instances']+=int(mask.any())
        if (i+1)%100==0:print('GT',i+1,flush=True)
    np.savez_compressed(OUT/'evaluation_truth.npz',masks=truth,kind=np.array(['cars','all_annotated_objects','current_absent_cars']))
    count=np.zeros((3,len(rows),4,len(ARMS),len(FIELDS)),np.int64);enter=np.zeros((3,len(rows),4,len(ARMS)),np.int64)
    for si,seed in enumerate(SEEDS):
        prob=np.load(BASE/('val_wide_fusion_%d_prob.npy'%seed),mmap_mode='r');threshold=protocol['road_thresholds'][str(seed)]
        for i in range(len(rows)):
            y=refs['y'][i];e=refs['ground'][i]&~refs['blocked'][i]&ROI;label_valid=y!=255
            for hi in range(4):
                if not pred['valid'][i,hi]:continue
                car,allgt,entrants=truth[i,hi];available=(y==1)&e&~allgt
                for mi,arm in enumerate(ARMS):
                    occ=np.zeros((96,96),bool) if arm=='current' else pred['masks'][i,hi,mi-1]
                    released=release(prob[i],threshold,refs['ground'][i],refs['blocked'][i],occ)
                    assert not (released&~e).any()
                    count[si,i,hi,mi]=[available.sum(),(e&car).sum(),(e&allgt).sum(),released.sum(),(released&available).sum(),(released&car).sum(),(released&allgt).sum(),(released&(y==0)).sum(),car.sum(),occ.sum(),(occ&car).sum(),(occ&~car&ROI).sum(),(~occ&car).sum()]
                    enter[si,i,hi,mi]=int((released&entrants).sum())
        print('EVAL_SEED',seed,flush=True)
    np.savez_compressed(OUT/'counts.npz',counts=count,entrant_conflicts=enter,fields=np.array(FIELDS),arms=np.array(ARMS),seeds=np.array(SEEDS),logs=refs['logs'],valid=pred['valid'])
    pairs={'cv_vs_hold':('hold','cv'),'ctrv_vs_hold':('hold','ctrv'),'gru_vs_hold':('hold','gru'),'gru_vs_ctrv':('ctrv','gru'),'gru_pad_vs_hold_pad':('hold_pad','gru_pad'),'gru_pad_vs_ctrv_pad':('ctrv_pad','gru_pad'),'gru_pad_vs_gru':('gru','gru')}
    # Last pair compares explicit expansion with unexpanded GRU.
    pairs['gru_pad_vs_gru']=('gru','gru_pad')
    result=dict(status='COMPLETE',no_training=True,scope=protocol['task'],road_seeds=list(SEEDS),horizons={},limits=protocol['limits'],input_audit=read(OUT/'input_audit.json'),gt_stats=dict(gtstats))
    for hi,h in enumerate(HORIZONS):
        take=pred['valid'][:,hi];logs=sorted(set(refs['logs'][take]));methods={};perseed={}
        for mi,arm in enumerate(ARMS):
            ds=[metrics(count[si,take,hi,mi].sum(0)) for si in range(3)];methods[arm]=average(ds);perseed[arm]=ds
            methods[arm]['entrant_conflict_cells']=float(enter[:,take,hi,mi].sum(1).mean())
        comparisons={n:dict(**compare(methods[a],methods[b]),strict_seed_success=sum(compare(x,y)['strict_joint_success'] for x,y in zip(perseed[a],perseed[b]))) for n,(a,b) in pairs.items()}
        bylog=np.stack([count[:,take&(refs['logs']==l),hi].sum(1) for l in logs],axis=1)
        rng=np.random.default_rng(20261127);draws=rng.multinomial(len(logs),np.full(len(logs),1/len(logs)),size=1000)
        boot={n:[] for n in pairs}
        for weights in draws:
            total=(bylog*weights[None,:,None,None]).sum(1)
            mm={a:average([metrics(total[si,mi]) for si in range(3)]) for mi,a in enumerate(ARMS)}
            for n,(a,b) in pairs.items():boot[n].append(compare(mm[a],mm[b]))
        intervals={n:{k:np.percentile([r[k] for r in vv if r[k] is not None],[2.5,97.5]).tolist() if any(r[k] is not None for r in vv) else None for k in ('car_conflict_reduction_pct','coverage_delta_pp')} for n,vv in boot.items()}
        result['horizons'][str(int(h))]=dict(frames=int(take.sum()),scenes=len(set(refs['scenes'][take])),logs=len(logs),actual_seconds_min=float(pred['dt'][take,hi].min()),actual_seconds_max=float(pred['dt'][take,hi].max()),methods=methods,comparisons=comparisons,bootstrap95=intervals,per_seed=perseed,
            per_log={str(l):{n:compare(average([metrics(bylog[s,j,ARMS.index(a)]) for s in range(3)]),average([metrics(bylog[s,j,ARMS.index(b)]) for s in range(3)])) for n,(a,b) in pairs.items()} for j,l in enumerate(logs)})
    for path,h in protocol['protected_sha256'].items():assert sha(path)==h,path
    save('summary.json',result);save('verification.json',dict(status='PASS',original_sources_unchanged=True,predictions_locked_before_future_annotations=True,no_training=True,no_gt_matching_filter=True,no_future_pose_input=True,unknown_and_current_blocked_never_newly_released=True,all_three_road_seeds_reported=True))
    print('EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','infer','evaluate']);a=p.parse_args();{'prepare':prepare,'infer':infer,'evaluate':evaluate}[a.command]()
