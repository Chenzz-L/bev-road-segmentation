"""Log-isolated, fixed-feature, width-only RoadHead capacity experiment.

All outputs are isolated from prior experiments. No BEVFusion model is loaded.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('ROAD_WORK_DIR', SOURCE))
DATA=Path(os.environ.get('NUSCENES_ROOT', ROOT.parent/'nuscenes_trainval_val'))
PROJECT=Path(os.environ.get('BEVFUSION_PROJECT', ROOT.parent/'BEVFusion'))
BASE=ROOT/'road_refinement/shared_bev'
TRAIN=BASE/'train200'
PROPOSAL=TRAIN/'training_coverage_split_v1'
OUT=Path(os.environ.get('ROAD_LOG_HEAD_DIR', TRAIN/'log_head_capacity_v1'))
CACHE=Path(os.environ.get('ROAD_LOG_HEAD_CACHE', OUT/'cache'))
# Set to the original combined camera/LiDAR/fusion feature cache if available.
OLD_CACHE=Path(os.environ.get('ROAD_FEATURE_CACHE', BASE/'feature_cache'))
SEEDS=(17,29,43)
WIDTHS={'base':32,'wide':96}
MODES=('lidar','fusion')
THRESHOLDS=(.35,.40,.45,.50,.55,.60,.65)
STEPS=1800
BATCH=24
axis=(np.arange(96)+.5)*.5-24
gx,gy=np.meshgrid(axis,axis)
ROI=~((abs(gx)<3)&(abs(gy)<1.5))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

class CapacityHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(256,width,1),nn.GroupNorm(4,width),nn.ReLU())
        blocks=[]
        for dilation in (1,2,4):
            blocks += [nn.Conv2d(width,width,3,dilation=dilation,padding=dilation),nn.GroupNorm(4,width),nn.ReLU()]
        self.context=nn.Sequential(*blocks)
        self.output=nn.Conv2d(width,1,1)
    def forward(self,x):
        z=self.project(x)
        return F.interpolate(self.output(z+self.context(z)),size=(96,96),mode='bilinear',align_corners=False)[:,0]

def sources():
    original=read(BASE/'manifest.json')
    return dict(train=read(TRAIN/'manifest.json'),dev=[r for r in original if r['split']=='dev'],
                val=[r for r in original if r['split']=='val'],reserved=read(BASE/'passable_walkway/frozen/new_manifest.json'))
def rows(split):return read(OUT/f'{split}_manifest.json')
def original_feature(r,mode,split):
    old=OLD_CACHE/(r['token']+'.npy')
    if old.exists():
        a=np.load(old,mmap_mode='r')
        return (a[80:336] if mode=='lidar' else a[336:592]),str(old)
    if mode=='lidar':
        p=TRAIN/'adaptive_v1/modalities'/(r['token']+'.npy')
        return np.load(p,mmap_mode='r')[80:336],str(p)
    p=TRAIN/('eval27_features' if split=='reserved' else 'features')/(r['token']+'.npy')
    return np.load(p,mmap_mode='r'),str(p)
def refs(split):
    with np.load(OUT/f'{split}_references.npz') as f:d={k:f[k] for k in f.files}
    assert list(d['tokens'])==[r['token'] for r in rows(split)]
    return d
def check_protected():
    protected=read(OUT/'protocol.json')['protected_sha256']
    result={p:digest(p)==h for p,h in protected.items()}
    assert all(result.values()),result
    return result

def setup():
    assert not (OUT/'protocol.json').exists(),'Do not overwrite existing experiment'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    old=sources();meta=read(DATA/'v1.0-trainval/scene.json')
    scenes={r['name']:r for r in meta}
    planned={s:read(PROPOSAL/f'{s}_manifest.proposed.json') for s in ('train','dev')}
    planned.update(val=old['val'],reserved=old['reserved'])
    protected={str(ROOT/k):v for k,v in read(PROPOSAL/'protocol.json')['protected_sha256'].items()}
    for p in [PROPOSAL/'train_manifest.proposed.json',PROPOSAL/'dev_manifest.proposed.json',
              PROPOSAL/'dev_log_blocklist.json',PROJECT/'bevfusion-pretrained.pth',
              SOURCE/'road_shared_heads.py',SOURCE/'road_shared_features.py']:
        protected[str(p)]=digest(p)
    for s in ('dev','val','reserved'):
        p=TRAIN/f'modality200_v1/{s}_pure_geometry.npz';protected[str(p)]=digest(p)
    for p,h in protected.items():assert digest(p)==h,p
    sets={s:{k:{scenes[r['scene']]['log_token'] if k=='log' else r[k] for r in rr} for k in ('token','scene','log')} for s,rr in planned.items()}
    overlap={}
    for i,a in enumerate(planned):
        for b in list(planned)[i+1:]:
            overlap[a+'__'+b]={k:len(sets[a][k]&sets[b][k]) for k in sets[a]}
            assert not any(overlap[a+'__'+b].values())
    assert [len(planned[s]) for s in planned]==[1956,564,542,108]
    assert sets['train']['token']|sets['dev']['token']=={r['token'] for s in ('train','dev') for r in old[s]}
    assert sets['dev']['log']==set(read(PROPOSAL/'dev_log_blocklist.json')['logs'])
    assert not sets['train']['log']&sets['dev']['log']
    counts={s:dict(frames=len(rr),scenes=len(sets[s]['scene']),logs=len(sets[s]['log'])) for s,rr in planned.items()}
    # Base uses exactly the historical module topology, but newly initialized weights.
    from road_shared_heads import RoadHead
    torch.manual_seed(17);legacy=RoadHead(256)
    torch.manual_seed(17);base=CapacityHead(32)
    assert state_hash(legacy.state_dict())==state_hash(base.state_dict())
    sample=torch.randn(2,256,64,64)
    with torch.no_grad():assert torch.equal(legacy(sample),base(sample))
    protocol=dict(name='log_head_capacity_v1',date_local='2026-11-23 Asia/Shanghai',
        hypothesis='Does width-only stronger decoding increase the fusion-minus-LiDAR road IoU advantage?',
        split=counts,split_overlaps=overlap,split_source=str(PROPOSAL),
        arms=['base_lidar','base_fusion','wide_lidar','wide_fusion'],seeds=list(SEEDS),
        architecture=dict(base='Exact historical RoadHead(256), hidden width 32',wide='Same topology with width 96; GroupNorm stays 4 groups',
            unchanged='3 context convolutions, dilations 1/2/4, residual topology, 64x64 input and bilinear 96x96 output',
            parameters={a:sum(p.numel() for p in CapacityHead(w).parameters()) for a,w in WIDTHS.items()}),
        feature='Existing frozen BEVFusion cached 256x64x64 LiDAR branch / fused features, unchanged extraction layer and spatial sampling',
        initialization='Fresh per seed and architecture; exact identical initialization across modalities within each width; no road-supervised checkpoint loaded',
        loss='Unweighted BCE over valid map drivable_area labels, no protection/auxiliary/residual objective',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8),batch_size=BATCH,updates=STEPS,
        training_budget='Same 1800 updates, same shuffled sample order within seed, partial final epoch batch retained; not matched wall time/FLOPs',
        checkpoints_every=100,threshold_grid=list(THRESHOLDS),selection='Max dev micro-IoU; ties retain first checkpoint and first threshold',
        evaluation_gate='Only after all 12 training runs complete and their chosen checkpoints/thresholds are hash-locked',
        primary='Per evaluation set: interaction = (wide fusion IoU - wide LiDAR IoU) - (base fusion IoU - base LiDAR IoU), mean of 3 paired seeds',
        bootstrap='6000 paired log-cluster draws shared across all arms/seeds; mean per-seed micro-IoU differences; 95% and 97.5% intervals (2 evaluation sets)',
        stable_expansion='Both evaluation sets: interaction >0 for all 3 seeds, 97.5% interval lower >0, mean wide-fusion IoU > base-fusion IoU, fixed-0.6 mean interaction >0',
        diagnostics=['selected checkpoint full train/dev BCE and IoU','fixed threshold 0.6','recall, precision, nonroad FPR, raw walkway FP','pure LiDAR and shared obstacle release metrics','per-log and region counts'],
        limits=['Width-only capacity test, not all decoder architectures','Evaluation sets reused, not blind','Road-task log isolation does not audit detection pretraining exposure','Different logs may share geography','Development used for selection, not an unbiased performance estimate','Frozen features remain unchanged even if head underfits','Not all unchanged evaluation obstacles are LiDAR-only'],
        runtime=dict(torch=torch.__version__,python=platform.python_version(),float32_compute=True,tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,hard_timeout_per_worker_hours=6),
        protected_sha256=protected,script_sha256=digest(__file__),legacy_head_equivalence=True)
    save(OUT/'protocol.json',protocol)
    for s,rr in planned.items():save(OUT/f'{s}_manifest.json',rr)
    save(OUT/'dev_log_blocklist.json',read(PROPOSAL/'dev_log_blocklist.json'))
    oldrefs={}
    for s in old:
        with np.load(TRAIN/f'adaptive_v1/{s}_geometry.npz') as f:
            assert list(f['tokens'])==[r['token'] for r in old[s]]
            oldrefs[s]={k:f[k] for k in ('tokens','y','walk','ground','blocked')}
        if s in ('train','dev'):oldrefs[s]['direct']=oldrefs[s]['blocked']
        else:
            with np.load(TRAIN/f'modality200_v1/{s}_pure_geometry.npz') as f:
                assert list(f['tokens'])==list(oldrefs[s]['tokens'])
                oldrefs[s]['direct']=f['direct']
    for s,rr in planned.items():
        d={}
        for key in ('tokens','y','walk','ground','blocked','direct'):
            d[key]=np.stack([oldrefs[r['source_split']][key][r['source_index']] for r in rr]) if s in ('train','dev') else oldrefs[s][key]
        assert list(d['tokens'])==[r['token'] for r in rr]
        d['logs']=np.array([scenes[r['scene']]['log_token'] for r in rr])
        d['scenes']=np.array([r['scene'] for r in rr])
        logmeta={r['token']:r['location'] for r in read(DATA/'v1.0-trainval/log.json')}
        d['regions']=np.array([logmeta[l] for l in d['logs']])
        np.savez_compressed(OUT/f'{s}_references.npz',**d)
    receipts=[]
    for s,rr in planned.items():
        for mode in MODES:
            target=CACHE/f'{s}_{mode}.npy'
            assert not target.exists(),target
            mmap=np.lib.format.open_memmap(target,mode='w+',dtype=np.float16,shape=(len(rr),256,64,64))
            items=[]
            for i,r in enumerate(rr):
                a,path=original_feature(r,mode,r.get('source_split',s))
                assert a.shape==(256,64,64) and a.dtype==np.float16 and np.isfinite(a).all()
                mmap[i]=a
                items.append(dict(token=r['token'],source=path,array_sha256=array_hash(a)))
                if (i+1)%400==0:print('CACHE',s,mode,i+1,flush=True)
            mmap.flush();del mmap
            receipts.append(dict(split=s,mode=mode,cache=str(target),cache_sha256=digest(target),items=items))
            print('CACHED',s,mode,len(rr),flush=True)
    save(OUT/'feature_receipts.json',receipts)
    save(OUT/'setup_verification.json',dict(protected=check_protected(),all_features_finite=True,token_aligned=True,legacy_head_equivalent=True,split_overlaps=overlap))
    print('SETUP_DONE',flush=True)

def runtime():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert digest(__file__)==read(OUT/'protocol.json')['script_sha256']
def load_gpu(split,mode):
    x=torch.from_numpy(np.load(CACHE/f'{split}_{mode}.npy')).cuda()
    d=refs(split);y=torch.tensor(d['y'],device='cuda')
    return x,y
@torch.no_grad()
def predict(net,x):
    net.eval()
    return torch.cat([net(b.float().contiguous(memory_format=torch.channels_last)) for b in x.split(BATCH)])
def basic(mask,y):
    valid=y!=255;road=y==1;negative=y==0
    tp=int((mask&road).sum());fp=int((mask&negative).sum());fn=int((~mask&road).sum());tn=int((~mask&negative).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,union=tp+fp+fn,iou=tp/(tp+fp+fn),recall=tp/(tp+fn),precision=tp/max(tp+fp,1),nonroad_fpr=fp/(fp+tn))
def score(mask,d):
    y=d['y'];r=basic(mask,y)
    r['walkway_raw_fp']=int((mask&d['walk']).sum())
    for kind,blocked in [('common',d['blocked']),('pure',d['direct'])]:
        eligible=d['ground']&~blocked
        good=eligible&(y==1);bad=eligible&(y==0)
        r[kind+'_correct_release']=int((mask&good).sum())
        r[kind+'_eligible_road']=int(good.sum())
        r[kind+'_effective_coverage']=float((mask&good).sum()/good.sum())
        r[kind+'_offroad_pass_fp']=int((mask&bad).sum())
        r[kind+'_walkway_pass_fp']=int((mask&eligible&d['walk']).sum())
        missing=~d['ground']&~blocked&(y==1)&ROI
        r[kind+'_missing_recall']=float(mask[missing].mean())
    return r

def smoke():
    runtime()
    from road_shared_heads import RoadHead
    x,y=load_gpu('train','lidar');x=x[:4].float().contiguous(memory_format=torch.channels_last);y=y[:4]
    result={}
    for arch,w in WIDTHS.items():
        torch.manual_seed(17);net=CapacityHead(w).cuda().to(memory_format=torch.channels_last)
        if arch=='base':
            ref=RoadHead(256).cuda().to(memory_format=torch.channels_last);ref.load_state_dict(net.state_dict())
            assert torch.equal(net(x),ref(x));del ref
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        z=net(x);valid=y!=255;loss=F.binary_cross_entropy_with_logits(z[valid],y[valid].float())
        assert z.shape==(4,96,96) and torch.isfinite(loss)
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        opt.step();result[arch]=dict(loss=float(loss),parameters=sum(p.numel() for p in net.parameters()))
    save(OUT/'smoke.json',dict(success=True,results=result,not_used_for_selection=True))
    print('SMOKE_DONE',result,flush=True)

def train(seed):
    runtime();start=time.monotonic();selections={}
    for mode in MODES:
        x,y=load_gpu('train',mode);dx,dy=load_gpu('dev',mode)
        for arch,width in WIDTHS.items():
            target=OUT/arch/mode/f'{seed}.pt'
            assert not target.exists(),target
            torch.manual_seed(seed);net=CapacityHead(width).cuda().to(memory_format=torch.channels_last)
            init_hash=state_hash(net.state_dict());order=torch.Generator().manual_seed(seed);order_hash=hashlib.sha256()
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01,betas=(.9,.999),eps=1e-8)
            best=-1.;history=[];step=0;running_loss=0.;armstart=time.monotonic()
            while step<STEPS:
                net.train()
                for idx in torch.randperm(len(x),generator=order).split(BATCH):
                    order_hash.update(idx.numpy().tobytes())
                    z=net(x[idx].float().contiguous(memory_format=torch.channels_last));target_y=y[idx];valid=target_y!=255
                    loss=F.binary_cross_entropy_with_logits(z[valid],target_y[valid].float())
                    assert torch.isfinite(loss),('nonfinite',seed,arch,mode,step)
                    opt.zero_grad();loss.backward();opt.step();step+=1;running_loss+=float(loss.detach())
                    if step%100==0:
                        logit=predict(net,dx);prob=logit.sigmoid();valid_dev=dy!=255
                        dev_loss=float(F.binary_cross_entropy_with_logits(logit[valid_dev],dy[valid_dev].float()))
                        scores=[]
                        for threshold in THRESHOLDS:
                            m=prob>=threshold
                            tp=int((m&(dy==1)).sum());union=int(((m|(dy==1))&valid_dev).sum());iou=tp/union
                            scores.append(dict(threshold=threshold,iou=iou,tp=tp,union=union))
                            if iou>best:
                                best=iou;selected=dict(step=step,threshold=threshold,dev_iou=iou)
                                bestweights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                        history.append(dict(step=step,mean_training_batch_bce=running_loss/100,dev_bce=dev_loss,threshold_scores=scores,best=best))
                        running_loss=0.
                        save(OUT/f'progress_{seed}.json',dict(seed=seed,arch=arch,mode=mode,step=step,selected=selected,elapsed_seconds=time.monotonic()-start))
                        print('TRAIN',seed,arch,mode,step,'dev_best',round(best,6),'dev_BCE',round(dev_loss,6),'seconds',round(time.monotonic()-armstart),flush=True)
                        net.train()
                    if step==STEPS:break
            target.parent.mkdir(parents=True,exist_ok=True)
            checkpoint=dict(weights=bestweights,selected=selected,arch=arch,mode=mode,seed=seed,initial_hash=init_hash,
                            sample_order_sha256=order_hash.hexdigest(),history=history,updates=step,protocol_sha256=digest(OUT/'protocol.json'))
            torch.save(checkpoint,target)
            net.load_state_dict(bestweights)
            diagnostics={}
            for s,xx,yy in [('train',x,y),('dev',dx,dy)]:
                logit=predict(net,xx);valid_diag=yy!=255
                bce=float(F.binary_cross_entropy_with_logits(logit[valid_diag],yy[valid_diag].float()))
                prob=logit.sigmoid().cpu().numpy()
                diagnostics[s]=dict(bce=bce,selected=basic(prob>=selected['threshold'],yy.cpu().numpy()),fixed060=basic(prob>=.6,yy.cpu().numpy()))
                if s=='dev':
                    np.save(OUT/f'dev_{arch}_{mode}_{seed}_prob.npy',prob)
                    assert abs(diagnostics[s]['selected']['iou']-selected['dev_iou'])<1e-12
            selections[arch+'_'+mode]=dict(**selected,initial_hash=init_hash,sample_order_sha256=order_hash.hexdigest(),
                parameters=sum(p.numel() for p in net.parameters()),checkpoint_sha256=digest(target),elapsed_seconds=time.monotonic()-armstart,diagnostics=diagnostics,history=history)
            save(OUT/f'selection_{seed}.json',selections)
            print('ARM_DONE',seed,arch,mode,selected,flush=True)
            del net,opt,logit,prob,checkpoint,bestweights
        del x,y,dx,dy
    assert len({r['sample_order_sha256'] for r in selections.values()})==1
    for arch in WIDTHS:assert selections[arch+'_lidar']['initial_hash']==selections[arch+'_fusion']['initial_hash']
    save(OUT/f'train_{seed}_done.json',dict(seed=seed,completed=4,seconds=time.monotonic()-start,gpu=torch.cuda.get_device_name()))
    print('TRAIN_DONE',seed,flush=True)

def lock():
    assert not (OUT/'locked.json').exists()
    records={}
    for seed in SEEDS:
        assert read(OUT/f'train_{seed}_done.json')['completed']==4
        selection=read(OUT/f'selection_{seed}.json');assert len(selection)==4
        for arch in WIDTHS:
            for mode in MODES:
                p=OUT/arch/mode/f'{seed}.pt';r=selection[arch+'_'+mode]
                assert digest(p)==r['checkpoint_sha256']
                records[f'{arch}_{mode}_{seed}']=dict(checkpoint_sha256=r['checkpoint_sha256'],threshold=r['threshold'],step=r['step'])
    save(OUT/'locked.json',dict(records=records,protected=check_protected(),all_12_completed_before_evaluation=True))
    print('LOCKED_12',flush=True)

def evaluate(seed):
    runtime();locked=read(OUT/'locked.json');report={}
    assert not (OUT/f'results_{seed}.json').exists()
    for split in ('val','reserved'):
        d=refs(split);report[split]={};counts={}
        for mode in MODES:
            x,y=load_gpu(split,mode)
            for arch,width in WIDTHS.items():
                p=OUT/arch/mode/f'{seed}.pt';record=locked['records'][f'{arch}_{mode}_{seed}']
                assert digest(p)==record['checkpoint_sha256']
                ck=torch.load(p,map_location='cpu');net=CapacityHead(width).cuda().to(memory_format=torch.channels_last)
                net.load_state_dict(ck['weights']);prob=predict(net,x).sigmoid().cpu().numpy()
                np.save(OUT/f'{split}_{arch}_{mode}_{seed}_prob.npy',prob)
                masks=prob>=record['threshold'];fixed=prob>=.6
                report[split][arch+'_'+mode]=dict(selected=score(masks,d),fixed060=score(fixed,d))
                for name,m in [('selected',masks),('fixed060',fixed)]:
                    counts[arch+'_'+mode+'_'+name+'_tp']=(m&(d['y']==1)).sum((1,2))
                    counts[arch+'_'+mode+'_'+name+'_union']=((m|(d['y']==1))&(d['y']!=255)).sum((1,2))
                del net,prob
            del x,y
        np.savez_compressed(OUT/f'{split}_{seed}_frame_counts.npz',**counts)
    save(OUT/f'results_{seed}.json',report)
    print('EVALUATE_DONE',seed,flush=True)

def summarize():
    reports={s:read(OUT/f'results_{s}.json') for s in SEEDS};summary={}
    for split in ('val','reserved'):
        out={};d=refs(split);logs,index=np.unique(d['logs'],return_inverse=True)
        draws=np.random.default_rng(20261123).multinomial(len(logs),np.full(len(logs),1/len(logs)),size=6000)
        for threshold in ('selected','fixed060'):
            means={arm:{k:float(np.mean([reports[s][split][arm][threshold][k] for s in SEEDS])) for k in reports[SEEDS[0]][split][arm][threshold]} for arm in reports[SEEDS[0]][split]}
            bootstrap={};perlog={}
            for arch in WIDTHS:
                for mode in MODES:
                    arm=arch+'_'+mode;boot=[];perlog[arm]=[]
                    for seed in SEEDS:
                        with np.load(OUT/f'{split}_{seed}_frame_counts.npz') as f:
                            tp=np.bincount(index,weights=f[arm+'_'+threshold+'_tp'],minlength=len(logs))
                            union=np.bincount(index,weights=f[arm+'_'+threshold+'_union'],minlength=len(logs))
                        boot.append((draws@tp)/(draws@union))
                        perlog[arm].append(dict(seed=seed,tp=tp.tolist(),union=union.tolist()))
                    bootstrap[arm]=np.array(boot)
            bg=bootstrap['base_fusion']-bootstrap['base_lidar'];wg=bootstrap['wide_fusion']-bootstrap['wide_lidar']
            interactions=[100*((reports[s][split]['wide_fusion'][threshold]['iou']-reports[s][split]['wide_lidar'][threshold]['iou'])-(reports[s][split]['base_fusion'][threshold]['iou']-reports[s][split]['base_lidar'][threshold]['iou'])) for s in SEEDS]
            stat=dict(mean_pp=float(np.mean(interactions)),per_seed_pp=dict(zip(map(str,SEEDS),interactions)),
                paired_log_ci95_pp=np.percentile(100*(wg-bg).mean(0),[2.5,97.5]).tolist(),
                paired_log_ci975_pp=np.percentile(100*(wg-bg).mean(0),[1.25,98.75]).tolist())
            out[threshold]=dict(means=means,base_fusion_gain_pp=100*(means['base_fusion']['iou']-means['base_lidar']['iou']),
                wide_fusion_gain_pp=100*(means['wide_fusion']['iou']-means['wide_lidar']['iou']),interaction=stat,
                head_effect_lidar_pp=100*(means['wide_lidar']['iou']-means['base_lidar']['iou']),
                head_effect_fusion_pp=100*(means['wide_fusion']['iou']-means['base_fusion']['iou']))
            save(OUT/f'{split}_{threshold}_log_counts.json',dict(logs=logs.tolist(),counts=perlog))
        r=out['selected'];out['stable_expansion']=bool(all(v>0 for v in r['interaction']['per_seed_pp'].values()) and r['interaction']['paired_log_ci975_pp'][0]>0 and r['head_effect_fusion_pp']>0 and out['fixed060']['interaction']['mean_pp']>0)
        summary[split]=out
    save(OUT/'summary.json',dict(protocol=read(OUT/'protocol.json'),evaluation=summary,stable_expansion_both=all(v['stable_expansion'] for v in summary.values())))
    print(json.dumps({s:{k:v for k,v in r['selected'].items() if k!='means'} for s,r in summary.items()},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',choices=['setup','smoke','train','lock','evaluate','summary']);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args()
    if a.job=='train':train(a.seed)
    elif a.job=='evaluate':evaluate(a.seed)
    else:dict(setup=setup,smoke=smoke,lock=lock,summary=summarize)[a.job]()
