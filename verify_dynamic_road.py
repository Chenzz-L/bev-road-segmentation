"""Independent SDK footprint, forecast replay, and count checks for dynamic trial."""
import json
from pathlib import Path
import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import dynamic_road_trial as run
from dynamic_road_core import GX,GY,ROI,footprint_global,trajectory,METHODS

def main():
    out=run.OUT;inp=run.read(out/'causal_inputs.json');p=run.read(out/'protocol.json');count=0;maxerr=0.
    lookup={(r['sample'],o['detection_index']):o for r in inp for o in r['objects']}
    # Compare at the published six trajectory knots, including unused/unmatched anchors.
    replayed=0
    for f in run.gzrows(run.FORECASTS):
        key=(f['sample_token'],f['detection_index'])
        if key not in lookup:continue
        obj=lookup[key]
        for method in ('cv','ctrv','gru'):
            actual=np.array([trajectory(obj,t,method) for t in np.arange(1,7)*.5]);expected=np.array(f['predictions'][method])
            error=float(abs(actual-expected).max());maxerr=max(maxerr,error);assert np.allclose(actual,expected,atol=1e-7,rtol=0)
        replayed+=1
    for row in inp:
        ego=row['ego'];q=Quaternion(matrix=np.array(ego['rotation']))
        for obj in row['objects'][:2]:
            center=[*obj['origin'],obj['z']];box=Box(center,obj['size'],Quaternion(matrix=np.array(obj['rotation'])))
            box.translate(-np.array(ego['translation']));box.rotate(q.inverse)
            yaw=box.orientation.yaw_pitch_roll[0];dx=GX-box.center[0];dy=GY-box.center[1];c,s=np.cos(yaw),np.sin(yaw)
            expected=(abs(c*dx+s*dy)<=box.wlh[1]/2)&(abs(-s*dx+c*dy)<=box.wlh[0]/2)&ROI
            if box.corners()[2].min()>2.2 or box.corners()[2].max()<-.5:expected[:]=False
            actual=footprint_global(center,obj['size'],obj['rotation'],ego['translation'],ego['rotation'])
            assert np.array_equal(actual,expected);count+=1
    with np.load(out/'prediction_masks.npz') as z:pred={k:z[k] for k in z.files}
    with np.load(out/'evaluation_truth.npz') as z:truth=z['masks']
    with np.load(out/'counts.npz') as z:counts=z['counts']
    refs=run.refdata();checks=0
    for si,seed in enumerate(run.SEEDS):
        prob=np.load(run.BASE/('val_wide_fusion_%d_prob.npy'%seed),mmap_mode='r')
        for i in range(len(inp)):
            e=refs['ground'][i]&~refs['blocked'][i]&ROI;base=(prob[i]>=p['road_thresholds'][str(seed)])&e;y=refs['y'][i]
            for hi in range(4):
                if not pred['valid'][i,hi]:assert not counts[si,i,hi].any();continue
                car,actors,_=truth[i,hi];good=(y==1)&e&~actors
                for mi,arm in enumerate(run.ARMS):
                    mask=np.zeros((96,96),bool) if arm=='current' else pred['masks'][i,hi,mi-1];r=base&~mask
                    expected=[good.sum(),(e&car).sum(),(e&actors).sum(),r.sum(),(r&good).sum(),(r&car).sum(),(r&actors).sum(),(r&(y==0)).sum(),car.sum(),mask.sum(),(mask&car).sum(),(mask&~car&ROI).sum(),(~mask&car).sum()]
                    assert np.array_equal(counts[si,i,hi,mi],expected);assert not (r&~base).any();checks+=1
    metadata={n:run.sha(run.DATA/'v1.0-trainval'/n) for n in ['sample.json','sample_annotation.json','ego_pose.json','instance.json','category.json']}
    run.save('independent_verification.json',dict(status='PASS',sdk_box_footprints_compared=count,forecast_replays=replayed,forecast_max_abs_m=maxerr,independent_count_vectors=checks,current_candidate_subset_verified=True,ground_truth_metadata_sha256=metadata,
        cache_boundary='Upstream forecast JSON records contain evaluation metadata; inference copies only tracking_id, used_gru and predictions. Targets, labels, errors and instance matching IDs never enter dynamic_road_core. Future occupancy annotation table is loaded only after prediction lock.'))
    print('INDEPENDENT_VERIFICATION_PASS',count,replayed,checks,flush=True)

if __name__=='__main__':main()
