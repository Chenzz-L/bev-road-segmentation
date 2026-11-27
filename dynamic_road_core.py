"""Causal raster fusion primitives. This module has no annotation/data access."""
import numpy as np

AXIS=(np.arange(96)+.5)*.5-24
GX,GY=np.meshgrid(AXIS,AXIS)
ROI=~((np.abs(GX)<3)&(np.abs(GY)<1.5))
METHODS=('hold','cv','ctrv','gru','hold_pad','cv_pad','ctrv_pad','gru_pad')

def trajectory(obj,dt,method):
    origin=np.asarray(obj['origin'],float);velocity=np.asarray(obj['velocity'],float)
    name=method.replace('_pad','')
    if name=='hold':return origin
    linear=origin+dt*velocity
    if name=='cv':return linear
    if name=='ctrv':
        angle=dt*obj['yaw_rate'];a=dt*np.sinc(angle/np.pi);b=dt*.5*angle*np.sinc(angle/(2*np.pi))**2
        return origin+np.array([a*velocity[0]-b*velocity[1],b*velocity[0]+a*velocity[1]])
    if name!='gru':raise ValueError(name)
    knots=np.arange(7)*.5
    residual=np.vstack([np.zeros(2),np.asarray(obj['gru_residual'])])
    # Query actual target keyframe times; outside [0,3] only <=0.1 s endpoint hold.
    if not 0<=dt<=3.1:raise ValueError('unsupported prediction horizon')
    return linear+np.array([np.interp(dt,knots,residual[:,j]) for j in (0,1)])

def rectangle(center,size,yaw):
    dx,dy=GX-center[0],GY-center[1];c,s=np.cos(yaw),np.sin(yaw)
    return (np.abs(c*dx+s*dy)<=size[1]/2)&(np.abs(-s*dx+c*dy)<=size[0]/2)&ROI

def footprint_global(center,size,rotation,ego_translation,ego_rotation,padding=0.):
    # Full 3D rigid transform; planar bounding footprint in the current ego frame.
    center=np.asarray(center,float);size=np.asarray(size,float)
    r=np.asarray(ego_rotation,float).T@np.asarray(rotation,float)
    local=(center-np.asarray(ego_translation))@np.asarray(ego_rotation)
    # Match Quaternion.yaw_pitch_roll[0], used by the existing road Box raster.
    # atan2(R10,R00) is a different Euler convention when pitch/roll are nonzero.
    yaw=np.arctan2(-r[0,1],r[0,0]);extent=size.copy();extent[:2]+=2*padding
    # Retain original height-overlap convention used by road obstacle footprints.
    half_z=(np.abs(r[2])*size[[1,0,2]]).sum()/2
    if local[2]-half_z>2.2 or local[2]+half_z<-.5:return np.zeros((96,96),bool)
    return rectangle(local[:2],extent,yaw)

def predict_occupancy(objects,ego,dt,method):
    result=np.zeros((96,96),bool)
    margin=.25+.25*dt if method.endswith('_pad') else 0.
    for obj in objects:
        xy=trajectory(obj,dt,method)
        result |= footprint_global([xy[0],xy[1],obj['z']],obj['size'],obj['rotation'],ego['translation'],ego['rotation'],margin)
    return result

def release(probability,threshold,ground,blocked,future_occupancy):
    candidate=(probability>=threshold)&ground&~blocked&ROI
    return candidate&~future_occupancy

def tests():
    # Non-square rectangle catches nuScenes width/length and XY indexing errors.
    m=rectangle([5,0],[2,6,2],0)
    assert m[48,59] and not m[53,59]
    rotated=rectangle([0,5],[2,6,2],np.pi/2)
    assert rotated[59,48] and not rotated[59,53]
    ego=dict(translation=[100,200,0],rotation=[[0,-1,0],[1,0,0],[0,0,1]])
    a=footprint_global([100,205,1],[2,6,2],ego['rotation'],ego['translation'],ego['rotation'])
    assert np.array_equal(a,m)
    obj=dict(origin=[0,0],velocity=[2,0],yaw_rate=0.,gru_residual=np.zeros((6,2)).tolist())
    for t in (0,1,2,3,3.05):
        assert np.allclose(trajectory(obj,t,'cv'),[2*t,0])
        assert np.allclose(trajectory(obj,t,'cv'),trajectory(obj,t,'ctrv'))
        assert np.allclose(trajectory(obj,t,'cv'),trajectory(obj,t,'gru'))
    obj['yaw_rate']=np.pi/2
    assert np.allclose(trajectory(obj,1,'ctrv'),[4/np.pi,4/np.pi])
    p=np.ones((96,96));g=np.ones_like(p,bool);b=np.zeros_like(g);b[50,65]=True;g[51,65]=False
    pred=release(p,.5,g,b,np.zeros_like(g))
    assert not pred[50,65] and not pred[51,65] and not (pred&~ROI).any()
    return dict(status='PASS',width_length_axes=True,global_to_current_ego=True,cv_ctrv_limits=True,zero_gru_residual_parity=True,no_new_release_without_current_evidence=True)

if __name__=='__main__':print(tests())
