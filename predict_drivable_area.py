"""Run a saved BEV road segmenter on a raw nuScenes keyframe."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion
import torch

from road import DATA, encode, gx, gy, limit
from road_scale_train import new_model
from road_vector_labels import VectorMap


def run(args):
    nusc=NuScenes("v1.0-trainval",str(args.dataroot),verbose=False)
    scene=next(s for s in nusc.scene if s["name"]==args.scene)
    if not 0<=args.frame<scene["nbr_samples"]:
        raise ValueError(f"Frame must lie in [0,{scene['nbr_samples']-1}] for {args.scene}")
    token=scene["first_sample_token"]
    for _ in range(args.frame):token=nusc.get("sample",token)["next"]
    lidar=nusc.get("sample",token)["data"]["LIDAR_TOP"]
    sd=nusc.get("sample_data",lidar)
    calib=nusc.get("calibrated_sensor",sd["calibrated_sensor_token"])
    raw=encode(args.dataroot/sd["filename"],calib)
    inp=np.concatenate([raw,np.stack((gx,gy))/limit]).astype(np.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp=torch.load(args.checkpoint,map_location="cpu")
    model=new_model(cp["model"]).eval().to(device)
    model.load_state_dict(cp["weights"])
    with torch.inference_mode():
        probability=model(torch.from_numpy(inp[None]).to(device))[0,0].sigmoid().cpu().numpy()
    mask=probability>=cp["threshold"]
    args.output.mkdir(parents=True,exist_ok=True)
    prefix=args.output/f"{scene['name']}_{args.frame:02d}"
    np.savez_compressed(prefix.with_suffix(".npz"),probability=probability,mask=mask,
                        token=lidar,threshold=cp["threshold"],cell_m=.5)
    # Map and global pose are used only to show a reference after inference.
    images=[raw[0]];titles=["LiDAR returns"]
    if args.map_root:
        p=nusc.get("ego_pose",sd["ego_pose_token"])
        rot=Quaternion(p["rotation"]).rotation_matrix
        pose=(*p["translation"][:2],np.arctan2(rot[1,0],rot[0,0]))
        region=nusc.get("log",scene["log_token"])["location"]
        truth=VectorMap(region,args.map_root).label(pose)
        images.append(np.ma.masked_where(truth==255,truth));titles.append("Map reference")
    images.append(mask);titles.append("Prediction")
    fig,axes=plt.subplots(1,len(images),figsize=(2.8*len(images),3.1))
    colors=ListedColormap(["#233849","#edb960"])
    for i,(ax,img,title) in enumerate(zip(axes,images,titles)):
        ax.imshow(img.T,origin="lower",cmap="gray" if i==0 else colors,vmin=0,vmax=1)
        ax.invert_xaxis();ax.set(xticks=[],yticks=[],title=title)
    fig.suptitle(f"{args.scene} / frame {args.frame} / gold = drivable area")
    fig.tight_layout();fig.savefig(prefix.with_suffix(".png"),dpi=155);plt.close(fig)
    print(lidar,prefix,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataroot",type=Path,default=DATA)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--scene",default="scene-0928")
    p.add_argument("--frame",type=int,default=12)
    p.add_argument("--map-root",type=Path,help="Add the map reference panel using nuScenes map expansion assets")
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
