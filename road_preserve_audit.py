"""Attribute newly deleted true-road cells before changing the training objective."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from road import centers,grid_points
from road_adaptive_data import DEST as SOURCE,manifests

DEST=SOURCE.parent/"road_preserve_v1"


def boundary(y):
    yy,xx=np.mgrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    return np.stack([(f==1)&binary_dilation(f==0,structure=disk) for f in y])


def main():
    DEST.mkdir(exist_ok=True)
    report={};cases=[]
    for split in ("val","reserved"):
        rows=manifests()[split]
        with np.load(SOURCE/f"{split}_geometry.npz") as d:
            y,ground,blocked=[d[k] for k in ("y","ground","blocked")]
            assert list(d["tokens"])==[r["token"] for r in rows]
        edge=boundary(y)
        distance=np.linalg.norm(centers,axis=2)
        groups=dict(ground=ground,no_ground=~ground,boundary_1m=edge,interior=~edge,
                    near_under_12m=np.broadcast_to(distance<12,y.shape),far_12m_plus=np.broadcast_to(distance>=12,y.shape),
                    blocked=blocked,not_blocked=~blocked)
        byseed={}
        for seed in (17,29,43):
            with np.load(SOURCE/f"{split}_{seed}_masks.npz") as p:
                base,equal=p["baseline"],p["equal"]
                assert list(p["tokens"])==[r["token"] for r in rows]
            deleted=base&~equal&(y==1);added=~base&equal&(y==1)
            ref=base&(y==1)
            stats={}
            for name,g in groups.items():
                n=int((deleted&g).sum());exposure=int((ref&g).sum())
                stats[name]=dict(deleted=n,share_of_deletions=n/int(deleted.sum()),baseline_true_road=exposure,
                                 deletion_rate=n/max(exposure,1),recovered=int((added&g).sum()))
            byseed[str(seed)]=dict(deleted=int(deleted.sum()),recovered=int(added.sum()),groups=stats)
            if seed==17:
                scores=(deleted&~ground&~edge).sum((1,2))
                choices=list(np.argsort(scores)[-2:][::-1])
                choices.append(int(np.argmax((deleted&ground&edge).sum((1,2)))))
                for i in dict.fromkeys(choices):
                    r=rows[i];out=DEST/f"audit_{split}_{r['scene']}_{i}.png"
                    fig,axes=plt.subplots(1,3,figsize=(13,4))
                    with Image.open(r["images"]["CAM_FRONT"]["img_path"]) as im:axes[0].imshow(im)
                    axes[0].set_title("CAM_FRONT: scene context")
                    road=np.zeros((*y[i].shape,3))+[.12,.16,.2]
                    road[y[i]==1]=[.65,.76,.72]
                    change=road.copy();change[deleted[i]]=[.95,.25,.2];change[added[i]]=[.12,.8,.4]
                    ground_view=road.copy();ground_view[deleted[i]&~ground[i]]=[.95,.4,.1]
                    ground_view[deleted[i]&ground[i]]=[.6,.3,.95]
                    for ax,img,title in zip(axes[1:],(change,ground_view),
                        ("Red: new road misses; green: recovered","New misses: orange=no ground, purple=ground")):
                        ax.imshow(np.rot90(img)[:,::-1],origin="upper");ax.set_title(title,fontsize=9)
                    for ax in axes:ax.axis("off")
                    fig.suptitle(f"{split} / {r['scene']} / {r['token'][:8]} | ego forward up in BEV")
                    fig.tight_layout();fig.savefig(out,dpi=130);plt.close(fig)
                    cases.append(dict(split=split,scene=r["scene"],token=r["token"],file=out.name,
                        deleted=int(deleted[i].sum()),no_ground=int((deleted[i]&~ground[i]).sum()),
                        boundary=int((deleted[i]&edge[i]).sum()),recovered=int(added[i].sum())))
        report[split]=byseed
    (DEST/"audit.json").write_text(json.dumps(dict(splits=report,cases=cases),indent=2))
    print(json.dumps({s:report[s]["17"] for s in report}),flush=True)


if __name__=="__main__":main()
