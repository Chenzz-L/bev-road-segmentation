"""Report protected-road results and revisit the preselected audit examples."""
import hashlib
import json
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from road_preserve_train import DEST
from road_adaptive_data import DEST as SOURCE,manifests


def main():
    seeds=(17,29,43)
    reports={s:json.loads((DEST/f"results_{s}.json").read_text()) for s in seeds}
    summary=dict(protocol=json.loads((DEST/"protocol.json").read_text()),locked=json.loads((DEST/"locked.json").read_text()),evaluation={})
    for split in ("val","reserved"):
        results={}
        for mode in ("baseline","previous_equal","recall_selection_only","preserve"):
            records=[reports[s]["evaluation"][split]["metrics"][mode] for s in seeds]
            results[mode]={k:dict(mean=mean(r[k] for r in records),values=[r[k] for r in records]) for k in ("iou","recall","walkway_raw_fp","walkway_pass_fp","offroad_pass_fp","missing_recall")}
        contrasts={}
        for ref in ("baseline","previous_equal","recall_selection_only"):
            a,b=results["preserve"],results[ref]
            contrasts[ref]=dict(iou_delta_pp=100*(a["iou"]["mean"]-b["iou"]["mean"]),
                recall_delta_pp=100*(a["recall"]["mean"]-b["recall"]["mean"]),
                walkway_pass_reduction_pct=100*(1-a["walkway_pass_fp"]["mean"]/b["walkway_pass_fp"]["mean"]),
                joint_improvement_seeds=[s for i,s in enumerate(seeds) if a["iou"]["values"][i]>b["iou"]["values"][i] and a["walkway_pass_fp"]["values"][i]<b["walkway_pass_fp"]["values"][i]],
                recall_within_02pp_seeds=[s for i,s in enumerate(seeds) if a["recall"]["values"][i]>=b["recall"]["values"][i]-.002])
        summary["evaluation"][split]=dict(metrics=results,contrasts=contrasts,changes={str(s):reports[s]["evaluation"][split]["changes"] for s in seeds})
    examples=[]
    for c in json.loads((DEST/"audit.json").read_text())["cases"]:
        split=c["split"];rows=manifests()[split];idx=[r["token"] for r in rows].index(c["token"])
        with np.load(SOURCE/f"{split}_geometry.npz") as d:y=d["y"][idx];walk=d["walk"][idx];free=d["ground"][idx]&~d["blocked"][idx]
        with np.load(DEST/f"{split}_17_masks.npz") as d:base,old,new=[d[k][idx] for k in ("baseline","previous_equal","preserve")]
        restored=base&~old&new&(y==1)
        file=DEST/f"compare_{split}_{c['scene']}_{idx}.png"
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        with Image.open(rows[idx]["images"]["CAM_FRONT"]["img_path"]) as im:axes[0].imshow(im)
        axes[0].set_title("CAM_FRONT: scene context")
        for ax,m,title in zip(axes[1:],(old,new),("Previous equal correction","Road-preserving correction")):
            image=np.zeros((*y.shape,3))+[.12,.16,.2];image[y==1]=[.65,.76,.72]
            image[base&~m&(y==1)]=[.95,.25,.2]
            image[m&walk&free]=[.65,.3,.95]
            if title.startswith("Road"):image[restored]=[.12,.8,.4]
            ax.imshow(np.rot90(image)[:,::-1]);ax.set_title(title)
        for ax in axes:ax.axis("off")
        fig.suptitle(f"{split} / {c['scene']} | red: new road miss; green: restored; purple: walkway pass FP",fontsize=10)
        fig.tight_layout();fig.savefig(file,dpi=130);plt.close(fig)
        examples.append(dict(split=split,scene=c["scene"],token=c["token"],file=file.name,
            previous_new_misses=int((base&~old&(y==1)).sum()),current_new_misses=int((base&~new&(y==1)).sum()),restored=int(restored.sum()),
            previous_walkway_pass=int((old&walk&free).sum()),current_walkway_pass=int((new&walk&free).sum())))
    summary["preselected_examples"]=examples
    root=Path(__file__).resolve().parent
    summary["source_sha256"]={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ("road_preserve_audit.py","road_preserve_train.py","road_preserve_summary.py")}
    (DEST/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({s:r["contrasts"] for s,r in summary["evaluation"].items()},indent=2))


if __name__=="__main__":main()
