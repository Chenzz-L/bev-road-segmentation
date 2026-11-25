"""Paired density and road-boundary breakdown of existing segmentation checkpoints."""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from road_camera import MODES, OUT, WORK, UNet, gx, gy, limit


def predict(mode, split, base, cameras):
    ckpt = torch.load(WORK / (mode+".pt"), map_location="cpu")
    model = UNet()
    model.a[0] = nn.Conv2d(10, 16, 3, padding=1)
    model.load_state_dict(ckpt["weights"])
    model.cuda().eval()
    lidar = base["x"].astype("float32")
    rgb = cameras.astype("float32")/255
    if mode == "lidar":
        rgb.fill(0)
    if mode == "camera":
        lidar.fill(0)
    xy = np.broadcast_to(np.stack((gx,gy))/limit,(len(lidar),2,*gx.shape))
    x = np.concatenate((lidar,rgb,xy), axis=1).astype("float32")
    with torch.no_grad():
        prob = torch.cat([model(batch.cuda())[:,0].sigmoid().cpu()
                          for batch in torch.from_numpy(x).split(20)]).numpy()
    if split == "val":
        with np.load(WORK / (mode+"_pred.npz")) as saved:
            assert np.array_equal(base["scene"], saved["scene"])
            assert np.array_equal(prob >= ckpt["threshold"], saved["masks"])
    return prob, ckpt["threshold"]


def boundary_band(y):
    # Opposite-class grid centers within 1 m, using only valid labels inside the image.
    yy,xx = np.mgrid[-2:3,-2:3]
    disk = ((xx*xx+yy*yy)<=4).astype("uint8")
    result = np.empty(y.shape, dtype=bool)
    for i, frame in enumerate(y):
        road, background = frame==1, frame==0
        spread = lambda mask: cv2.dilate(mask.astype("uint8"),disk,
                                          borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        result[i] = (road & spread(background)) | (background & spread(road))
    return result


def aggregate(a, per_frame, n):
    result = np.zeros((n,a.shape[1]),dtype=np.int64)
    np.add.at(result,per_frame,a)
    return result


def miou(c):
    tp,fp,fn,tn = c.T
    return .5*(tp/(tp+fn+fp)+tn/(tn+fn+fp))


def measure(selected, y, pred, index, draws):
    road = selected & (y==1)
    background = selected & (y==0)
    nr = road.sum((1,2))
    nb = background.sum((1,2))
    out = {"cells":int(nr.sum()+nb.sum()),"frames":int(np.sum(nr+nb>0))}
    boot = {}
    for mode, mask in pred.items():
        tp = (mask & road).sum((1,2))
        fp = (mask & background).sum((1,2))
        c = aggregate(np.column_stack((tp, fp, nr-tp, nb-fp)),index,draws.shape[1])
        totals = c.sum(0)
        tp,fp,fn,tn = map(int,totals)
        out[mode] = dict(TP=tp,FP=fp,FN=fn,TN=tn,mIoU=float(miou(totals)),
                         miss_rate=fn/(tp+fn),false_positive_rate=fp/(tn+fp))
        boot[mode] = miou(draws@c)
    lidar,fused = pred["lidar"],pred["fusion"]
    fixed = (~lidar & road & fused) | (lidar & background & ~fused)
    broken = (lidar & road & ~fused) | (~lidar & background & fused)
    out["fixed"] = int(fixed.sum())
    out["broken"] = int(broken.sum())
    assert out["broken"]-out["fixed"] == sum(out["fusion"][k]-out["lidar"][k] for k in ["FP","FN"])
    out["delta_mIoU_pp"] = 100*(out["fusion"]["mIoU"]-out["lidar"]["mIoU"])
    out["delta_ci95_pp"] = np.percentile(100*(boot["fusion"]-boot["lidar"]),[2.5,97.5]).tolist()
    return out


def chart(groups):
    names = ("empty","one_two","three_plus")
    labels = ("0 returns","1–2 returns","3+ returns")
    fig, axes = plt.subplots(1,2,figsize=(8.4,3.2),sharey=True)
    for ax, area, title in zip(axes,("edge","interior"),("Within 1 m of opposite class","Remaining cells")):
        for i,name in enumerate(names):
            r = groups[name+"_"+area]
            lo,hi = r["delta_ci95_pp"]
            d = r["delta_mIoU_pp"]
            ax.plot([lo,hi],[i,i],color="#46627d",lw=2)
            ax.scatter(d,i,color="#9c3d28",zorder=3)
        ax.axvline(0,color="#444",lw=1,ls="--")
        ax.set_title(title,fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Fusion minus LiDAR mIoU (pp)")
        ax.grid(axis="x",alpha=.18)
    axes[0].invert_yaxis()
    fig.suptitle("Paired differences with 95% scene-bootstrap intervals")
    fig.tight_layout()
    fig.savefig(WORK/"density_boundary.png",dpi=165)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    modes = {}
    alternative = {}
    for split in ("dev","val"):
        base = np.load(OUT / (split+".npz"))
        with np.load(WORK / (split+".npz")) as data:
            assert np.array_equal(base["token"],data["token"])
            cameras = data["features"]
        prob, threshold = {}, {}
        for mode in MODES:
            prob[mode],threshold[mode] = predict(mode,split,base,cameras)
        y = base["y"]
        if split=="dev":
            road = y==1
            recall = np.mean(prob["lidar"][road]>=threshold["lidar"])
            matched = {m:float(np.quantile(prob[m][road],1-recall)) for m in MODES}
            matched["lidar"] = threshold["lidar"]
            dev_recall = {m:float(np.mean(prob[m][road]>=matched[m])) for m in MODES}
        else:
            modes = {m:prob[m]>=threshold[m] for m in MODES}
            alternative = {m:prob[m]>=matched[m] for m in MODES}
    scenes, index = np.unique(base["scene"],return_inverse=True)
    draws = np.random.default_rng(37).multinomial(len(scenes),np.full(len(scenes),1/len(scenes)),size=2000)
    log_count = base["x"][:,0].astype("float32")
    returns = np.rint(np.expm1(3*log_count)).astype(int)
    valid = y!=255
    masks = dict(overall=valid)
    edge = boundary_band(y)
    masks.update(edge=edge,interior=valid & ~edge)
    with np.load(OUT/"train.npz") as train:
        pattern = train["x"][:,0]>0
        cuts = np.quantile(pattern.mean((1,2)),[1/3,2/3])
    bins = np.digitize((returns>0).mean((1,2)),cuts)
    for j, key in enumerate(("frame_sparse","frame_mid","frame_dense")):
        masks[key] = valid & (bins[:,None,None]==j)
    for key,selected in [("empty",returns==0),("one_two",(returns==1)|(returns==2)),
                         ("three_plus",returns>=3)]:
        masks[key] = valid & selected
        masks[key+"_edge"] = edge & selected
        masks[key+"_interior"] = masks["interior"] & selected
    report = dict(frames=len(y),scenes=len(scenes),frame_coverage_cuts=cuts.tolist(),
                  original_thresholds=threshold,dev_recall_matched_thresholds=matched,
                  matched_dev_road_recall=dev_recall)
    for name,pred in (("original",modes),("matched_dev_recall",alternative)):
        groups = {key:measure(group,y,pred,index,draws) for key,group in masks.items()}
        for keys in (("edge","interior"),("empty","one_two","three_plus"),
                     ("frame_sparse","frame_mid","frame_dense")):
            for metric in ("cells","fixed","broken"):
                assert sum(groups[key][metric] for key in keys)==groups["overall"][metric]
        report[name] = groups
    chart(report["original"])
    (WORK/"density_boundary.json").write_text(json.dumps(report,indent=2))
    print("Verified original masks, exhaustive bins and paired error counts.")


if __name__=="__main__":
    main()
