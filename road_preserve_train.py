"""Protect baseline-correct road logits, select only under development recall constraints."""
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from road_adaptive_train import load_split,new_model,forward,predict,metrics,paired
from road_adaptive_data import DEST as SOURCE
from road_200_data import DEST as TRAIN200
from road_shared_heads import RoadHead

DEST=TRAIN200/"road_preserve_v1"
STRENGTHS=(0,.25,1,4)
THETA=float(np.log(.6/.4))


def key_name(strength):return "p"+str(strength).replace(".","_")


def setup():
    DEST.mkdir(exist_ok=True)
    protocol=dict(seeds=[17,29,43],strengths=list(STRENGTHS),threshold=.6,epochs=12,lr=.0003,batch_size=24,
        architecture="unchanged equal-weight residual model from adaptive_v1; frozen 200-scene baseline",
        loss="BCE + lambda * weighted mean squared downward logit change on baseline-correct true road",
        protection="true road AND baseline probability >= 0.6; weight 2 without ground, 1 with ground; training labels only",
        checkpoint_selection="dev IoU >= baseline, walkway pass FP <= baseline, recall >= baseline - 0.002; maximize IoU; fallback to unchanged baseline if none",
        strength_selection="among .25,1,4 maximize number of seeds with feasible improved dev IoU, then mean dev IoU, then minimize mean walkway pass FP",
        control="lambda 0 uses the same recall-constrained checkpoint selection",
        evaluation="evaluate only lambda 0 and dev-selected lambda, on existing 150/27-scene sets after all training",
        validation_caveat="Existing evaluation sets have been inspected before; this is not a new blind test")
    path=DEST/"protocol.json"
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2))
    print(json.dumps(protocol),flush=True)


def baseline():
    weights=torch.load(TRAIN200/"baseline_v1/road_head.pt",map_location="cpu")["weights"]
    net=RoadHead(256).cuda().to(memory_format=torch.channels_last).eval()
    net.load_state_dict(weights);net.requires_grad_(False)
    return net,weights


def fit(seed,strength,tr,dev,weights):
    net=new_model("equal",seed,weights)
    opt=torch.optim.AdamW(net.parameters(),lr=.0003)
    order=torch.Generator().manual_seed(seed)
    ref=metrics((dev["original"]>=THETA).cpu().numpy(),dev)
    # An unchanged baseline is a valid fallback, explicitly reported as epoch zero.
    selected=dict(epoch=0,fallback=True,feasible=True,**ref)
    best=(ref["iou"],{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},selected)
    protected=(tr["target"]==1)&(tr["original"]>=THETA)
    protection_weight=torch.tensor(2.-tr["ground"].astype(float),dtype=torch.float32,device="cuda")
    history=[]
    for epoch in range(1,13):
        net.train()
        for idx in torch.randperm(len(tr["rows"]),generator=order).split(24):
            z,_=forward(net,"equal",tr,idx)
            y=tr["target"][idx];valid=y!=255
            loss=F.binary_cross_entropy_with_logits(z[valid],y[valid].float())
            if strength:
                protect=protected[idx]
                w=protection_weight[idx]*protect
                downward=F.relu(tr["original"][idx]-z).square()
                loss=loss+strength*(downward*w).sum()/w.sum().clamp_min(1)
            opt.zero_grad();loss.backward();opt.step()
        z,_=predict(net,"equal",dev)
        stats=metrics(z>=THETA,dev)
        feasible=(stats["iou"]>=ref["iou"] and stats["walkway_pass_fp"]<=ref["walkway_pass_fp"]
                  and stats["recall"]>=ref["recall"]-.002)
        record=dict(epoch=epoch,fallback=False,feasible=feasible,**stats)
        history.append(record)
        if feasible and stats["iou"]>best[0]:
            best=(stats["iou"],{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},record)
        print(seed,strength,epoch,round(stats["iou"],5),round(stats["recall"]-ref["recall"],5),stats["walkway_pass_fp"],feasible,flush=True)
    torch.save(dict(weights=best[1],selected=best[2],strength=strength,seed=seed),DEST/f"{key_name(strength)}_{seed}.pt")
    return dict(selected=best[2],dev_baseline=ref,history=history,
                best_unconstrained=max(history,key=lambda r:r["iou"]))


def train(seed):
    base,weights=baseline()
    tr=load_split("train",base);dev=load_split("dev",base)
    records={}
    for strength in STRENGTHS:
        records[key_name(strength)]=fit(seed,strength,tr,dev,weights)
        (DEST/f"selection_{seed}.json").write_text(json.dumps(records,indent=2))
    print("TRAIN_DONE",seed,flush=True)


def lock():
    records={s:json.loads((DEST/f"selection_{s}.json").read_text()) for s in (17,29,43)}
    ranking={}
    for strength in STRENGTHS[1:]:
        rows=[records[s][key_name(strength)] for s in records]
        ranking[str(strength)]=[sum(r["selected"]["iou"]>r["dev_baseline"]["iou"] for r in rows),
                               float(np.mean([r["selected"]["iou"] for r in rows])),
                               -float(np.mean([r["selected"]["walkway_pass_fp"] for r in rows]))]
    chosen=max(STRENGTHS[1:],key=lambda s:ranking[str(s)])
    result=dict(strength=chosen,ranking=ranking,seeds=[17,29,43],threshold=.6)
    path=DEST/"locked.json"
    if path.exists():assert json.loads(path.read_text())==result
    else:path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def evaluate(seed):
    chosen=json.loads((DEST/"locked.json").read_text())["strength"]
    base,weights=baseline()
    report=dict(seed=seed,strength=chosen,evaluation={})
    for split in ("val","reserved"):
        d=load_split(split,base)
        original=(d["original"]>=THETA).cpu().numpy()
        with np.load(SOURCE/f"{split}_{seed}_masks.npz") as old:
            assert list(old["tokens"])==[r["token"] for r in d["rows"]]
            previous=old["equal"]
            replay=int((original!=old["baseline"]).sum())
        results={"baseline":metrics(original,d),"previous_equal":metrics(previous,d)}
        preds={}
        for name,strength in (("recall_selection_only",0),("preserve",chosen)):
            ckpt=torch.load(DEST/f"{key_name(strength)}_{seed}.pt",map_location="cpu")
            net=new_model("equal",seed,weights);net.load_state_dict(ckpt["weights"])
            z,_=predict(net,"equal",d);m=z>=THETA
            preds[name]=m
            results[name]=dict(fallback=ckpt["selected"]["fallback"],**metrics(m,d))
        np.savez_compressed(DEST/f"{split}_{seed}_masks.npz",tokens=[r["token"] for r in d["rows"]],baseline=original,previous_equal=previous,**preds)
        changes={}
        for name,m in preds.items():
            changes[name]=dict(
                deleted_true_road=int((original&~m&(d["y"]==1)).sum()),
                recovered_true_road=int((~original&m&(d["y"]==1)).sum()),
                restored_previous_misses=int((original&~previous&m&(d["y"]==1)).sum()))
        report["evaluation"][split]=dict(metrics=results,replayed_baseline_changed_cells=replay,changes=changes,
            paired={name:paired(m,original,d) for name,m in preds.items()},
            preserve_vs_control=paired(preds["preserve"],preds["recall_selection_only"],d))
        del d
    (DEST/f"results_{seed}.json").write_text(json.dumps(report,indent=2))
    print("EVAL_DONE",seed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("job",choices=("setup","train","lock","evaluate"));p.add_argument("--seed",type=int,default=17)
    a=p.parse_args()
    if a.job=="setup":setup()
    elif a.job=="train":train(a.seed)
    elif a.job=="lock":lock()
    else:evaluate(a.seed)
