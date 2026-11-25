"""Test whether the fixed-budget road boundary experiment transfers to SegFormer-B0."""
import argparse
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from audit_road_density import boundary_band
from road import OUT, UNet
from road_generalize import WORK, fitting, read, tensors
from road_generalize_report import SEEDS, as_dict, contour, frame_counts, metrics


class RoadSegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        config = SegformerConfig(num_channels=6, num_labels=1, hidden_sizes=[32,64,160,256],
                                 depths=[2,2,2,2], decoder_hidden_size=256)
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


@torch.inference_mode()
def latency(model, x):
    for _ in range(30):
        model(x)
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(120):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(x)
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return {"median_ms":float(np.median(elapsed)),"p95_ms":float(np.percentile(elapsed,95))}


def evaluate():
    model = RoadSegFormer().cuda().eval()
    prior = json.loads((WORK/"results.json").read_text())
    result = {"seeds":list(SEEDS), "training_updates":18*32,
              "parameters":{"segformer_b0":sum(p.numel() for p in model.parameters()),
                            "unet":sum(p.numel() for p in UNet().parameters())}}
    for split, path in (("val", OUT/"val.npz"), ("fresh", WORK/"fresh.npz")):
        data = read(path)
        x,y = tensors(data)[0],data["y"]
        band = boundary_band(y)
        gt = [contour(row==1,row!=255) for row in y]
        part = {}
        for loss in ("plain","edge"):
            rows = []
            for seed in SEEDS:
                cp = torch.load(WORK/f"segformer_full_{loss}_{seed}.pt",map_location="cpu")
                model.load_state_dict(cp["weights"])
                with torch.no_grad():
                    pred = torch.cat([model(t)[:,0].sigmoid()>=cp["threshold"] for t in x.split(20)]).cpu().numpy()
                rows.append(metrics(frame_counts(pred,y,band,gt).sum(0)))
            part["segformer_"+loss] = {"mean":as_dict(np.mean(rows,axis=0)),
                                        "per_seed":[as_dict(row) for row in rows]}
            part["unet_"+loss] = prior[split]["methods"]["full_"+loss]["mean"]
        result[split] = part
    small = UNet().cuda().eval()
    small.load_state_dict(torch.load(WORK/"full_edge_17.pt",map_location="cpu")["weights"])
    model.load_state_dict(torch.load(WORK/"segformer_full_edge_17.pt",map_location="cpu")["weights"])
    result["batch1_forward_fp32"] = {"device":torch.cuda.get_device_name(),
                                     "unet":latency(small,x[:1]),"segformer_b0":latency(model,x[:1])}
    (WORK/"segformer_comparison.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result["parameters"]))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job",choices=("plain","edge","evaluate"))
    parser.add_argument("--seed",type=int,default=17)
    args = parser.parse_args()
    if args.job=="evaluate":
        evaluate()
    else:
        fitting("full",args.job,args.seed,RoadSegFormer,"segformer_")
