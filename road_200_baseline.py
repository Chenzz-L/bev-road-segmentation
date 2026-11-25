"""Freeze, replay, and apply the 200-scene road baseline in the server project."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from road_shared_features import OUT, CACHE, PROJECT
from road_shared_heads import RoadHead

EXPERIMENT = OUT / "train200"
BUNDLE = EXPERIMENT / "baseline_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def freeze():
    candidates = [json.loads((EXPERIMENT / f"selection_{s}.json").read_text())["expanded"]
                  for s in (17, 29, 43)]
    selected = max(candidates, key=lambda r: (r["dev_road_iou"], -r["seed"]))
    BUNDLE.mkdir(exist_ok=True)
    source = EXPERIMENT / f"expanded_{selected['seed']}.pt"
    target = BUNDLE / "road_head.pt"
    if target.exists():
        assert digest(target) == digest(source), "Existing frozen checkpoint differs"
    else:
        shutil.copy2(source, target)
    inputs = [EXPERIMENT / "manifest.json", EXPERIMENT / "protocol.json",
              PROJECT / "bevfusion-pretrained.pth", PROJECT / "bevfusion_full_val.py"]
    inputs += [Path(__file__).parent / name for name in
               ("road_shared_heads.py", "road_shared_features.py", "road_obstacle_check.py",
                "road_obstacle_boxes.py", "road_200_data.py", "road_200_train.py")]
    config = dict(version="road200-v1", selection_rule="maximum development IoU among seeds 17,29,43",
                  seed=selected["seed"], checkpoint_step=selected["step"],
                  threshold=selected["threshold"], dev_iou=selected["dev_road_iou"],
                  train_scenes=200, train_frames=2400, frozen_bevfusion=True,
                  auxiliary_or_residual_head=False, checkpoint_sha256=digest(target),
                  feature=dict(shape=[256,64,64], source="fusion_layer before BEV backbone",
                               modalities="six cameras plus current LiDAR sweep",
                               extent_m=[-24,24,-24,24], cell_m=.75),
                  output=dict(shape=[96,96], cell_m=.5, row="ego y", column="ego x",
                              passable="road & ground & ~blocked",
                              unknown_road="road & ~ground & ~blocked",
                              blocked="existing elevated LiDAR or predicted vehicle footprint"),
                  reference_inputs_sha256={str(p):digest(p) for p in inputs})
    path = BUNDLE / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == config, "Frozen config differs; create a new version"
    else:
        path.write_text(json.dumps(config, indent=2))
    print(json.dumps({k:config[k] for k in ("version","seed","threshold","dev_iou")}), flush=True)


class Baseline:
    def __init__(self, bundle=BUNDLE, device="cuda"):
        self.config = json.loads((bundle / "config.json").read_text())
        assert digest(bundle / "road_head.pt") == self.config["checkpoint_sha256"]
        self.device = device
        self.net = RoadHead(256).to(device).to(memory_format=torch.channels_last).eval()
        ckpt = torch.load(bundle / "road_head.pt", map_location="cpu")
        self.net.load_state_dict(ckpt["weights"])
        assert ckpt["threshold"] == self.config["threshold"]

    @torch.inference_mode()
    def predict(self, features, ground, blocked):
        features = np.asarray(features)
        if features.ndim == 3:
            features, ground, blocked = features[None], ground[None], blocked[None]
        if features.shape[1:] == (592,64,64):
            features = features[:,336:592]
        assert features.shape[1:] == (256,64,64) and np.isfinite(features).all()
        ground, blocked = np.asarray(ground, bool), np.asarray(blocked, bool)
        assert ground.shape == blocked.shape == (len(features),96,96)
        x = torch.tensor(features.copy(), device=self.device, dtype=torch.float32)
        probability = self.net(x.contiguous(memory_format=torch.channels_last)).sigmoid().cpu().numpy()
        road = probability >= self.config["threshold"]
        return dict(probability=probability, road=road, ground=ground, blocked=blocked,
                    passable=road & ground & ~blocked, unknown_road=road & ~ground & ~blocked)


def verify(reserved_device=None):
    baseline = Baseline()
    seed = baseline.config["seed"]
    for path, expected in baseline.config["reference_inputs_sha256"].items():
        assert digest(path) == expected, f"Reference changed: {path}"
    report = {}
    for name in ("validation150", "reserved27"):
        if name == "validation150":
            rows = [r for r in json.loads((OUT / "manifest.json").read_text()) if r["split"] == "val"]
            paths = [CACHE / (r["token"] + ".npy") for r in rows]
            with np.load(OUT / "references.npz") as d:
                assert list(d["tokens"]) == [r["token"] for r in rows]
                ground, blocked, y = d["ground"], d["blocked"], d["label"]
            with np.load(EXPERIMENT / f"expanded_{seed}_masks.npz") as d:
                assert list(d["tokens"]) == [r["token"] for r in rows]
                expected = d["masks"]
        else:
            if reserved_device is not None:
                baseline = Baseline(device=reserved_device)
            frozen = OUT / "passable_walkway/frozen"
            rows = json.loads((frozen / "new_manifest.json").read_text())
            paths = [EXPERIMENT / "eval27_features" / (r["token"] + ".npy") for r in rows]
            refs = []
            for r in rows:
                with np.load(frozen / "frames" / (r["token"] + ".npz")) as d:
                    refs.append([d[k] for k in ("ground","blocked","y")])
            ground, blocked, y = [np.stack([r[i] for r in refs]) for i in range(3)]
            with np.load(EXPERIMENT / f"eval27_{seed}_masks.npz") as d:
                assert list(d["tokens"]) == [r["token"] for r in rows]
                expected = d["expanded"]
        outputs = []
        for start in range(0,len(rows),24):
            features = []
            for p in paths[start:start+24]:
                a = np.load(p, mmap_mode="r")
                features.append((a[336:592] if a.shape[0] == 592 else a).copy())
            outputs.append(baseline.predict(np.stack(features),ground[start:start+24],blocked[start:start+24]))
        result = {k:np.concatenate([r[k] for r in outputs]) for k in outputs[0]}
        road = result["road"]
        assert np.array_equal(road, expected), f"Replay mismatch in {name}"
        assert not (result["passable"] & blocked).any()
        assert np.array_equal(result["passable"] | result["unknown_road"], road & ~blocked)
        np.savez_compressed(BUNDLE / f"{name}_outputs.npz",tokens=[r["token"] for r in rows],**result)
        report[name] = dict(frames=len(rows),scenes=len({r["scene"] for r in rows}),
                            gpu=torch.cuda.get_device_name(baseline.device),
                            changed_road_cells=int((road != expected).sum()),
                            iou=float((road & (y==1)).sum()/((road | (y==1)) & (y!=255)).sum()),
                            passable_cells=int(result["passable"].sum()),
                            unknown_road_cells=int(result["unknown_road"].sum()))
        print(name,report[name],flush=True)
    (BUNDLE / "verification.json").write_text(json.dumps(report,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job",choices=("freeze","verify","predict"))
    p.add_argument("--feature",type=Path)
    p.add_argument("--reference",type=Path,help="NPZ containing aligned ground and blocked masks")
    p.add_argument("--token",help="Required for a multi-frame reference NPZ")
    p.add_argument("--output",type=Path)
    p.add_argument("--reserved-device",help="Device used for the original reserved-set evaluation")
    a = p.parse_args()
    if a.job == "freeze":freeze()
    elif a.job == "verify":verify(a.reserved_device)
    else:
        if not all((a.feature,a.reference,a.output)):p.error("predict requires --feature, --reference, --output")
        with np.load(a.reference) as d:
            if "tokens" in d:
                if a.token is None:p.error("multi-frame reference requires --token")
                assert a.feature.stem == a.token, "Feature and reference token differ"
                i = list(d["tokens"]).index(a.token)
                ground, blocked = d["ground"][i], d["blocked"][i]
            else:
                assert a.feature.stem == a.reference.stem, "Feature and reference token differ"
                ground, blocked = d["ground"], d["blocked"]
        output = Baseline().predict(np.load(a.feature),ground,blocked)
        a.output.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(a.output,token=a.feature.stem,**{k:v[0] for k,v in output.items()})
        print(str(a.output),flush=True)


if __name__ == "__main__":main()
