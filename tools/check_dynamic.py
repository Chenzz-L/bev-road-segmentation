"""CPU geometry/model checks and integrity checks for the archived replay."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dynamic_road_core import tests
from road_log_head_control import CapacityHead

assert tests()['status'] == 'PASS'
torch.set_num_threads(1)
with torch.inference_mode():
    output = CapacityHead(96).eval()(torch.zeros(1, 256, 64, 64))
assert output.shape == (1, 96, 96) and torch.isfinite(output).all()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

folder = ROOT / 'results/rolling_dynamic_v1'
verification = read(folder / 'verification.json')
assert verification['status'] == 'PASS' and verification['frames'] == 40
assert verification['sensor_inputs_unchanged']
assert verification['fresh_road_inference_every_frame']
audit = read(folder / 'fresh_inference.json')
for key in ('feature_sha256', 'probability_sha256'):
    assert len({r[key] for r in audit['frames']}) == 40
assert all(r['ground_equal'] and r['blocked_equal'] and r['threshold_mask_iou'] > .99
           for r in audit['overlap_checks'])
for entry in read(ROOT / 'docs/dynamic-source-manifest.json'):
    actual = hashlib.sha256((ROOT / entry['file']).read_bytes()).hexdigest()
    assert actual == entry['published_sha256'], entry['file']
print('PASS: causal geometry, capacity-head forward, archived replay and publication hashes')
