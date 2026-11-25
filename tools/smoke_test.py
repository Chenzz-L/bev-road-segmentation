"""Synthetic CPU checks: grid boundaries, empty LiDAR and road-head gradients."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from road import grid_points
from road_shared_heads import RoadHead
torch.set_num_threads(1)
empty=grid_points(np.empty((0,3)),np.empty((0,)))
assert empty.shape==(4,96,96) and np.isfinite(empty).all()
cloud=np.array([[0,0,0],[23.9,23.9,1],[24.1,0,0]],float)
grid=grid_points(cloud,np.array([128,128,128]))
assert np.count_nonzero(grid[0])==2
net=RoadHead(256)
pred=net(torch.randn(1,256,64,64))
assert pred.shape==(1,96,96) and torch.isfinite(pred).all()
torch.nn.functional.binary_cross_entropy_with_logits(pred,torch.zeros_like(pred)).backward()
assert net.project[0].weight.grad is not None
assert torch.isfinite(net.project[0].weight.grad).all()
print('PASS: empty/ROI LiDAR rasterization and road-head forward/backward')
