# 输入与准备顺序

1. nuScenes 的 metadata、samples、sweeps 及 expansion 地图。`NUSCENES_MAP_ROOT` 指向包含 `maps/expansion` 的目录。
2. 点云 shard 缓存和 READY 标记，以及 `road.py` 导出的 train/dev/val 栅格。原工程按数据 shard 筛选训练样本；迁移时应明确自己采用的样本清单，不能沿用旧统计数。
3. 原 BEVFusion 工程的 `bevfusion_full_val.py`、`bevfusion-pretrained.pth`、`full_val/nuscenes_infos_val.pkl` 与检测输出。上游推理框架和 CUDA 扩展须独立安装。
4. `road_full_validation.py`、`road_obstacle_check.py`、`road_obstacle_boxes.py` 及 `road_rgb_ablation.py` 所需几何/地图缓存。
5. `road_shared_features.py`、`road_shared_heads.py`、`road_200_data.py` 和 `road_200_baseline.py` 的特征、场景清单与基线权重。
6. `road_adaptive_data.py` 生成修正模块数据，再运行 README 中道路保留约束实验。

完整重跑需要按源脚本输入断言准备这些中间文件。本次发布未把数百 GB 数据、GPU 训练环境或私人缓存打包；已经执行的轻量检查见 VALIDATION.md。
