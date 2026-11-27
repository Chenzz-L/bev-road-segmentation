# 连续道路感知与动态占据

## 两种演示的区别

| 产物 | 时间轴含义 | 道路预测 |
|---|---|---|
| [连续回放](../results/rolling_dynamic_v1/report.html) | 40 个实际采集关键帧，约 19.50 秒 | 每帧重新提取特征并推理 |
| [单锚点时域查看器](../results/dynamic_occupancy_v1/report.html) | 同一帧向未来 0 / 1 / 2 / 3 秒预测 | 当前帧道路保持固定 |

GitHub 文件页不会直接运行 HTML。克隆仓库后打开对应 `report.html`，或下载 HTML 到本地浏览器；图像已内嵌，播放不依赖服务器。关联 JSON 和 GIF 位于同一目录。连续回放支持暂停、拖动帧和倍速；README 中的 GIF 可直接预览。

连续回放固定使用之前演示的 `scene-0798`，没有根据道路预测好坏重新选场景。自车实际行驶 128.516 米；局部 BEV 中自车位于中心，全局视图展示其已走过的轨迹。全局视图的画幅范围用完整片段定位，仅用于离线排版，不输入预测器。

## 模型与处理链

1. 使用当前关键帧六路图像和单帧 LiDAR，运行冻结的预训练 BEVFusion；点云第五维按既有提取方式置零。
2. 复用 `road_log_head_control.CapacityHead(96)` 的融合头，种子 17，阈值 0.35；这与 README 中历史道路保护残差修正实验是不同模型版本。它不代表增加图像或动态模块使道路 IoU 获得了新提升。
3. 由当前点云地面证据、离地回波和当前检测框计算候选区：`道路概率达阈值 AND 有地面证据 AND 未被当前障碍物阻挡 AND 非自车区域`。
4. 复用[雷达跟踪与轨迹预测项目](https://github.com/Chenzz-L/bevfusion-radar-tracking)中对应时间戳的因果状态和 GRU 输出。保留当前小汽车检测，包括没有 GT 匹配的检测；没有轨迹预测时使用当前检测速度的 CV 回退。
5. 将预测位置和当前检测框尺寸、朝向变换到当前自车坐标系，生成未来时刻占据。最终候选区再排除该占据；它是瞬时占据约束，不包含两时刻之间的完整扫掠路径。

地图标签不进入回放推理。原始预测缓存含评估元数据，但这里只读取 `tracking_id`、`used_gru`、`predictions`；目标标签、误差、GT 实例匹配不用于筛选。当前未知或受阻区域仍保持限制。没有未来自车位姿输入，也没有轨迹规划或闭环控制。

## 已完成的核查

- 40 帧特征指纹和道路概率指纹分别全部不同；每帧输入为对应的图像和点云。
- 与历史实验重合的 4 帧，道路阈值掩码 IoU 最低 0.999659；地面及当前障碍物掩码完全一致。浮点差异详情保留在 [fresh_inference.json](../results/rolling_dynamic_v1/fresh_inference.json)。
- 原权重、阈值、源数据及评估文件指纹保持不变。本次没有训练或重新选参。
- 动态占据评估的独立核查包含 5,379 条预测重放、1,042 次 SDK 框栅格对比和 43,686 组计数向量；见 [独立验证](../results/dynamic_occupancy_v1/independent_verification.json)。

这些核查证明数据流和复现一致性，不等于真实道路安全性或融合增益证明。

## 动态占据的已有评估

沿用原 150 场景 / 542 锚点，按各预测时域能否获得有效未来关键帧分别评估。下表为三个固定道路头种子均值、无额外框膨胀的 GRU 相对 CTRV 对照；冲突是放行栅格与未来标注车辆占据的重叠计数，覆盖是评估口径下的有效道路覆盖。

| 预测时域 | 有效锚点 | 冲突栅格相对减少 | 覆盖差值 |
|---|---:|---:|---:|
| 1 秒 | 376 | 7.24% | +0.079 个百分点 |
| 2 秒 | 352 | 8.34% | +0.116 个百分点 |
| 3 秒 | 348 | 4.51% | +0.080 个百分点 |

各时域涉及 148 场景 / 18 logs。3 秒冲突减少的 log bootstrap 区间包含零，不据此声称稳定显著收益。原验证集已查看过，跟踪参数也曾在验证集调整；27 个保留道路场景与 GRU 训练 / 开发场景存在重叠，未用作独立动态评估。完整方法、逐 log / 种子结果、膨胀敏感性与限制见 [summary.json](../results/dynamic_occupancy_v1/summary.json) 和 [protocol.json](../results/dynamic_occupancy_v1/protocol.json)。

## 环境与输入

复用已配置的 Linux MMDetection3D v1.4.0 / BEVFusion CUDA 环境。新增可视化依赖为 Pillow，中文字体可通过 `ROAD_CJK_FONT` 指定；默认使用 Noto Serif CJK。环境变量见 [`.env.example`](../.env.example)，需要在 shell 导出，不会自动加载。

```bash
export NUSCENES_ROOT=/path/to/nuscenes
export BEVFUSION_PROJECT=/path/to/BEVFusion
export ROAD_WORK_DIR=/path/to/road_outputs
export ROAD_LOG_HEAD_DIR=$ROAD_WORK_DIR/road_refinement/shared_bev/train200/log_head_capacity_v1
export TRAJECTORY_PROJECT=/path/to/trajectory_prediction
export ROAD_CJK_FONT=/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc
export PYTHONPATH=$BEVFUSION_PROJECT/mmdetection3d-1.4.0:$PYTHONPATH
```

必须预先准备以下产物，源码仓库不包含数据、权重或大数组：

| 位置 | 输入 |
|---|---|
| `NUSCENES_ROOT` | `v1.0-trainval` 元数据、scene-0798 的六路相机与 LiDAR 关键帧；批量动态评估还需标注 |
| `BEVFUSION_PROJECT` | `bevfusion-pretrained.pth`、`bevfusion_full_val.py` 及其基础配置、`mmdetection3d-1.4.0` |
| `BEVFUSION_PROJECT/full_val/predictions/pred_instances_3d` | `results_nusc.json` |
| `ROAD_LOG_HEAD_DIR` | `wide/fusion/17.pt`、`selection_17.json`、`val_manifest.json`、`val_references.npz`、`val_wide_fusion_17_prob.npy` |
| `TRAJECTORY_PROJECT/outputs/val_v1` | `penalty8_states.jsonl.gz`、`penalty8_forecasts.jsonl.gz` |
| `TRAJECTORY_PROJECT/outputs/gru_val_v1` | `forecasts.jsonl.gz` |

连续回放按顺序执行：

```bash
python rolling_dynamic_road.py prepare
python rolling_dynamic_road.py infer
python rolling_dynamic_road.py motion
python render_rolling_dynamic.py
```

`prepare` 拒绝覆盖已有 `ROAD_LOG_HEAD_DIR/rolling_dynamic_v1`。需要重跑时使用新的完整输入目录，保留历史结果。配置固定为本次已核查的 40 帧场景；不是任意场景的通用命令行工具。

可选的批量动态评估命令为 `dynamic_road_trial.py prepare` → `infer` → `evaluate` → `verify_dynamic_road.py` → `report_dynamic_road.py`。还需三个种子 17 / 29 / 43 的权重、概率和阈值，原开发 / 保留集 manifest、GRU 的 `gru_data_v2/split.json`、`gru_run_v1/best.pt`、`outputs/gru_val_v1/summary.json`。预测锁定后才读取未来标注。

`road_log_head_control.py` 同时保留增强头的历史训练入口，其 setup 额外依赖已锁定的 log 划分、旧特征和参考数组；回放只加载已训练权重，不调用这些训练入口。

## 发布记录

`results/` 是历史运行记录，服务器绝对路径已替换为环境变量占位符；里面的源脚本指纹仍对应实际运行时版本，不是重新运行公开副本得出的结果。公开代码仅做路径可配置化、持久缓存默认值与字体路径适配；算法和权重不变。[发布指纹清单](dynamic-source-manifest.json)同时保存原始和公开文件的 SHA256。不要将归档 `protocol.json` 直接当作新运行配置；新运行必须调用 `prepare` 生成自己的路径与指纹。

nuScenes 图像仅用于研究演示，数据归原数据提供方；使用者应遵守其数据许可。该演示不提供完整数据集或模型权重。
