200 场景融合道路基线：road200-v1

服务器目录
/path/to/workspace/driving_pilots/road_refinement/shared_bev/train200/baseline_v1

固定设置
训练数据为 200 个场景、2400 帧。六路相机与当前帧 LiDAR 经冻结的
BEVFusion 提取融合特征，再训练原 RoadHead(256)，采用原 BCE 损失。
根据开发集 IoU 选择 seed 17、第 1700 步权重，阈值固定为 0.60。
开发集 IoU 为 85.1106%。保留场景与验证集不参与本次模型选择。
没有接入此前的人行道辅助头或残差修正头。

本固定模型的实验指标（单个模型，与三种子平均值区分）
原验证子集：150 场景、542 帧，道路 IoU 77.4530%，人行道误放行 12543 格。
保留评估集：27 场景、108 帧，道路 IoU 73.0058%，人行道误放行 1382 格。
这里的 542 帧是验证子集，不是完整官方验证集的所有帧。

文件
road_head.pt：固定道路头权重。
config.json：阈值、网格定义、数据与代码及预训练模型的 SHA256。
verification.json：回放验证结果。
validation150_outputs.npz、reserved27_outputs.npz：服务器上的逐帧输出。
example_scene0218.npz：单帧命令行推理示例输出。

输出解释
probability：道路概率。
road：概率 >= 0.60 的道路预测。
ground：已有地面回波证据。
blocked：现有 LiDAR 高点或预测车辆框形成的占用区域。
passable = road & ground & ~blocked。
unknown_road = road & ~ground & ~blocked，表示道路预测但地面证据不足。
最终输出为 96×96 栅格，每格 0.5 米；列对应 ego x，行对应 ego y。

服务器运行方式
cd /path/to/workspace/driving_pilots

重放全部 650 帧（使用与原实验相同的 GPU 类型核对逐格结果）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../BEVFusion/runtime/bin/python road_200_baseline.py verify --reserved-device cuda:1

单帧推理
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../BEVFusion/runtime/bin/python road_200_baseline.py predict --feature road_refinement/shared_bev/train200/eval27_features/f159bc15ff384dc4a7db7acbe6bee52c.npy --reference road_refinement/shared_bev/passable_walkway/frozen/frames/f159bc15ff384dc4a7db7acbe6bee52c.npz --output road_refinement/shared_bev/train200/baseline_v1/example_scene0218.npz

推理入口接收已缓存的融合特征和对齐的 ground/blocked 参考栅格，运行时不读取
道路真值或人行道真值。六路图像及点云的特征提取仍由 road_200_data.py 完成。
特征可以是 256×64×64 融合张量，也兼容旧缓存 592×64×64 的融合通道切片。
权重、模型结构与阈值已固定，后续实验应写到新版本目录。
