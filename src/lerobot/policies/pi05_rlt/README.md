# π0.5 RL-Token

当前实验保留已知论文差异，未进行算法修正。决定与追踪见
[当前基线](../../../../examples/pi05_rlt/BASELINE.md) 和
[论文核对](../../../../examples/pi05_rlt/PAPER_AUDIT.md)。

来源：[Zhengsw03/RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open/tree/cccf949c782266f3dab227ec4265a6360e0511e2)。
来源提交和许可记录于 [NOTICE](NOTICE)。

| 文件 | 内容 |
| --- | --- |
| `configuration_pi05_rlt.py` | 注册 `pi05_rlt` 配置，双 Piper 默认 14 维状态/动作 |
| `modeling_pi05_rlt.py` | RL token 编码器、因果重建解码器、Gaussian Actor、Twin Critic、策略推理 |
| `vla_compat.py` | 使用本地 π0.5 的 prefix/suffix API 提取 Transformer 输出 |
| `processor_pi05_rlt.py` | Stage 1 缓存处理与 π0.5 预处理/后处理委托 |
| `learning.py` | Stage 2 折扣目标、BC/Q 损失、延迟更新、Polyak 目标及恢复 |
| `replay.py` | 动作来源、真实执行窗口、重叠回放、有效前缀、分层采样及 NPZ 存储 |
| `training.py` | mmap 缓存校验和 Stage 1 检查点接口 |
| `critical.py` | 可选关键阶段分类器及带滞回/最短持续时间的检测器 |

与参考实现保持一致的关键点：后 Transformer 图像位置、追加 RL token、停止梯度的因果重建、
非残差 Actor、固定 Gaussian std、target Actor/Twin Critic、逐步折扣、人工动作 BC 与来源标记。
适配主要涉及新版 LeRobot 的配置/处理器/分块接口、14 维双臂接口、设备迁移、缓存与检查点验证。
修复了关键阶段检测器的持续计时被每次高概率刷新、episode reset 未清除计时的问题。

训练入口位于 `lerobot.scripts.rlt`。完整命令与 ROS 接入边界见
[双 Piper 使用说明](../../../../examples/pi05_rlt/README.md)。
算法模块不打开 CAN，也不使用 LeRobot 机器人封装；真实在线收集/人工接管由外部控制层负责。
