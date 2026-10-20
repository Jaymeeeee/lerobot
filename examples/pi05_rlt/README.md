# AgileX 双 Piper 的 π0.5 RL-Token

当前按 2026-10-20 的实验决定保留已有算法，先训练观察效果，已知论文差异暂不修正。
差异清单及后续逐项比较方式见 [BASELINE.md](BASELINE.md)，详细核对见 [PAPER_AUDIT.md](PAPER_AUDIT.md)。

本目录对应 LeRobot 分支 `test/pi05_rlt`，算法代码位于
[`src/lerobot/policies/pi05_rlt`](../../src/lerobot/policies/pi05_rlt)。实现来源为
[RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open/tree/cccf949c782266f3dab227ec4265a6360e0511e2)，
按本地新版 LeRobot 接口适配，默认状态/动作改为双 Piper 的 14 维。

已经实现：冻结 π0.5 的特征缓存、Stage 1 编码器/因果解码器训练、带人工干预来源标记的回放、
Stage 2 Actor/Twin-Critic 更新、可选关键阶段分类器、训练恢复及 LeRobot 格式导出/推理。
训练代码不依赖 ROS、CAN 或 LeRobot 的机器人硬件接口。

**本次范围是算法库。** 四 CAN 模式的 ROS 在线采样、模型驱动时主臂跟随、人工接管仲裁、四臂复位，
仍需后续在 AgileX_ws 的控制层接入。本目录的 Stage 2 启动脚本读取已有回放训练，不会启动机器人，
也不会自行产生真实环境奖励。`set_actor_enabled`/`set_critical_manual` 是 VLA 与 RL Actor 之间的门控，
不能代替人工接管时的硬件指令仲裁。

## 环境和自检

从 `~/AgileX_ws/third_party/lerobot` 执行命令。启动脚本默认使用现有 `lerobot` Conda 环境的 Python，
通过 `uv run` 执行，并隔离 ROS 的 Python 路径。其他机器可用 `RLT_PYTHON=/path/to/python` 覆盖。
脚本默认 `HF_HUB_OFFLINE=1`，真实训练需先准备本地 π0.5 权重和 tokenizer。

```bash
cd ~/AgileX_ws/third_party/lerobot
git branch --show-current  # test/pi05_rlt
bash examples/pi05_rlt/run.sh smoke --output_dir /tmp/rlt-smoke-001
```

输出目录必须不存在。自检使用 8 维隐藏层的合成数据，执行真实反向传播，依次检查 Stage 1、恢复训练、
特征编码、分类器、Stage 2、恢复、导出和重新加载。它验证实现与接口，不能验证真机效果或算法收益。
运行单元测试需安装项目的测试依赖：`uv run pytest tests/policies/pi05_rlt -q`。

还提供 `lerobot-rlt-precompute`、`lerobot-rlt-stage1`、`lerobot-rlt-stage2`、`lerobot-rlt-encode`、
`lerobot-rlt-critical`、`lerobot-rlt-export` 入口；需重新执行 editable 安装生成这些命令。
直接使用下面的 `run.sh` 无需重新安装。

## 1. 先准备适合双臂的 π0.5

需要一个已经完成双 Piper 任务 SFT 的本地 π0.5 `pretrained_model` 目录，包含 `config.json`、
`model.safetensors`、预处理/后处理配置及归一化统计量。当前 ACT 权重不能作为 π0.5 骨干。
代码通过现有 `PI05Policy` 加载骨干，不下载或重新实现 π0.5。

数据使用 AgileX_ws 已有转换器输出的 LeRobot 数据集。状态与动作的顺序均为：

```text
左 J1..J6, 左夹爪宽度, 右 J1..J6, 右夹爪宽度
```

这与 `src/data_collection/config/lerobot_conversion.yaml` 的 14 维顺序一致。关节为 rad、夹爪宽度为 m；
RL 内部使用骨干预处理器的归一化空间，实际执行前必须经过对应的后处理器。
骨干应使用 `observation.images.cam_h/cam_l/cam_r` 等与实际数据完全一致的相机特征名和顺序。
缓存阶段会继承骨干的图像特征、归一化配置和 tokenizer，并检查特征维度。
本版本要求绝对动作、单时刻观测；相对动作和视觉/本体记忆扩展会明确拒绝。

## 2. 缓存冻结骨干的图像 token

以下路径需替换成实际数据与 π0.5 权重目录：

```bash
bash examples/pi05_rlt/run.sh precompute \
  --pi05_path /path/to/pi05/pretrained_model \
  --tokenizer_path /path/to/local/paligemma-tokenizer \
  --dataset_root /path/to/converted_lerobot_dataset \
  --dataset_repo_id local/insert_plug \
  --output_dir /path/to/rlt/prefix_cache --device cuda --batch_size 1
```

若权重附带的处理器已指向可用的本地 tokenizer，可省略 `--tokenizer_path`。
可用 `--max_frames 32` 试跑，或 `--episodes 0 1 2` 分片；每个进程必须使用独立输出目录。
输出 `prefix_out.mmap` 为 float16，另有掩码、归一化状态/动作、episode/frame 索引及 `meta.json`。
缓存保存的是**经过骨干 Transformer 后**的图像位置，随后才丢弃语言/离散状态位置，
因此这些图像特征已经包含上下文信息。缓存完成前不会标记 `complete=true`；训练会拒绝不完整缓存。
指纹使用配置内容及权重文件大小/修改时间，避免误混缓存；它不是权重内容的加密摘要。

## 3. Stage 1：训练 RL token

```bash
bash examples/pi05_rlt/run.sh train_stage1 \
  --config examples/pi05_rlt/agilex_pi05_rlt.json \
  --precomputed_path /path/to/rlt/prefix_cache \
  --output_dir /path/to/rlt/stage1 \
  --steps 30000 --batch_size 16 --use_amp --device cuda
```

先用 `--dry_run` 做一次真实优化更新；输出目录含 `last_checkpoint.pt`、`best_checkpoint.pt` 和指标日志。
多片缓存用逗号连接。恢复时保持原参数/总步数不变并增加
`--resume_from /path/to/rlt/stage1/last_checkpoint.pt`；`--steps` 表示 Stage 1 总步数。
恢复包含模型、优化器和调度器，但不保存 DataLoader 的采样位置，因此不保证与未中断训练逐步一致。
`best_checkpoint` 按保存时的训练批次损失选取，不是验证集最优模型。

默认网络与参考项目一致：2048 维 token、4 层编码器、4 层因果解码器。
这部分训练有显著显存/存储开销，合成自检中的小模型配置只能用于检查流程。
Stage 1 冻结骨干并使用缓存，因此训练时不把整个 π0.5 放入显存。

## 4. Stage 2：回放及 Actor/Critic 更新

Stage 2 必须使用带真实边界、执行动作和奖励的回放，不能把示教视频直接当成在线 RL 数据。
控制层需要在重规划边界记录 `(z_rl, normalized_state, VLA_reference)`，每步记录最终下发的动作，
而非直接记录尚未执行的模型输出。来源标记为 `BASE=0`、`RL=1`、`HUMAN=2`、`MIXED=3`。
末尾未执行完的 chunk 必须用 `valid_mask` 标记有效前缀；终止奖励写在最后一个有效执行步。
奖励的定义/成功判定由任务层提供；默认 TD 目标裁剪到 ±2。

库内的接入接口：

```python
from lerobot.policies.pi05_rlt.learning import RLTLearner
from lerobot.policies.pi05_rlt.replay import (
    BoundaryContext,
    ExecutionWindow,
    ReplayBuffer,
    TransitionSource,
    build_transition,
)
from lerobot.policies.pi05_rlt.training import load_stage1

policy, _ = load_stage1("/path/to/stage1/last_checkpoint.pt", device="cuda")
learner = RLTLearner(policy)
replay = ReplayBuffer(capacity=100_000, seed=42)

# 以下值由 ROS 控制层在真实边界采样，均为归一化张量；此处展示接口。
boundary = BoundaryContext(z_rl, normalized_state, vla_reference, "online", episode_id, step_id)
window = ExecutionWindow(boundary)
window.append(actual_normalized_action, TransitionSource.HUMAN, intervention=True)
# ...逐步 append，最多 C 步；在下一真实观测边界构造 next_boundary。
transition = build_transition(
    window, next_boundary, horizon=10, action_dim=14, terminal=episode_done, reward=terminal_reward
)
replay.add(transition)
metrics = learner.learn(replay, batch_size=256, new_transitions=1)
replay.save("/path/to/replay.npz")
learner.save("/path/to/stage2.pt")
```

该片段是控制层的接口示意，`z_rl`、动作、奖励等需要调用方提供。库也提供
`StepRecord` / `DeferredBoundarySnapshot` / `build_stride2_transitions`，用于像参考项目一样构造
每隔 2 步开始的重叠回放；快照字典按连续片段内的相对步索引提供，必须包括所需的起止边界。
**执行和 Actor 输入始终取 VLA 前 C 个连续动作**，stride=2 仅作用于回放窗口起点。
人工接管时需立即清除控制层剩余指令及 `policy.drop_queued_actions()`；不能等当前 chunk 执行完。
若将学习和推理放到不同线程，调用方必须使用 `learner.lock` 同步策略权重读取，或给推理线程独立权重快照。

已有回放后，可脱离机器人训练/调试：

```bash
bash examples/pi05_rlt/run.sh train_stage2 \
  --stage1_checkpoint /path/to/rlt/stage1/last_checkpoint.pt \
  --replay_path /path/to/replay.npz --output_dir /path/to/rlt/stage2 \
  --steps 1000 --batch_size 256 --device cuda
```

恢复用 `--resume_from /path/to/rlt/stage2/last_checkpoint.pt` 替代 `--stage1_checkpoint`。
这里 `--steps` 是本次追加的更新次数。回放和学习器检查点分别保存；在线集成时应在同一个暂停边界保存两者。
默认每 2 次 Critic 更新一次 Actor，UTD=5（在线 `learner.learn` 接口），固定探索 std=0.05、
BC 权重=10、参考动作 dropout=0.5、Polyak tau=0.005。
人类/混合动作对实际执行动作做 BC，其余来源对冻结 VLA 参考做 BC；
TD 目标采用逐步折扣奖励和 `gamma ** executed_steps`，不会将整段错误折扣为一步。

## 5. 可选关键阶段分类器

为缓存中的每一帧提供 0/1 关键阶段标签 `.npy` 文件，再执行：

```bash
bash examples/pi05_rlt/run.sh encode_cache \
  --stage1_checkpoint /path/to/rlt/stage1/last_checkpoint.pt \
  --cache /path/to/rlt/prefix_cache --labels /path/to/critical_labels.npy \
  --output /path/to/rlt/features.npz --device cuda
bash examples/pi05_rlt/run.sh train_critical \
  --features /path/to/rlt/features.npz --output /path/to/rlt/critical.pt
```

标签顺序与缓存帧顺序一致；分类器按 episode 划分训练/验证，避免同一轨迹泄漏。
`encode_cache` 不生成 VLA 参考 chunk 或环境奖励，其输出不能直接替代 Stage 2 replay。
推理时设置 `deploy_auto_critical=true` 和 `critical_classifier_path` 启用分类器；
仅手动门控可设置 `deploy_gating=true`，然后调用 `set_actor_enabled` 和 `set_critical_manual`。
默认不启用关键阶段门控，加载训练后的 Actor 即执行 Actor 输出。自动门控接口面向单机器人单批次推理。

## 6. 导出与推理

```bash
bash examples/pi05_rlt/run.sh export_policy \
  --checkpoint /path/to/rlt/stage2/last_checkpoint.pt \
  --output_dir /path/to/rlt/pretrained_model \
  --pi05_path /path/to/pi05/pretrained_model
```

输出标准 `config.json`、`model.safetensors` 和处理器文件；骨干权重继续保留在 `pi05_pretrained_path`，
不会复制到 RLT 权重文件。Stage 1 的 Actor 尚未训练，导出时必须指定 `--vla_only`。
`--without_processors` 仅供测试或调用方自行保证预处理契约的场景使用。

```python
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy

path = "/path/to/rlt/pretrained_model"
policy = PI05RLTPolicy.from_pretrained(path, local_files_only=True).to("cuda").eval()
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": "cuda"}}
)
policy.attach_frozen_vla_from_config()
batch = pre(observation)  # 三路相机、14 维状态、task；需由当前 ROS 采样层提供
normalized_chunk = policy.predict_action_chunk(batch)  # [1, C, 14]
command_chunk = post(normalized_chunk)  # 物理空间；仍由 ROS 控制层校验并下发
```

新 episode 调用 `policy.reset()` 清除队列；策略支持 `select_action` 逐步执行和 `predict_action_chunk`
整段预测。真机控制层必须负责归一化/物理空间转换、关节顺序、限位、超时、人工接管和复位。
本实现的训练检查点使用明确格式及 `weights_only=True`；未承诺直接读取参考项目所有历史脚本的 `.pt` 格式。
