# 本次集成验证

验证日期：2026-10-19。Python 3.12.15 / PyTorch 2.11.0，使用 CPU。

- `tests/policies/pi05_rlt`：35 项通过，覆盖因果重建、停止梯度、掩码、Actor/Critic 损失、
  逐步折扣、目标网络、人工动作 BC、延迟更新、UTD、边界回放、恢复、策略注册、处理器和推理队列。
- 加上现有 `test_factory_dataset_feature_hook.py`、`test_factory_peft_revision.py`、
  `pi0_pi05/test_pi0_embed_suffix.py`：总计 **60 passed, 1 skipped**。
- Ruff 检查与格式检查通过，`bash -n examples/pi05_rlt/run.sh` 通过。
- `run.sh precompute --help` 正常，确认当前环境可导入真实 π0.5 / 数据集模块。
- `run.sh smoke --output_dir /tmp/agilex-rlt-smoke-final` 通过：Stage 1 更新/恢复 →
  缓存编码 → 关键阶段分类器 → Stage 2 更新/恢复 → 导出/重新加载和参数一致性检查。

验证使用合成缓存和小模型，骨干推理单元测试使用 API 替身；没有加载真实 π0.5 权重，
没有进行完整训练或真机 rollout。未验证 GPU 显存需求、真实任务收敛、收益和 ROS 接管时序。
测试工具安装到 `/tmp/agilex-rlt-test-deps`，没有更改现有 Conda 环境的依赖版本。
