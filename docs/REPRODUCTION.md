# 初始化权重的训练复现

v0.2.0 的成员／留影编码器从共享卡面编码器 `encoder.pt`（SHA-256 `8d67b87580188473d3408c4c1995f0e20b943d40ddcd2f8fc6a0f7a7fd26abda`）初始化，成员／留影参数模型从共享参数模型 `fields.pt`（SHA-256 `15351548a593c67cd939b72c3b03908e2665be5c39a0ca8148bd0cfc33feb0ce`）初始化。这两个权重随 v0.1.0 发布附件分发。本文件说明如何重建它们；v0.2.0 各模型自身的训练步骤见 [TRAINING.md](TRAINING.md)。

`scripts/reproduce_training.py` 将这两个权重的训练链整理为顺序执行的流程。它使用 `reproduction/stages/` 保存的阶段源码和 `reproduction/renderers/` 中的渲染器，而不是当前源码；`reproduction/metadata/` 记录检查点、资源和环境的哈希与版本。

该训练链所用的数据目录需按 v0.1.0 标签的源码和 RESOURCES.md 组装（`git checkout v0.1.0` 后运行 `boxvision.py prepare`），其中包含阶段源码需要的 `index.npz`。训练环境需要可用的 GPU PyTorch、对应的 torchvision，以及 numpy、Pillow、opencv、onnx；原训练的库版本见 `reproduction/metadata/environment.json`，其中的 git 构建版本不是普通 PyPI 安装命令。训练器把裁块整体放入显存，不能根据推理模型体积估计训练显存需求。

先只检查源码、master 快照和命令计划：

```bash
python scripts/reproduce_training.py --data data --output training-run --dry-run
```

完整训练（21 个顺序阶段）：

```bash
python scripts/reproduce_training.py --data data --output training-run
```

输出目录必须为空，避免覆盖旧实验。流程包含基础卡面／等级模型、多语言原生字段训练、两类整屏生成、整屏微调和困难字段训练。每个子阶段日志写入 `logs/`，`run.json` 记录退出码；失败会停止并保留已有产物。

已有公开初始化权重时，可跳过三个基础模型训练阶段，继续重建对应数据及后续训练：

```bash
python scripts/reproduce_training.py --data data --output training-run --initializers weights/initializers
```

初始化文件必须是 `encoder-v4.pt`、`level-v6.pt`、`fields-v7.pt`，并通过已记录 SHA-256 校验。`--baseline-only` 只运行到整屏微调。

master 字节与记录的快照不一致时默认停止。若有意用新卡或更新数据做新实验，可显式使用 `--allow-resource-change`；这种结果必须标记为新资源实验，不声称复现历史输入。运行中还会记录输入 provenance 与源码清单哈希。

训练完成不自动将最后一个检查点作为结果；开发集用于选择，最终测试单独执行。GPU／驱动／算子版本变化可能导致权重字节及指标变化，随机种子不构成跨环境逐字节确定性保证。
