# 可执行训练复现入口

`scripts/reproduce_training.py` 将实际训练链整理为顺序执行的流程。它使用 `reproduction/stages/` 保存的阶段源码，而不是把后来修改过的生成器冒充原始训练版本。这里仅保留重建已发布 checkpoint 所需的训练源码、渲染器和哈希清单；训练日志、smoke 产物、旧报告和截图不放在发布仓库。

先按 RESOURCES.md 准备 `data`。训练环境需要可用的 GPU PyTorch、对应的 torchvision，以及 numpy、Pillow、opencv、onnx；本次实际环境是 ModelScope 的定制 HIP 镜像，版本见 `reproduction/metadata/environment.json`。不要把其中的 git 构建版本误当作普通 PyPI 安装命令。当前训练器把裁块整体放入显存，预训练数据规模较大，不能根据推理模型体积估计训练显存需求。

先只检查源码、master 快照和命令计划：

```bash
python scripts/reproduce_training.py --data data --output training-run --dry-run
```

完整训练（21 个顺序阶段）：

```bash
python scripts/reproduce_training.py --data data --output training-run
```

输出目录必须为空，避免覆盖旧实验。流程包含基础卡面／等级模型、多语言原生字段训练、两类整屏生成、整屏微调、相同数据追加训练对照，以及修正截图压缩顺序后的困难字段训练。每个子阶段日志写入 `logs/`，`run.json` 记录退出码与耗时；失败会停止并保留已有产物。

已有公开初始化权重时，可跳过三个基础模型训练阶段，继续重建对应数据及后续训练：

```bash
python scripts/reproduce_training.py --data data --output training-run --initializers weights/initializers
```

初始化文件必须是 `encoder-v4.pt`、`level-v6.pt`、`fields-v7.pt`，并通过已记录 SHA-256 校验。`--baseline-only` 只运行原 v8 模型训练链，不执行后续对照和困难数据实验。

master 字节与历史快照不一致时默认停止。若有意用新卡或更新数据做新实验，可显式使用 `--allow-resource-change`；这种结果必须标记为新资源实验，不声称复现历史输入。该检查不能替代完整的数据集和渲染验证，运行中还会记录输入 provenance 与源码清单哈希。

## checkpoint 选择和最终测试

训练完成不自动将最后一个 checkpoint 作为发布模型。普通开发集与开发压力集用于模型选择，最终测试单独生成和执行。当前候选选择是追加训练编码器第 2400 步，以及困难字段模型第 5500 步，选择依据单独保存。

各阶段已分别实际执行完整训练；新包装入口完成了计划检查和小规模集成验证，但尚未通过此入口从零再次执行完整规模的训练。集成验证不产出可部署权重。GPU／驱动／算子版本变化也可能导致权重字节及指标变化，随机种子不构成跨环境逐字节确定性保证。
