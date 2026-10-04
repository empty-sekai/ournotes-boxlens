# 合成训练与验收复现

本文件介绍 v0.2.0 模型的数据生成、训练、选择和评估入口。各模型的实际参数与选用检查点见 `reports/training/model-selection.json`，初始化权重的训练链见 [REPRODUCTION.md](REPRODUCTION.md)。

## 数据边界

训练主力为客户端资源合成；真实截图只是回归。每张合成图记录哈希、屏幕原始分辨率、上传分辨率、缩放、模糊、压缩顺序、locale、显示模式和可见字段真值。潜在培养状态另存 `latent_state`，禁止用它当不可见字段的识别目标。

`scenarios.py` 的 train／validation／stress／final 四档使用不同的原始分辨率列表和压缩质量离散值，并有不同 UI 尺度、模糊和重复压缩范围，不是只换随机种子。成员和留影 ID 分命名空间。

APK 1.0.1-25 的语言选择证据：日／英主字体为 A-OTF-ShinGoPr6N-Regular，简／繁中为 FZLTH_GB18030L2_R，韩语为 Pretendard-SemiBold；数字角色均为 VibeMOPro-Medium。字体由语言配置驱动，不能仅由“区服”推断。使用真实 glyph、材质配置、字距和回退信息；SDF 栅格化仍是近似，不能声称与所有设备逐像素一致。

等级上限与卡阶／觉醒范围使用 masterdata。能力视图中的 power 数字是随机干扰值，**不是经过 native 算法校验的完整战力模拟**，不参与导出目标。

## 列表整屏场景

`bdon_vision.scenes` 按列表组件的原生几何放置原生卡块：1920×1080 参考画布按 `min(W/1920, H/1080)` 缩放，成员每行 6 张、留影每行 4 张并水平居中，顶部留白 180 个参考单位，再加滚动偏移。画面另含带雾感的模糊背景、顶栏标题与排序／视图按钮、少量界面浮层、图库外卡面和无卡负样本。可选的场景背景放在 `DATA/backgrounds/adv-stage/*.webp`，由使用者自行准备；缺失时只用卡面模糊、噪声和渐变背景。

```bash
python -m bdon_vision.scenes generate --data DATA --output scenes/v2.1-train --count 16000 --seed 20261111 --profile train --workers 48
python -m bdon_vision.scenes generate --data DATA --output scenes/v2.1-validation --count 2000 --seed 20261112 --profile validation --workers 48
python -m bdon_vision.scenes generate --data DATA --output scenes/v2.1-stress --count 1000 --seed 20261113 --profile stress --workers 48
python -m bdon_vision.scenes generate --data DATA --output scenes/v2.1-final --count 2000 --seed 20261114 --profile final --workers 48
python -m bdon_vision.scenes verify scenes/v2.1-train
python -m bdon_vision.scenes stats scenes/v2.1-train --output scenes/v2.1-train-stats.json
```

真值格式为 `bdon-synthetic/2`，新增字段列在 `truth.json` 的 `schema_extension` 中。`visible_fraction` 同时扣除屏外部分和界面遮挡；`foreign_cards` 记录图库外卡面的卡框和字段，不含 `id`。train 档排除 `id % 7 == 0` 的身份，图库外卡面来源和卡面背景也排除；编号是 5 的倍数的场景背景只用于非训练档。`final` 档的分辨率和压缩质量与其他档都不同，只在模型冻结后使用。图片可存为无损 WebP，像素与施加采集损伤后的结果一致。每张图由生成器版本、profile、seed 和序号单独决定，与进程数和分块方式无关。

当前源码的生成器标识为 `scenes-v2.1`。训练和测试还使用了一组上一生成器版本 `scenes-v2` 的数据（种子 20261101–20261104），它们只能以真值哈希核对，不能由当前源码逐字节重建。

## 卡框定位器

```bash
python -m bdon_vision.train_locator --train scenes/v2-train "synthetic/train-*" --validation scenes/v2-validation --validation-limit 600 --backbone mobilenet_v3_large-5c1a4163.pth --output runs/locator-a --steps 15000 --lr 0.002 --warmup 500 --seed 1
python -m bdon_vision.train_locator --train scenes/v2.1-train scenes/v2-train "synthetic/train-*" --validation scenes/v2.1-validation --validation-limit 600 --initial runs/locator-a/best.pt --output runs/locator-b --steps 4000 --lr 0.0006 --warmup 100 --eval-every 500 --seed 3
python -m bdon_vision.export_locator --weights runs/locator-b/best.pt --output models/locator.onnx
python -m bdon_vision.evaluate_locator --model models/locator.onnx --set v2.1-validation=scenes/v2.1-validation --output locator-validation.json
```

`synthetic/train-*` 是 `bdon_vision.synthetic generate --profile train` 生成的 16 组、每组 40 张单类卡片截图（种子 300000–315000）。训练按 512 像素随机裁块，以中心热图 focal loss、尺寸 L1 和 GIoU 为目标，按开发验证 AP50 选用检查点。

## 卡面编码器

```bash
python -m bdon_vision.kind_corpus --data DATA --datasets scenes/v2.1-train --output banks/encoder/train --arrays context --stride 2
python -m bdon_vision.kind_corpus --data DATA --datasets scenes/v2.1-validation --output banks/encoder/validation --arrays art baseline
python -m bdon_vision.kind_corpus --data DATA --datasets scenes/v2.1-stress --output banks/encoder/stress --arrays art baseline
python -m bdon_vision.train_kind_encoder --kind member --data DATA --train banks/encoder/train --validation banks/encoder/validation --stress banks/encoder/stress --initial encoder.pt --output runs/member-encoder --steps 3000 --lr 0.00012 --head-lr 0.00012 --margin 0 --open-weight 0 --same-name-margin 0 --kd-weight 1 --ref-mode frozen --cross-kind
python -m bdon_vision.export_kind_encoder --kind member --data DATA --model runs/member-encoder/step-001750.pt --output models --thresholds runs/member-encoder/selection.json
```

留影把 `--kind` 换成 `snap`。训练查询来自整屏卡框裁块，参考来自图库卡面；`id % 7 == 0` 的身份在两个分支中都被排除。每 250 步在开发验证集上搜索接受阈值（训练外身份留一误接受与图库外卡面误接受均不超过 1%，接受精度不低于 99.5%），取已知与训练外身份平均接受正确率最高的检查点；并列时比较压力集。导出时检查 PyTorch 与 ONNX 输出差异、批量与单张一致性和向量范数。

## 参数与卡阶模型

```bash
python -m bdon_vision.kind_fields --datasets scenes/v2.1-train --output banks/fields/v2.1-train
python -m bdon_vision.kind_fields --datasets scenes/v2.1-validation --output banks/fields/v2.1-validation
python -m bdon_vision.kind_fields --datasets scenes/v2.1-stress --output banks/fields/v2.1-stress
python -m bdon_vision.kind_hardcrops --data DATA --kind member --output banks/hard-g-member --count 60000 --seed 190001 --split train --occlusion-rate .08
python -m bdon_vision.kind_hardcrops --data DATA --kind member --output banks/hard-n-member --count 60000 --seed 180001 --split train --occlusion-rate 0
python -m bdon_vision.kind_hardcrops --data DATA --kind member --output banks/hard-l-member --count 16000 --seed 170001 --split validation
python -m bdon_vision.train_kind_fields --kind member --train banks/fields/v2.1-train --hard banks/hard-g-member --dev validation:normal:banks/fields/v2.1-validation stress:stress:banks/fields/v2.1-stress hard:hard:banks/hard-l-member --initial fields.pt --output runs/member-fields --steps 8000 --lr 0.0003
python -m bdon_vision.train_kind_fields --kind member --train banks/fields/v2.1-train --hard banks/hard-g-member --dev validation:normal:banks/fields/v2.1-validation stress:stress:banks/fields/v2.1-stress hard:hard:banks/hard-l-member --initial runs/member-fields/selected.pt --output runs/member-fields-2 --steps 3000 --eval-every 250 --lr 0.0001 --jitter-shift 4 7 --jitter-scale .035
python -m bdon_vision.train_kind_rank --kind member --train banks/fields/v2.1-train --hard banks/hard-n-member --dev validation:normal:banks/fields/v2.1-validation stress:stress:banks/fields/v2.1-stress hard:hard:banks/hard-l-member --output runs/member-rank --steps 6000
```

`kind_hardcrops` 每个样本渲染一张原生卡块（卡面、卡框、卡阶图标）和一层本地化参数文字，贴到类似列表的背景上，再对整张卡块模拟屏幕采集：缩小到屏幕尺度、模糊、随机编码块相位、1–4 次 JPEG 或 WebP 压缩，最后按部署时的裁切几何取参数区和卡阶图标。`--occlusion-rate` 控制带界面浮层的比例；文字或图标像素被遮住至少 2% 的样本记为类别 0。留影把 `--kind` 换成 `snap`、种子加 2；成员第二条命令是成员参数的第 2 阶段。各模型使用的裁块库见报告。

参数模型分为成员 106 类、留影 101 类，从共享参数模型初始化；卡阶模型 6 类，随机初始化。训练时对裁切位置和尺度做随机抖动，使模型适应定位误差。每个检查点都在开发集上求出“所有开发裁块零误读”的最低置信度阈值（不低于 0.995，差值固定 0.5），再取该阈值下普通与压力整屏召回最高的检查点；选定阈值就是部署阈值。误读包括可见值读错，以及对不可见目标输出了与卡片状态不同的值。`select_kind_models` 可对已有检查点目录重新执行同一选择。

## 端到端评估

```bash
python -m bdon_vision.evaluate_engine --data DATA --set v2.1-final=scenes/v2.1-final --set v2-final=scenes/v2-final --output engine-final.json
python -m bdon_vision.audit_merge --dataset scenes/v2.1-validation --observations observations.json --output box-audit.json
```

`evaluate_engine` 的计分规则见 [EVALUATION.md](EVALUATION.md)。`evaluate_kind_fields` 与 `evaluate_kind_encoder` 在开发集上分别评估参数／卡阶和编码器，只用于开发阶段。`audit_merge` 额外检查 box 合并后的字段、凭空出现的值、虚假冲突和真实冲突丢失。

候选 ONNX 先放入单独数据目录验收，不覆盖已使用的版本。裁块分类准确率不等于端到端整屏准确率，压力集错读也不能用普通集高分掩盖。

GPU 环境使用可用的 PyTorch／torchvision 与 ONNX；不要把 GPU 包加入普通 CPU 使用者的运行依赖。
