# 合成训练与验收复现

完整记录的可执行入口和历史源码使用方式见 [REPRODUCTION.md](REPRODUCTION.md)。当前文件介绍各模块；复现具体历史权重时应使用固定阶段源码。

## 数据边界

训练主力为客户端资源合成；真实截图只是校准和回归。每张合成图记录哈希、屏幕原始分辨率、上传分辨率、缩放、模糊、压缩顺序、locale、显示模式和可见字段真值。潜在培养状态另存 `latent_state`，禁止用它当不可见字段的识别目标。

`scenarios.py` 的 train／validation／stress 使用不同的原始分辨率列表和压缩质量离散值，并有不同 UI 尺度、模糊和重复压缩范围。不是只换随机种子。四个相邻场景共享玩家状态，以测试重排、重叠和去重。成员和留影 ID 分命名空间。

APK 1.0.1-25 的语言选择证据：日／英主字体为 A-OTF-ShinGoPr6N-Regular，简／繁中为 FZLTH_GB18030L2_R，韩语为 Pretendard-SemiBold；数字角色均为 VibeMOPro-Medium。字体由语言配置驱动，不能仅由“区服”推断。使用真实 glyph、材质配置、字距和回退信息；SDF 栅格化仍是近似，不能声称与所有设备逐像素一致。

等级上限与卡阶／觉醒范围使用 masterdata。能力视图中的 power 数字目前是随机干扰值，**不是经过 native 算法校验的完整战力模拟**。真实力量值相关逻辑不参与当前导出目标，仍应保留该合成真实性限制。

## 生成与验证

在本目录作为 Python 模块入口运行：

```bash
python -m bdon_vision.synthetic generate --data DATA --output synthetic/train --count 2048 --profile train --seed 5201 --modes level,training,total,performance,technic,visual,hide --locales ja,en,zh-Hant,zh-Hans,ko
python -m bdon_vision.synthetic generate --data DATA --output synthetic/validation --count 560 --profile validation --seed 6203 --modes level,training,total,performance,technic,visual,hide --locales ja,en,zh-Hant,zh-Hans,ko
python -m bdon_vision.synthetic evaluate --data DATA --dataset synthetic/validation --output results/validation
python -m bdon_vision.evaluate_open_set --data DATA --dataset synthetic/validation --output results/open-set
python -m bdon_vision.audit_merge --dataset synthetic/validation --observations results/validation/observations.json --output results/validation/box-audit.json
```

数据生成最后才将 `complete` 置为 true。训练裁块构建检查全部截图哈希。评估要求身份及 bbox IoU≥0.5，记录漏卡、多报、字段正确／错误／未知、隐藏字段误报，以及按语言、卡片尺寸、压缩和分辨率分层。`audit_merge` 额外检查 box 合并后的字段、凭空出现的值、虚假冲突和真实冲突丢失。

## 场景微调

GPU 环境使用已验证的 PyTorch／ONNX 依赖；不要把 GPU 包加入普通 CPU 使用者的运行依赖。实际训练脚本及模型的 JSON 报告保存参数、步数、样本量和数据指纹。

```bash
python -m bdon_vision.corpus --data DATA --datasets synthetic/train --output corpora/scenes-train
python -m bdon_vision.corpus --data DATA --datasets synthetic/validation --output corpora/scenes-validation
python -m bdon_vision.train_scene_encoder --data DATA --train corpora/scenes-train --validation corpora/scenes-validation --initial models/previous/encoder.pt --output models/candidate-encoder --steps 2400 --batch 128
python -m bdon_vision.train_fields --train corpora/scenes-train --validation corpora/scenes-validation --initial models/previous/fields.pt --replay corpora/native-fields-train --output models/candidate-fields --steps 2400 --batch 512
```

字段共有 106 类：0 表示其他／隐藏参数；1–100 表示等级；101–105 表示可见特训次数。图库模型训练排除指定身份的两个分支，验证时仅添加图像向量，不更新权重。

卡阶 1–5 均包含在整屏合成中，实际可见训练计数分别为 7208／7089／7371／7357／7372。卡阶推理使用原生精灵与局部背景的模板匹配，没有单独训练卡阶分类网络。它与等级／特训字段神经网络是不同模块，不能把卡阶模板验证描述成神经网络训练准确率。

`scripts/verify_rank_matrix.py --data DATA --output rank-matrix` 遍历当前图库每个身份、masterdata 中每个合法已拥有卡阶及四种分辨率。为隔离卡阶变量，该矩阵固定等级为 1；等级泛化另看混合等级整屏数据和字段训练分布。现行整屏拼接保留原生透明边距，避免把卡框外伸出的属性／卡阶图标截掉；历史复现使用冻结的旧生成器，不能用新图冒充原始训练输入。

候选 ONNX 先放入单独数据目录验收，不覆盖已使用的版本。保留真实回归、完整合成验证、压力集、未知卡、空背景、跨图冲突、图库更新集成和有限 CPU 性能证据。裁块分类准确率不等于端到端整屏准确率，压力集错读也不能用普通集高分掩盖。
