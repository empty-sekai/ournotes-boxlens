# Our Notes BoxLens

仓库名：`ournotes-boxlens`。项目源码及训练权重采用 MIT，保留第三方许可声明；游戏资源不随项目分发，见 [资源边界](docs/ASSET_BOUNDARY.md)。输入输出见 [CONTRACT.md](docs/CONTRACT.md)。

从成员／留影列表截图识别 master ID，以及截图实际显示的等级、卡阶和成员特训次数。支持多张截图合并、重复上传去重、来源图预览和手工校对。推理使用 CPU；训练才需要 GPU。

## 仓库结构

- `bdon_vision/`：当前识别、合成数据与本地导入界面。
- `docs/`：使用说明、训练方法、资源边界、模型卡和评估说明。
- `scripts/`：资源准备、训练复现和评估工具。
- `tests/`、`schemas/`：数据完整性测试和输入输出契约。
- `reports/`：只保留训练覆盖、最终验收和性能指标；不含中间 smoke、截图或逐图调试记录，见 [报告索引](reports/README.md)。
- `reproduction/`：发布权重训练谱系所需的冻结源码和哈希清单；不是运行产物或历史实验堆栈。

**当前版本：v0.1.0，面向清晰列表截图。** 无法可靠读取的字段返回 `null`；但遮挡、渐隐和重压缩也可能造成误读，请先将卡片滚动到画面中央并重新截取清晰图片，再检查或手工校对结果。输出只表示截图观察到的卡片，不能据此认定未出现的卡片不在玩家 box 中。模型置信度是内部得分，不是经过校准的正确概率。

## 本地启动

在含 `boxvision.py` 的目录运行。需要 Python 3.10+ 和数据目录：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python boxvision.py serve --data DATA --port 18776
```

打开 `http://127.0.0.1:18776/`。选择成员／留影截图，执行识别，查看待校对项、来源原图和裁块，然后导出 JSON。网页只向本机服务发送图片，最多 30 张、请求体 48 MB、单图 2000 万像素；服务绑定本机回环地址。Linux 使用 `.venv/bin/python`。

命令行批量导入：

```powershell
python boxvision.py scan --data DATA --threads 2 --output ./result screenshot-a.jpg screenshot-b.jpg
```

输出 `observations.json`（每张截图的定位、来源指纹和字段）、`box.json`（合并结果）、带序号及来源指纹的标注图。同名输入文件不会覆盖彼此的标注图。成员与留影使用不同 ID 命名空间。

## 数据和模型

运行目录需要使用者自行准备 `catalog.json`、`index.npz`、卡面／canonical 图库、卡阶精灵，再将权重包中的 `encoder.onnx` 和 `fields.onnx` 放入该目录的 `models/`。公开附件只包含源码与权重，不包含游戏资源或开箱即用的图库。缺少多模式字段模型时拒绝启动，避免旧的纯等级模型把能力值当等级。`models/gallery.npz` 是可重建的缓存。

`scripts/prepare.py` 将显式提供的 master、UI、字体与卡面目录组装为数据目录；APK 导出与完整步骤见 [资源准备](docs/RESOURCES.md)。已有完整数据包的使用者无需重新提取。`boxvision.py prepare --data DATA` 从已准备的 master 表和卡面重建索引，`--offline` 可禁止补取素材；见 [新卡更新说明](docs/NEW-CARDS.md)。

## 校对和冲突语义

- 等级视图只读取等级；特训视图只读取特训次数；能力／百分比视图不转换为等级。
- 同一字段只有一个已观察到的值时合并该值；存在不同值时置为 `null` 并保留 `conflicts` 和全部来源，不取最大值。
- 重复截图不改变最终字段及来源证据；观察次数保留上传事实。
- 手工修正包含之前的值及修正记录。手工校对不能计入模型自动识别准确率。
- 游戏更新、UI 变更、未见过的布局或强降质图可能漏卡／漏字段。用户仍应检查总数及待校对项。

## 训练和验收

训练以合成数据为主，使用原始卡面、UI 精灵、prefab 文本位置和客户端字体。简／繁中、日、英、韩分别渲染，并覆盖不同屏幕比例、分辨率、UI 缩放、JPEG/WebP、多次压缩、模糊和滚动裁切。真实截图用于校准和回归，不能充当独立泛化准确率。

见 [训练与验证说明](docs/TRAINING.md)、[字体证据](docs/evidence/language-fonts.json)、[模型卡](docs/MODEL_CARD.md)、[数据卡](docs/DATA_CARD.md)、[覆盖边界](docs/COVERAGE.md) 和 [追加训练对照](docs/EXPERIMENTS.md)。当前版本已完成清晰全卡阶矩阵 2440/2440、修正后的 840 张独立整屏测试、真实回归和 2／4 逻辑 CPU 基准，详见 [评估卡](docs/EVALUATION.md)。压力场景仍会漏卡／拒识，不用接受值精度代替端到端覆盖率。

## 检查

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

涵盖跨图合并、同名不同来源、冲突保存、隐藏字段及裁切图标等数据完整性回归。

可用 `python -m pip install -e ".[test]"` 安装开发入口和 Schema 校验依赖，再执行 `python scripts/validate_contract.py box result/box.json`。安装后提供 `ournotes-boxlens` 命令，参数与 `python boxvision.py` 相同。
