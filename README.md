# Our Notes BoxLens

仓库名：`ournotes-boxlens`。项目源码及训练权重采用 MIT，保留第三方许可声明；游戏资源不随项目分发，见 [资源边界](docs/ASSET_BOUNDARY.md)。输入输出见 [CONTRACT.md](docs/CONTRACT.md)。

从成员／留影列表截图识别 master ID，以及截图实际显示的等级、卡阶和成员特训次数。支持多张截图合并、重复上传去重、来源图预览和手工校对。推理只用 CPU 和 ONNX Runtime；训练才需要 GPU。

## 识别流程

1. `locator.onnx` 在整张截图上找出成员和留影卡框，包括被屏幕边缘切掉一部分的卡。
2. 每类卡各有一个卡面编码器（`member-encoder.onnx`、`snap-encoder.onnx`），把卡框内的卡面窗口编码成 128 维向量，再在本地图库里检索最相近的卡。相似度和与第二名的差距都达到阈值才接受这个 ID；否则该卡框进入 `unidentified`，附最相近的候选供人工确认。
3. 每类卡各有一个参数读取模型（`member-fields.onnx`、`snap-fields.onnx`）和一个卡阶图标模型（`member-rank.onnx`、`snap-rank.onnx`）。读取区域必须完整位于截图内，置信度达到阈值才输出值；等级还要符合 masterdata 的等级上限。

图库由使用者本地的卡面图编码而成，不属于模型权重。新卡只需更新 master 表和卡面，再重建图库，模型不用重训，见 [新卡接入](docs/NEW-CARDS.md)。

**当前版本：v0.2.0，面向列表截图。** 无法可靠读取的字段返回 `null`，无法确认身份的卡框列在 `unidentified` 里，二者都需要校对；遮挡、渐隐和重压缩仍可能造成误读，请把卡片滚动到画面中央、重新截取清晰图片后再检查。输出只表示截图观察到的卡片，不能据此认定未出现的卡片不在玩家 box 中。模型置信度和相似度是内部得分，不是经过校准的正确概率。

## 仓库结构

- `bdon_vision/`：识别、合成数据、训练与评估模块和本地导入界面。
- `docs/`：使用说明、训练方法、资源边界、模型卡和评估说明。
- `scripts/`：资源准备、训练复现和验证工具。
- `tests/`、`schemas/`：数据完整性测试和输入输出契约。
- `reports/`：模型选择、训练数据覆盖和最终验收的精简指标，见 [报告索引](reports/README.md)。
- `reproduction/`：初始化权重训练链所需的冻结源码和哈希清单。

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

输出 `observations.json`（每张截图的卡框、身份、未识别卡框、来源指纹和字段）、`box.json`（合并结果）、带序号及来源指纹的标注图。同名输入文件不会覆盖彼此的标注图。成员与留影使用不同 ID 命名空间。

## 接入组卡推荐

可选的正式 `deck-ui` 入口将截图识别、显式人工补齐和 [OurNotes Deck](https://github.com/empty-sekai/ournotes-deck) 的真实搜索连在一起：

```powershell
python boxvision.py deck-ui --data DATA --deck-data deck-data.json --solver-bin ournotes-recommend.exe --workspace ./local/deck-ui --port 18790 --box-port 18793
```

打开 `http://127.0.0.1:18790/`。需要与图库匹配的 DeckData 和支持 `ournotes-deck.recommendation-request/1` 的推荐程序；源码不附带游戏资源。未知字段不会默认填为等级／Rank 1，完整持有集合须由使用者声明。详细补齐契约、mock 复现、版本校验和目标语义见 [组卡输入管线](docs/DECK_PIPELINE.md)。

## 数据和模型

运行目录需要使用者自行准备 `catalog.json` 和卡面图，再把权重包中的 7 个 ONNX 文件和 `recognition.json` 放进该目录的 `models/`。`recognition.json` 记录每个模型的文件名、SHA-256、输入输出和接受阈值；启动时逐个校验哈希，缺文件或哈希不符都拒绝启动。`models/member-gallery.npz`、`models/snap-gallery.npz` 是可重建的图库缓存，指纹包含编码器哈希和每张参考卡面的字节。

公开附件只包含源码与权重，不包含游戏资源或开箱即用的图库。`scripts/prepare.py` 将显式提供的 master、UI、字体与卡面目录组装为数据目录；APK 导出与完整步骤见 [资源准备](docs/RESOURCES.md)。已有数据目录时，`boxvision.py prepare --data DATA` 从 master 表和卡面重建目录和图库，`--offline` 可禁止补取素材。

## 校对和冲突语义

- 等级视图只读取等级；特训视图只读取特训次数；能力／百分比视图不转换为等级。
- `unidentified` 中的卡框不进入合并结果；它们可能是图库外的新卡，也可能是被切掉太多或画面太差的已知卡。
- 同一字段只有一个已观察到的值时合并该值；存在不同值时置为 `null` 并保留 `conflicts` 和全部来源，不取最大值。
- 重复截图不改变最终字段及来源证据；观察次数保留上传事实。
- 手工修正包含之前的值及修正记录。手工校对不能计入模型自动识别准确率。
- 游戏更新、UI 变更、未见过的布局或强降质图可能漏卡／漏字段。用户仍应检查总数及待校对项。

## 训练和验收

训练以合成数据为主：按列表组件的原生几何把原始卡面、UI 精灵和客户端字体渲染成整屏截图，覆盖简／繁中、日、英、韩五种语言、七种参数视图、不同屏幕比例、分辨率、UI 缩放、JPEG/WebP 多次压缩、模糊、滚动裁切、浮层遮挡和图库外卡面。真实截图只用于回归，不能充当独立泛化准确率。

见 [训练与验证说明](docs/TRAINING.md)、[字体证据](docs/evidence/language-fonts.json)、[模型卡](docs/MODEL_CARD.md)、[数据卡](docs/DATA_CARD.md) 和 [覆盖边界](docs/COVERAGE.md)。模型和阈值在开发集上冻结后，在两组各 2000 张、分辨率与压缩配置都不同于训练的整屏测试上评估：图库卡身份接受值零错误，召回 96.8%／97.0%；可见等级、特训次数和卡阶在图库卡上零误读，覆盖率分别约 96%、96% 和 99.6%。详见 [评估卡](docs/EVALUATION.md)。

## 检查

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

涵盖模型清单校验、跨图合并、同名不同来源、冲突保存、隐藏字段及裁切字段／图标等数据完整性回归。

可用 `python -m pip install -e ".[test]"` 安装开发入口和 Schema 校验依赖，再执行 `python scripts/validate_contract.py box result/box.json`。安装后提供 `ournotes-boxlens` 命令，参数与 `python boxvision.py` 相同。
