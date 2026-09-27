# 从本地游戏文件重建训练资源

资源准备需要 Python 3.11+。本项目使用公开 nnnotes 的固定提交 `bb542af0d7e22a0255a7ee45099a51eb7fbadd12`，不再导入开发者工作区里的辅助脚本，也不自动发现私有配置。字体和 UI 从用户提供的 APK 导出，解密配置显式传入。

```bash
python -m pip install -r requirements/runtime.txt -r requirements/resources.txt
python scripts/extract_ui.py --apk LOCAL/base.apk --config LOCAL/nnnotes.toml --cache CACHE --output exports/game-ui
python scripts/export_fonts.py --apk LOCAL/base.apk --config LOCAL/nnnotes.toml --cache CACHE --master-text MASTER/MasterText.json --output exports/card-fonts
python scripts/prepare.py --master-dir MASTER --ui-dir exports/game-ui --font-dir exports/card-fonts --art-dir ART --offline --output data
```

各路径均由使用者指定，输出目录需要为空。`LOCAL/nnnotes.toml` 采用 nnnotes 的配置格式，提供本地资源包所需解密信息。配置不复制到输出、不打印到日志、不进入仓库；这里不需要登录游戏账号。

`MASTER` 是相匹配版本的已解码 master JSON 目录，可由 nnnotes 的 `master decode` 流程获得。必须包含成员、留影、文本、等级上限和卡阶表。原训练使用 APK 内嵌培养表及含 Snap 70 的后续卡片／文本快照；不同快照不能冒充字节相同的数据集。`provenance.json` 记录实际输入文件哈希，验证报告需关联该清单。

`ART` 的命名约定：

- `member-<assetID>.webp`：成员通常缩略图。
- `member-<assetID>-square.webp`：成员 square 图，用于列表的等比填充裁切。
- `snap-<assetID>.webp`：留影缩略图。

`--offline` 禁止联网。省略该选项时，缺少的卡图按准备器中的现有 bdon.moe 路由下载；实际目录记录来源 URL 和哈希。解密后的 UI/font 数据和用户截图不因运行本工具而上传到任何地方。

组装过程复制本项目内的静态渲染器，解码卡阶等精灵，生成 canonical 图、catalog 和 RootSIFT 索引，并为导出资源记录哈希。模型权重单独放入 `data/models/encoder.onnx` 和 `data/models/fields.onnx`；第一次推理会生成可重建图库缓存。没有权重也可以先运行合成数据生成。以上提取结果仅保留在使用者本地，不属于公开源码／权重附件；禁止将其加入本项目公开发布包。

## 已验证的复现范围

对同一 APK 和 master 快照：新的显式参数字体导出与原训练清单完全一致，共 306 个字形；新导出的 UI 组装出 122 张卡，缺失素材为 0；五语言 × 成员七模式／留影五模式，共 60 种组合，与原研究导出的 RGBA 像素哈希全部相同。

这证明资源准备入口可以复现本次静态渲染输入，不表示静态渲染与运行中 Unity 在所有设备上逐像素一致。字体 SDF 抗锯齿仍为近似；不同 Pillow、UnityPy、nnnotes 或资源版本可能改变输出，须记录版本并重新验收。

上游固定版本：[nnnotes 提交](https://github.com/MetaSekaiLab/nnnotes/commit/bb542af0d7e22a0255a7ee45099a51eb7fbadd12)。上游配置与资源解码说明以该版本文档为准。
