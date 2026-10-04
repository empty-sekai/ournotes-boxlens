# 新卡接入，无需重新训练

卡面编码器输出通用向量，不输出固定 master ID 分类；身份由截图卡面与本地图库的向量检索决定。新卡只需加入图库，模型权重不变。

1. 备份当前完整数据目录，更新 `master/MasterMemberCard.json`／`MasterSupportCard.json` 及对应的 `MasterText.json`。培养规则变化时同步等级、觉醒和卡阶表；不要混用不兼容版本。
2. 放入对应卡面文件。成员需要 `native/assets/member-<assetID>.webp` 和 `member-<assetID>-square.webp`；留影需要 `snap-<assetID>.webp`。准备程序对缺失文件使用已有 bdon.moe 资源地址补取，并验证图像可解码；`--offline` 禁止补取。
3. 执行 `python boxvision.py prepare --data DATA`。检查输出的 `unavailable` 列表，缺失素材的卡片不会进入可识别目录。准备结束时会用当前编码器重建 `models/member-gallery.npz` 和 `models/snap-gallery.npz`。
4. 重启识别服务。图库缓存指纹包含编码器哈希、卡片顺序和每张参考卡面的字节，目录或卡面变化后自动重建，不能只手改 `catalog.json` 而沿用旧缓存。
5. 用新卡截图确认 ID、定位、字段和旧卡回归。保存 master 表、素材和模型哈希及验证报告。

新卡加入图库之前，截图中的新卡通常列在 `unidentified` 中，也可能被误认成外观相近的已知卡（最终测试中图库外卡面的误接受率约 1%）。新卡上线后应尽快更新图库。

新卡使用旧的 UI／字体时通常不需模型训练；如果布局、数字字体、卡阶图标或渲染发生变化，需要额外验证，必要时更新渲染器并重训参数或卡阶模型。

`scripts/verify_gallery_update.py` 是集成验证：在独立目录里从目录中暂时移除指定身份，确认它们进入 `unidentified`；再加回目录条目，确认重新识别，并检查模型文件哈希未变。请提供自己的本地图片，包含待验证的身份；仓库不分发游戏截图。示例：

```bash
python scripts/verify_gallery_update.py --data DATA --output EMPTY_OUTPUT --member-image member-test.jpg --snap-image snap-test.jpg --member-id 7 --snap-id 70
```

它不改生产数据，不把集成检查称为独立准确率。

训练外身份测试采用 `id % 7 == 0` 的 ID，它们不进入编码器训练的查询和参考分支，只在评估时作为图库成员出现。图库外卡面测试使用不属于目录的美术图，测量未知卡是否被误认成已知卡；这两项测试分别验证“加入图库即可识别”和“未加入时拒识”。
