# 新卡接入，无需重新训练 ID 分类器

系统把截图卡面与可更新的图库匹配。卡面编码模型输出通用向量，不输出固定 master ID 分类；RootSIFT 定位索引和神经检索图库共同更新。

1. 备份当前完整数据目录，更新 `master/MasterMemberCard.json`／`MasterSupportCard.json` 及对应的 `MasterText.json`。培养规则变化时同步等级、觉醒和卡阶表；不要混用不兼容版本。
2. 放入对应卡面文件。成员需要 `native/assets/member-<assetID>.webp` 和 `member-<assetID>-square.webp`；留影需要 `snap-<assetID>.webp`。当前准备程序对缺失文件使用已有 bdon.moe 资源地址补取，并验证图像可解码。
3. 执行 `python boxvision.py prepare --data DATA`。检查输出 `unavailable` 列表，缺失素材的卡片不会进入可识别目录。成员 square 图按当前列表裁切规则构建 canonical 图。
4. 重启识别服务。新目录的顺序、RootSIFT 所有者索引和图库必须是一套，不能只手改 `catalog.json` 后继续使用旧索引。
5. 用新卡截图确认 ID、定位、字段和旧卡回归。图库缓存指纹包含编码器字节、卡片 ID 顺序、canonical 图字节；变化时重建向量。保存 master 表、素材和模型哈希及验证报告。

更新流程不修改模型权重。新卡使用旧的 UI／字体时通常不需模型训练；如果布局、数字字体、卡阶图标或渲染发生变化，需要额外验证，必要时更新渲染器与字段模型。

`scripts/verify_gallery_update.py` 是集成验证：在独立目录暂时移除指定身份，重建索引，再加入其目录及素材；检查卡片重新可识别且编码器哈希未变。请提供自己的本地图片，包含待验证的身份；仓库不分发游戏截图。示例：

```bash
python scripts/verify_gallery_update.py --data DATA --output EMPTY_OUTPUT --member-image member-test.jpg --snap-image snap-test.jpg --member-id 7 --snap-id 70
```

它不改生产数据，不把集成检查称为独立准确率。已完成的公开统计使用 member 7／snap 70 的校准回归；其他使用者可选择自己截图中存在的 ID。

训练外身份测试采用 `id % 7 == 0` 的 ID，禁止进入编码器训练的查询和参考分支。验证阶段才加入图库。另将这些 ID 从图库彻底移除，测量未知卡是否被误认成已知卡；这两项测试分别验证“可添加”和“未添加时拒识”。
