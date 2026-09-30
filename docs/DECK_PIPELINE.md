# 截图到组卡推荐

BoxLens 的可选 `deck-ui` 入口调用现有识别器和独立的 ournotes-deck 推荐程序。第三槽是队长；返回成员、对应留影、目标值、终止原因及完成／最优性状态。评分和搜索实现仍维护在 [ournotes-deck](https://github.com/empty-sekai/ournotes-deck)。

## 启动

安装本仓库后，准备自己的完整视觉 `DATA`、同区服 DeckData 和支持 v1 推荐接口的程序：

```powershell
python boxvision.py deck-ui --data DATA --deck-data deck-data.json --solver-bin ournotes-recommend.exe --workspace ./local/deck-ui --port 18790 --box-port 18793
```

浏览器打开 `http://127.0.0.1:18790/`。两项服务仅绑定回环地址，端口占用会拒绝启动，可改为其他独立端口。空 workspace 可直接上传自己的截图；也可使用明确生成的 mock workspace。使用 Ctrl+C 关闭时，界面会终止自己启动的识别子进程。

选择截图，声明是否为 mock，然后识别。所有冲突和低置信度保留未知；下一步上传人工补齐 JSON。输入完整且通过实际 master 校验后，才能选择场景运行推荐。切换场景、重扫或重新补齐会清除旧队伍，并拒绝迟到的旧响应；场景选择保持不变。刷新可载入 workspace 原有的样本，页面明确标注其来源。

## 数据身份与拥有集合

`visual-data-manifest.json` 支持 `region`、`masterVersion`、`deckDataSha256`、`resourceVersion`、`resourceHash`、`officialCatalogSha256`、`catalogSha256`、`indexSha256`、`encoderSha256`、`fieldModelSha256`、`gallerySha256` 及 `files` 相对路径哈希字典。出现的声明均与实际加载内容核对；DeckData 从同一字节快照解析和计算 SHA256。篡改 master 行但保留自报版本也会拒绝。每个推荐请求另冻结实际 DeckData／Roster 字节供程序读取，并保存这些真实输入的哈希与命令。

无 manifest 的旧目录会明确报告缺失／多余身份，不声称完整覆盖。图库身份覆盖只表示识别能力，不能扩大玩家持有集合。`coverage: observed_only` 在人工明确完成所有持有和培养信息前始终保留。不在绑定 master 的身份（例如旧快照留影 #70 不在当前 JP 数据中）不会被映射成其他卡片。

已独立验收的当前 JP 数据是 63 成员、64 留影（127 身份），master 1.0.0.300，客户端 1.0.4。索引 SHA256 `b54e63e94a68c61293bbd9685a8ccca47f5a0882ef9615de3c08c2fcda5e3522`。源码不附带这些游戏资源，也不推断其他区服使用相同数据。新身份只有实际观察或明确补充后才进入 Roster。

## 人工补齐契约

结构化库存是 `ournotes.player-inventory/1`，保留原始 `bdon-box/1`。补齐文件格式如下；示意值仅说明字段，不表示真实账号或当前 master 的合法培养范围：

```json
{
  "schema": "ournotes.inventory-completion/1",
  "source": "manual",
  "evidence": "人工核对的培养页与完整持有集合",
  "mock": false,
  "deckDataSha256": "填写实际 DeckData SHA256",
  "ownershipComplete": true,
  "playerStateComplete": true,
  "player": {
    "characterRanks": {"1": 10},
    "bandItems": {},
    "vipRank": 1,
    "events": [],
    "memory": {"musicRanks": {}, "unlockedMembers": [], "unlockedSupports": []},
    "ownedMemberCardIds": [1],
    "ownedSupportCardIds": []
  },
  "members": [{"id": 1, "level": 20, "awake": 1, "rank": 1, "liveSkillLevel": 2, "gekisouSkillLevel": 2}],
  "snaps": [],
  "addMissingIdentities": [],
  "excludeObservedIdentities": []
}
```

实际文件需提供所有 master 角色的 Rank；完整 owned 集合与导出卡池逐项一致，并至少有五个不同角色。技能等级、角色 Rank、道具、VIP、回忆、完整 owned 集合没有从列表截图取得，必须显式补充。卡片等级、卡阶、特训与技能上界从对应 master 的培养／技能行验证。`null`、缺必要字段或技能等级 999 均不会成为 complete。

未观察身份可在 `addMissingIdentities` 中显式填写 `kind`、`id`、`evidence`；误识别排除使用 `excludeObservedIdentities`，也需证据。人工纠错保留之前的值、冲突及来源。`mock-truth` 补齐仅适用于明确 mock 库存，不能补齐真实玩家输入。

## CLI 与 mock 复现

复用原始 native CardRenderer 生成七成员／三留影截图，单独输出真值，不向识别器传入真值：

```powershell
python boxvision.py deck-mock --data DATA --deck-data deck-data.json --members 7 --snaps 3 --output ./local/mock/fixtures
python boxvision.py scan --data DATA --threads 1 --output ./local/mock/scan ./local/mock/fixtures/01-level-a.png ./local/mock/fixtures/02-level-overlap.png ./local/mock/fixtures/03-training.png ./local/mock/fixtures/04-hidden.png ./local/mock/fixtures/05-snap.png ./local/mock/fixtures/06-conflict.png ./local/mock/fixtures/07-degraded-fields.png ./local/mock/fixtures/08-duplicate.png
python boxvision.py deck-adapt --data DATA --deck-data deck-data.json --region jp --mock --box ./local/mock/scan/box.json --completion ./local/mock/fixtures/completion.json --output ./local/mock
python boxvision.py deck-recommend --deck-data deck-data.json --solver-bin ournotes-recommend.exe --roster ./local/mock/roster.json --request request.json --output ./local/recommend-new
python boxvision.py deck-ui --data DATA --deck-data deck-data.json --solver-bin ournotes-recommend.exe --workspace ./local/mock
```

每次生成和推荐使用新的输出目录，避免覆盖证据。三留影布局包含可见边界断言。推荐需要与 Roster 相邻的 `adaptation.json`，拒绝未经审阅的库存。请求文件遵循 solver 的 `ournotes-deck.recommendation-request/1`；可由界面选择预置场景或上传明确提供的请求。

上传 JSON 在浏览器使用严格 UTF-8 解码并保留原文本传入 HTTP；服务拒绝顶层／嵌套／转义后重复键、NaN、Infinity、非法 UTF-8 和畸形 JSON。合法大整数、小数 token 与 CRLF 原样保存至实际 CLI 请求，浏览器不会先 JSON.parse 再 stringify 将它们丢失。

## 目标与完成状态

综合力按综合力单位显示，不附加随机根种子条件。演出期望、阈值概率和活动指标显示各自语义与有限 seedLaw／声明打法／模型条件。Complete 表示程序确实穷尽了请求范围，最优性限定于该卡池、约束、外部输入、打法与声明根分布；未知现实根种子分布下的无条件最优不在结论中。Skip、综合力和 Live 分别展示，不能相互代替演出验收。

Mission／Battle／Arena 强制撃奏；Battle／Arena 多人结果需要显式外部排名确认到达时间线，不能自动视为 Solo 第一。活动收益使用声明的活动时钟和结果上下文，服务器所选奖励未知时不能声称现实收益保证。

独立 QA 的最终场景矩阵为 **45 PASS / 0 FAIL / 2 BLOCKED**，两项 blocked 因冻结 JP master 没有 Arena 数据；不表示 Arena 已在当前 JP 原生场景实测。synthetic Arena 模型检查单独保留。采用的 R3 推荐程序 SHA256 为 `d645f10a2e23efbdf1ef722f49892a26829871a711e9bc79179006e4afd3a1ef`；已有 48,960 物理队伍 Top3、槽位、留影和逐根结果独立对拍证据，不在此次接入重跑长穷举。数值模型边界是离线 native 1.0.1-25，未据此声称 JP 1.0.4 运行时完全等价。

## 仓库检查

```powershell
python -m pip install -e ".[test]"
python -m pytest -q tests
```

公共回归使用手写 synthetic master 和无游戏图像的 JSON，不分发原始 master、chart、APK、截图、卡面、账号或内部协作文件。原有 scan、serve 与训练路径继续可用。根目录 public-manifest.json 记录原 v0.1.0 发布基线；此 PR 的新输入管线由 git commit 与构建结果记录，未更新或重标旧发布／模型证据。
