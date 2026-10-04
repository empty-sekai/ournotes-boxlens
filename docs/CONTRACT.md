# 输入输出契约

Schema 位于 `schemas/`，使用 JSON Schema 2020-12。发布的顶层字段和含义发生不兼容变化时必须升级版本；不静默改变 `null`、卡片 ID 命名空间或冲突语义。

## HTTP

`POST /api/scan`，`Content-Type: application/json`：

```json
{
  "schema": "ournotes-boxlens.scan-request/1",
  "player": "local",
  "images": [
    {"name": "member-list.png", "data": "data:image/png;base64,..."}
  ]
}
```

`data` 为 base64 编码图片，允许带 `data:image/...;base64,` 前缀，不接受下载 URL。`images` 为 1–30 个有序元素；`name` 可省略，提供时为 1–150 字符；`player` 可省略，最多 100 字符。允许省略 `schema`，其含义仍固定为请求 v1。未知字段、版本和类型返回 HTTP 400，不静默转字符串或截断。

请求体必须小于 48 MiB；单图解码后最多 2000 万像素。失败响应符合 `error.schema.json`：`{"error":"说明"}`。成功响应包含 `schema: ournotes-boxlens.scan-result/2`、合并 `box` 和与输入顺序一致的 `scans`，符合 `scan-response.schema.json`；`scans` 的每个元素与 `observations.json` 中的单张截图结构相同。

## CLI 和 GUI 导出

`scan --data DATA --output OUTPUT IMAGE...` 输出：

| 文件 | 契约 |
|---|---|
| `box.json` | `box.schema.json`；GUI 导出的 JSON 也使用同一契约 |
| `observations.json` | `observations.schema.json`（`urn:ournotes-boxlens:schema:observations:2`），按输入顺序排列 |
| `NNN-HASH-annotated.jpg` | 标注图；相对文件名记录在对应 scan 的 `annotated_file` |

合并数据版本标识为 `bdon-box/1`。`observations` 计上传观察次数，`unique_count` 为唯一卡数，`duplicate_observations` 为两者之差。重复上传不改变合并字段、冲突和来源证据，但观察计数反映上传事实。

## 单张截图

- `cards`：已确认身份的卡。每张卡以 `(kind,id)` 标识，`kind` 为 `member` 或 `snap`。`id` 是 master ID；`name`、`rarity`、`card_type` 取自当前图库，不用于身份匹配。
- `unidentified`：定位到但身份未被接受的卡框，包括图库外的新卡、被切掉或遮挡过多的卡和画面太差的卡。`candidate` 为最相近的图库卡 `{id,name}` 或 `null`，只供人工确认，不是识别结果；`review` 恒为 `true`。这些卡框不进入合并 `box`。
- 两个列表的元素都含 `bbox`、`locator_score`、`visible_fraction`、`identity_similarity`、`identity_margin`、`level`、`card_rank`、`awake_count`、`display_mode` 和 `review`。`identity_similarity` 是与最相近图库卡的余弦相似度，`identity_margin` 是它与第二名的差值；二者和 `locator_score` 都是内部得分，不是校准概率。
- `cards` 中的 `review` 在等级或卡阶未知时为 `true`。
- 列表按卡框所在行、再按横坐标排序。

## 字段和证据

- `level`、`card_rank`、`awake_count` 均为 `{value,confidence,reason?}`。`value: null` 表示未知，`reason` 说明原因（如 `cropped`、`cropped_rank_icon`、`not_visible_in_level_view`、`outside_masterdata_level_limit`）。卡阶／特训范围为 1–5，等级还要遵守目录中的培养上限。Snap 的 `awake_count` 保持未知。
- `display_mode` 是当前参数裁块的识别类别：`level`、`training`、`other` 或 `unknown`。`other` 包含能力值、百分比或隐藏参数，不能据此补全培养值。
- bbox 是解码图上的浮点像素 `[x,y,width,height]`，表示整张卡框；原点左上，x 向右、y 向下。部分可见卡的 bbox 可超出图像，`visible_fraction` 为图像内面积占比。
- `source_id` 对解码后的 BGR 像素和形状计算 SHA-256；压缩文件哈希和像素哈希不可混称。`source`／`sources` 是展示名称，不能单独作为唯一标识。
- 冲突对象的键是十进制字符串值，对应全部来源证据。冲突字段最终值为 `null`，不取最大值。
- 手工修正的字段增加 `source: manual`，`corrections` 记录原值、原冲突和新值；原始 scan 观察不被修改。
- `recognition_provenance`（`ournotes-boxlens.recognition-provenance/2`）记录识别时 `catalog.json` 与 `models/recognition.json` 的哈希、两个图库缓存的指纹、识别代码哈希和 ONNX Runtime 版本；合并 box 保留所有不同来源版本，便于追溯。没有此字段的数据不能补造来源记录。

Schema 检查类型与结构；跨字段约束还需数据审计，包括唯一键、计数关系、合法培养范围及冲突值与最终值一致性。模型指标不能包含手工修正结果。
