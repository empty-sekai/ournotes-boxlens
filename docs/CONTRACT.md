# 输入输出契约 v1

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

`data` 为 base64 编码图片，允许带 `data:image/...;base64,` 前缀，不接受下载 URL。`images` 为 1–30 个有序元素；`name` 可省略，提供时为 1–150 字符；`player` 可省略，最多 100 字符。兼容旧界面时允许省略 `schema`，其含义仍固定为 v1。未知字段、版本和类型返回 HTTP 400，不再静默转字符串或截断。

请求体必须小于 48 MiB；单图解码后最多 2000 万像素。失败响应符合 `error.schema.json`：`{"error":"说明"}`。成功响应包含 `schema: ournotes-boxlens.scan-result/1`、合并 `box` 和与输入顺序一致的 `scans`，符合 `scan-response.schema.json`。

## CLI 和 GUI 导出

`scan --data DATA --output OUTPUT IMAGE...` 输出：

| 文件 | 契约 |
|---|---|
| `box.json` | `box.schema.json`；GUI 导出的 JSON 也使用同一契约 |
| `observations.json` | `observations.schema.json`，按输入顺序排列 |
| `NNN-HASH-annotated.jpg` | 标注图；相对文件名记录在对应 scan 的 `annotated_file` |

已存在的合并数据版本标识 `bdon-box/1` 继续保留，仓库更名不改变其语义。`observations` 计上传观察次数，`unique_count` 为唯一卡数，`duplicate_observations` 为两者之差。重复上传不改变合并字段、冲突和来源证据，但观察计数反映上传事实。

## 字段和证据

- 每张卡以 `(kind,id)` 标识，`kind` 为 `member` 或 `snap`。`id` 是 master ID；`name` 是当前图库展示名，不用于身份匹配。
- `level`、`card_rank`、`awake_count` 均为 `{value,confidence,reason?}`。`value: null` 表示未知；卡阶／特训范围为 1–5，等级还要遵守目录中的培养上限。Snap 的 `awake_count` 保持未知。
- `display_mode` 是当前参数裁块的识别类别：`level`、`training`、`other` 或 `unknown`。`other` 包含能力值、百分比或隐藏参数，不能据此补全培养值。
- bbox 是解码图上的浮点像素 `[x,y,width,height]`；原点左上，x 向右、y 向下。部分可见卡的 bbox 可超出图像。
- `source_id` 对解码后的 BGR 像素和形状计算 SHA-256；压缩文件哈希和像素哈希不可混称。`source`／`sources` 是展示名称，不能单独作为唯一标识。
- 冲突对象的键是十进制字符串值，对应全部来源证据。冲突字段最终值为 `null`，不取最大值。
- 手工修正的字段增加 `source: manual`，`corrections` 记录原值、原冲突和新值；原始 scan 观察不被修改。
- 可选的 `recognition_provenance` 记录识别时的模型、图库／索引及识别代码哈希、阈值和 ONNX Runtime 版本；合并 box 保留所有不同来源版本，便于追溯。旧导出没有此字段时仍符合契约，不能为旧数据补造来源记录。

Schema 检查类型与结构；跨字段约束还需数据审计，包括唯一键、计数关系、合法培养范围及冲突值与最终值一致性。模型指标不能包含手工修正结果。
