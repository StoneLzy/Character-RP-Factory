# RAG Chunking Plan

## 目标

把高层 RAG 文档与 500 张场景卡一起入库，形成“高层背景召回 + 精确剧情召回”的双层知识库。

## 文档切分规则

| 文档 | 推荐切分 | metadata.doc_type |
| --- | --- | --- |
| `character_profile.md` | 按二级/三级标题切；每个 chunk 保留当前标题链 | `profile` |
| `relationships.md` | 按人物关系标题切；每个关系单独 chunk | `relationships` |
| `plot_summary.md` | 按来源类型/章节/事件标题切 | `plot` |
| `worldbuilding.md` | 按设定条目和索引小节切 | `worldbuilding` |
| `team_story.md` | 按 Re;IRIS、宿舍、班级、关系条目切 | `team_story` |
| `dialogue_patterns.md` | 按口吻规则、称呼规则、互动对象切 | `dialogue` |
| `scenes/*.md` | 每张场景卡 1 个 chunk；长卡可按“场景理解/结构化线索/证据”拆 | `scene_card` |

## Metadata 字段

- `doc_type`: 上表类型。
- `source_doc`: Markdown 文件名。
- `heading_path`: 当前标题链。
- `scene_ids`: chunk 内出现的 `scene-xxxxx` 列表。
- `source_files`: 对场景卡从 `source_file` 抽取；对高层文档可留空。
- `saki_presence`: 场景卡使用 `direct/mentioned/background`；高层文档可留空或 mixed。
- `topics`: 可从 `outputs/scene_summaries.jsonl` 回填。

## 检索策略

- RP 回复：检索 `profile + dialogue + relationships`，再补 1-3 张 scene_card。
- 剧情问答：检索 `plot + scene_card`。
- 关系问答：检索 `relationships + scene_card`。
- 世界观问答：检索 `worldbuilding`，必要时补 `plot`。
- 团队剧情：检索 `team_story + relationships`。

## 注意

- 不要把长段原始台词入库为生成目标；保留短证据和 scene_id 即可。
- 遇到 `待核对` 或 AU/活动限定内容，生成时应降低置信度。
- 高层文档和 scene_card 都要保留，二者互补，不互相替代。
