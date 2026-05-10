# 花海咲季 RAG 文档索引

## 文档层级

- `data/rag_docs/*.md`：高层知识文档，适合召回角色画像、关系、剧情线、世界观和口吻规则。
- `data/rag_docs/scenes/*.md`：500 张场景卡，适合追溯具体剧情和证据。
- `outputs/scene_summaries.jsonl`：结构化场景摘要，可作为程序化 metadata 来源。

## 数据状态

- 场景摘要数：500
- 咲季出场状态：direct: 391；mentioned: 106；background: 3
- 场景卡目录：`data/rag_docs/scenes`

## 文档说明

| 文档 | 类型 | 用途 | 推荐检索场景 | scene_id 覆盖 |
| --- | --- | --- | --- | --- |
| `character_profile.md` | profile | 咲季稳定人物画像、动机、性格、成长弧线、弱点和行为模式。 | 回答咲季是什么样的人、为什么行动、怎样成长时优先检索。 | 147 |
| `plot_summary.md` | plot | 咲季相关主线、支线、章节事件和关键舞台整理。 | 回答某段剧情发生了什么、某个事件如何影响咲季时优先检索。 | 318 |
| `relationships.md` | relationships | 咲季与制作人、佑芽、琴音、手毬、星南等人物的稳定关系。 | 回答角色关系、称呼、互动边界、外部评价时优先检索。 | 112 |
| `worldbuilding.md` | worldbuilding | 初星学园、宿舍、训练、选拔、H.I.F/N.I.A、偶像活动等设定。 | 回答制度、地点、赛事、课程、工作设定时优先检索。 | 169 |
| `team_story.md` | team_story | Re;IRIS、宿舍、班级、团队协作和竞争型友情。 | 回答团队互动、组合冲突、共同生活、队友协作时优先检索。 | 115 |
| `dialogue_patterns.md` | dialogue | 咲季中文口吻、称呼、情绪节奏、关系互动和 RP 写作边界。 | 生成咲季回复、约束语气、避免 OOC 时优先检索。 | 132 |

## 检索建议

- 角色扮演生成：优先检索 `character_profile.md` + `dialogue_patterns.md`，再按问题补充 `relationships.md` 或场景卡。
- 剧情问答：优先检索 `plot_summary.md`，再用 `scene_id` 召回 `scenes/*.md` 追证据。
- 人物关系问答：优先检索 `relationships.md`，关系不够细时召回对应场景卡。
- 设定问答：优先检索 `worldbuilding.md`，避免把 AU/活动限定内容当主线设定。
- 团队与 Re;IRIS 问答：优先检索 `team_story.md`。

## 证据回查

- 文档里的 `scene-xxxxx` 可在 `outputs/scene_summaries.jsonl` 中找到结构化摘要。
- 对应 Markdown 场景卡位于 `data/rag_docs/scenes/scene-xxxxx_*.md`。
- 如需原始 CSV 上下文，可用 `source_file` 和 `source_row` 回查 `CSV/`。
