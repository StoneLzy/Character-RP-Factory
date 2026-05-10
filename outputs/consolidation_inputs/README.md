# RAG 大总结输入包

这些 Markdown 是给网页版 GPT 做二次总结用的输入材料，不会调用本地 LLM。

## 使用方式

1. 按目标文档逐个上传或粘贴对应 `*_input.md`。
2. 使用文件开头的提示词，让网页版 GPT 输出最终 RAG Markdown。
3. 把输出保存/替换到 `data/rag_docs/` 对应文件。
4. 保留 `scene_id` 和 `source_file` 作为证据索引，不要复制长段原始台词。

## 文件对应关系

- `character_profile_input.md` -> `data/rag_docs/character_profile.md`
- `relationships_input.md` -> `data/rag_docs/relationships.md`
- `plot_summary_input.md` -> `data/rag_docs/plot_summary.md`
- `worldbuilding_input.md` -> `data/rag_docs/worldbuilding.md`
- `team_story_input.md` -> `data/rag_docs/team_story.md`
- `dialogue_patterns_input.md` -> `data/rag_docs/dialogue_patterns.md`

## 数据概况

- 场景摘要数：500
- 摘要状态：llm_fast_scene_card: 500
- 咲季出场状态：direct: 391；mentioned: 106；background: 3
- 主题分布：dialogue_style: 472；relationships: 465；team_story: 304；worldbuilding: 280；character_arc: 200；plot: 96

## 通用总结要求

- 用中文总结。
- 输出面向 RAG 检索的稳定事实，不写成剧情赏析文章。
- 区分确定事实、推断、单场景事件。
- 每个重要结论后保留若干 `scene_id` 作为证据。
- 不要长篇复述原台词。
- 遇到冲突信息时写 `待核对`，不要强行合并。
