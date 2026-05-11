# Character RP Factory

从剧情 CSV 构建角色扮演资产：清洗台词、导出微调样本、生成剧情 RAG 文档，并提供本地角色聊天 WebUI 与可选语音播放。

本仓库按“目标角色代号”组织数据和产物。默认配置指向一个本地私有角色数据集`hski`。

## 项目定位

Character RP Factory 不是单纯的台词抽取脚本。它把原始剧情 CSV 处理成三类资产：

- **训练数据**：带上下文的角色回复样本，支持原文/译文 JSONL 导出。
- **RAG 知识库**：覆盖目标角色出场剧情的场景卡、人物关系、角色弧线、世界观和口吻资料。
- **本地角色聊天**：基于 Ollama + RAG 的本地 WebUI，支持历史聊天、来源追溯和可选语音输出。

为了避免丢失剧情语境，RAG 部分会从完整原始 CSV 场景读取，而不是只看已经抽出来的目标角色台词。

## 当前能力

- 递归合并 `CSV/**/*.csv`，保留原始剧情来源。
- 清洗全量对话，筛选目标角色相关台词。
- 为每条目标角色回复拼接同场景前文上下文。
- 给样本打分，导出人工审核 CSV 和训练 JSONL。
- 从原始 CSV 切分章节、场景、对话块，生成目标角色相关场景卡。
- 支持本地 LLM 批量摘要场景，并把摘要合并成 RAG Markdown。
- 使用 ChromaDB 或内置 simple 后端构建本地向量索引。
- 提供检索、知识问答、角色聊天命令。
- 提供现代聊天应用风格 WebUI：历史聊天、流式回复、来源折叠、检索片段、原文追溯。
- 可选接入本地语音合成服务：页面显示中文回复，播放时可先翻译成目标语种再合成。

## 推荐标题

如果是 GitHub 仓库标题，我最推荐：

**Character RP Factory: Game Dialogue to RAG, LoRA & Voice Chat**

其他候选：

- **Character RP Factory**
- **Game Dialogue RP Factory**
- **CSV-to-Character-RAG**
- **Character RP Lab**
- **Dialogue-to-RP Toolkit**

我的取舍：`Character RP Factory` 适合作为仓库名，短、可扩展；副标题用 `Game Dialogue to RAG, LoRA & Voice Chat` 说明完整能力。

## 目录结构

```text
.
├── CSV/                         # 原始剧情 CSV，本地数据，不建议提交
├── config.example.yaml          # 配置模板
├── config.yaml                  # 本地实际配置
├── data/
│   ├── characters/<character_id>/ # 角色种子资料
│   ├── processed/                # 清洗中间产物
│   ├── rag_docs/                 # 最终 RAG Markdown 与 scenes 场景卡
│   └── chroma_db/                # 本地向量库，已 gitignore
├── outputs/                      # CSV/JSONL/摘要/报告等生成物
├── scripts/
├── src/crpf/                     # CLI、RAG、WebUI、TTS 实现
└── tests/
```

本地语音权重、参考音频和生成音频默认放在独立本地目录，已通过 `.gitignore` 排除。

## 安装

建议 Python 3.11+。

```bash
python3 -m pip install -e .
```

如果要使用 ChromaDB 向量库：

```bash
python3 -m pip install -e ".[rag]"
```

如果不安装包，也可以继续用：

```bash
PYTHONPATH=src python3 -m crpf.cli --help
```

安装后可直接使用：

```bash
crpf --help
```

## 配置

复制模板后编辑：

```bash
cp config.example.yaml config.yaml
```

关键配置：

- `paths.raw_csv_dir`: 原始 CSV 目录，默认 `CSV`
- `paths.output_dir`: 当前角色样本输出目录，建议使用 `outputs/<character_id>`
- `character.target_names`: 目标角色别名
- `rag.embedding_model`: embedding 模型，默认 `bge-m3`
- `rag.chat_model`: 本地聊天模型，默认 `qwen3.5:9b`
- `rag.ollama_base_url`: Ollama 地址，默认 `http://localhost:11434`
- `tts.base_url`: 本地语音合成 API 地址，默认 `http://127.0.0.1:9880`
- `tts.output_dir`: 语音缓存目录

## 训练样本流水线

从原始 CSV 到可微调 JSONL：

```bash
crpf merge --config config.yaml
crpf clean --config config.yaml
crpf build-context --config config.yaml
crpf score --config config.yaml
crpf export-jsonl --config config.yaml --language both
```

主要产物：

- `outputs/<character_id>/merged.csv`
- `outputs/<character_id>/cleaned.csv`
- `outputs/<character_id>/character_lines.csv`
- `outputs/<character_id>/samples_with_context.csv`
- `outputs/<character_id>/good_samples.csv`
- `outputs/<character_id>/review_samples.csv`
- `outputs/<character_id>/training_samples_ja.jsonl`
- `outputs/<character_id>/training_samples_zh.jsonl`

说明：

- `character_lines.csv` 只是目标角色回复索引。
- `build-context` 会回到完整 `cleaned.csv`，按同一个 `source_file` 和原始行号取前文，不会用“只剩目标角色台词”的文件硬拼上下文。
- `review_samples.csv` 带人工审核字段：`keep`、`notes`、`emotion`、`intent`、`tone`、`relationship`。
- `export-jsonl` 支持 `--format chatml` 和 `--format instruction`。

## RAG 文档流水线

快速生成模板：

```bash
crpf build-rag-docs --config config.yaml
```

从完整原始 CSV 构建目标角色相关场景资料：

```bash
crpf summarize-raw-rag --config config.yaml
```

使用本地 LLM 阅读场景并生成结构化摘要：

```bash
ollama pull qwen3.5:9b
crpf summarize-scenes-llm --config config.yaml --model qwen3.5:9b --summary-mode fast
```

`fast` 模式会关闭 thinking、缩短 prompt、限制生成长度，适合批量处理大量场景。测试时可以加：

```bash
crpf summarize-scenes-llm --config config.yaml --model qwen3.5:9b --summary-mode fast --limit 5
```

场景摘要完成后：

```bash
crpf review-rag --config config.yaml
crpf prepare-consolidation-inputs --config config.yaml
```

如果使用外部网页模型做二次合并，把结果放到 `outputs/consolidation_outputs/` 后同步：

```bash
crpf sync-consolidated-rag --config config.yaml
```

最终 RAG 文档：

- `data/rag_docs/index.md`
- `data/rag_docs/character_profile.md`
- `data/rag_docs/plot_summary.md`
- `data/rag_docs/relationships.md`
- `data/rag_docs/worldbuilding.md`
- `data/rag_docs/team_story.md`
- `data/rag_docs/dialogue_patterns.md`
- `data/rag_docs/scenes/*.md`

编辑原则：

- RAG 文档写稳定事实、剧情摘要、关系、口吻规则和世界观。
- 不要把原始台词整段复制进 RAG 文档。
- 不确定内容标 `待核对`。
- 角色性格和说话模式放 `character_profile.md` / `dialogue_patterns.md`。
- 剧情事实放 `plot_summary.md`，关系放 `relationships.md`，场域和活动设定放 `worldbuilding.md`。

## RAG 入库与问答

推荐使用 Ollama `bge-m3` 做 embedding：

```bash
ollama pull bge-m3
crpf build-rag-index --config config.yaml --reset
```

如果没有 ChromaDB，可以使用内置 simple 后端：

```bash
crpf build-rag-index --config config.yaml --backend simple --reset
```

检索验收：

```bash
crpf rag-query --config config.yaml "目标角色和关键人物的关系是什么？"
crpf rag-query --config config.yaml "目标角色的核心压力来自哪里？"
crpf rag-query --config config.yaml "目标角色的说话风格有什么特点？"
```

知识问答：

```bash
crpf rag-ask --config config.yaml "目标角色为什么害怕失败？"
crpf rag-ask --config config.yaml --top-k 4 "目标角色和制作人的关系是什么？"
crpf rag-ask --config config.yaml --show-context "目标角色的说话风格有什么特点？"
```

`rag-ask` 是知识问答，不做角色扮演。它会根据 RAG 资料用中文回答，并输出来源。

## 角色聊天

命令行单轮聊天：

```bash
crpf <role-chat-command> --config config.yaml "今天有点累，鼓励我一下"
crpf <role-chat-command> --config config.yaml --show-sources "讲讲某个关键剧情事件"
crpf <role-chat-command> --config config.yaml --mode rag --show-sources "目标角色和关键人物是什么关系？"
```

交互聊天：

```bash
crpf <role-chat-command> --config config.yaml
```

交互中可以输入 `/auto`、`/rag`、`/casual` 切换模式，输入 `/exit` 退出。

角色聊天的默认 `auto` 模式会自动判断是否需要检索：剧情、设定、关系、口吻问题会查 RAG；日常聊天、鼓励和计划建议会直接按角色人格回答。

> 当前代码里的角色聊天命令名保留了早期 demo 命名。公开使用前，可以把 CLI 入口重命名成更通用的 `chat-character` 或 `role-chat`。

## WebUI

启动本地页面：

```bash
crpf webui --config config.yaml --port 8765
```

然后打开：

```text
http://127.0.0.1:8765
```

也可以自动打开：

```bash
crpf webui --config config.yaml --port 8765 --open
```

WebUI 支持：

- 现代聊天应用风格对话布局。
- 左侧历史聊天，SQLite 存在 `data/chat_history.sqlite3`。
- 流式角色回复。
- `auto/rag/casual` 模式切换。
- RAG 来源折叠和检索片段折叠。
- 来源新内容小蓝点提示。
- 原文追溯：从来源跳到场景卡和原始对话。
- 可选播放角色语音。

## 本地语音

WebUI 的可选语音链路是：

```text
中文回复显示在页面上
-> 点击“播放语音”
-> 后端按配置翻译成目标语种台词
-> 本地语音服务合成
-> 音频缓存到本地输出目录
```

本地语音 API 示例：

```bash
cd "/path/to/local-voice-service"
python api_server.py --host 127.0.0.1 --port 9880 --config /path/to/tts_config.yaml
```

配置示例：

```yaml
tts:
  enabled: true
  provider: local_tts
  base_url: http://127.0.0.1:9880
  output_dir: path/to/local/audio/cache
  ref_audio_path: path/to/local/reference.wav
  prompt_lang: ja
  text_lang: ja
  translate_to_japanese: true
```

同一句回复会命中缓存，不会重复翻译和合成。缓存旁边的 `.json` 会记录原中文、目标语音文本、模型和路径。

如果要切回中文直出语音：

```yaml
tts:
  text_lang: zh
  translate_to_japanese: false
```

## 数据和版权边界

本仓库默认不提交：

- 原始 `CSV/` 数据。
- 具体角色素材、商业作品素材或私有语音数据。
- 本地语音权重、参考音频、生成音频。
- ChromaDB / SQLite 本地数据库。
- Python `__pycache__`。

这些内容都应保留在本地。RAG 文档也应以摘要和人工整理为主，不以复刻原始作品长台词为目标。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

当前测试覆盖 CSV 清洗、上下文构造、JSONL 导出、RAG 索引、RAG 问答、角色聊天、WebUI、来源追溯和 TTS 缓存逻辑。

## 常用命令速查

```bash
crpf merge --config config.yaml
crpf clean --config config.yaml
crpf build-context --config config.yaml
crpf score --config config.yaml
crpf export-jsonl --config config.yaml --language both

crpf summarize-raw-rag --config config.yaml
crpf summarize-scenes-llm --config config.yaml --model qwen3.5:9b --summary-mode fast
crpf review-rag --config config.yaml
crpf prepare-consolidation-inputs --config config.yaml
crpf sync-consolidated-rag --config config.yaml

crpf build-rag-index --config config.yaml --reset
crpf rag-query --config config.yaml "讲讲某个关键剧情事件"
crpf rag-ask --config config.yaml "目标角色和关键人物是什么关系？"
crpf webui --config config.yaml --port 8765
```
