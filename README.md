# 企业知识问答系统 (ask-model)

基于 RAG（检索增强生成）架构的企业级知识问答服务，融合阿里百炼远程知识库与本地 Chroma 向量库，集成三级对话记忆系统，对接 DeepSeek 大模型，提供准确、可溯源的企业知识问答能力。

---

## 目录

- [核心功能](#核心功能)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [系统架构](#系统架构)
- [完整 RAG 处理流程](#完整-rag-处理流程)
- [模块详解](#模块详解)
- [三级对话记忆机制](#三级对话记忆机制)
- [配置说明](#配置说明)
- [启动方式](#启动方式)
- [API 接口文档](#api-接口文档)
- [使用前需要做的事](#使用前需要做的事)
- [已知限制与注意事项](#已知限制与注意事项)

---

## 核心功能

| 功能 | 说明 |
|------|------|
| **多路召回** | 并发请求阿里百炼远程知识库 + 本地 Chroma 向量库，合并去重 |
| **Query 改写** | 融合历史对话上下文，由 LLM 生成 1~3 条高质量检索词 |
| **LLM 精排** | 对候选文档按相关性打分重排，结合 Token 截断控制上下文长度 |
| **三级对话记忆** | 短期窗口 + 中期摘要 + 长期向量持久化，支持跨会话知识召回 |
| **Prompt 工程** | 企业角色约束 + 知识库上下文注入 + 三层记忆融合 |
| **结果后处理** | 引用溯源标注、敏感词脱敏、Markdown 格式规范化 |
| **流式/非流式** | 支持 SSE 流式输出和完整响应两种模式 |
| **会话隔离** | 按 session_id 隔离记忆，支持多用户并发 |

---

## 技术栈

| 组件 | 技术选型 | 用途 |
|------|----------|------|
| Web 框架 | FastAPI + Uvicorn | HTTP 服务、SSE 流式响应 |
| LLM | DeepSeek Pro (deepseek-v4-pro) | 查询改写、重排打分、最终生成 |
| 远程知识库 | 阿里百炼 (DashScope) | 企业知识文档远程检索 |
| 向量数据库 | ChromaDB (PersistentClient) | 本地向量存储与语义检索 |
| Embedding | sentence-transformers (text2vec-base-chinese) | 文本向量化（本地离线模型） |
| 配置管理 | pydantic-settings + python-dotenv | 环境变量类型安全加载 |
| HTTP 客户端 | httpx (AsyncClient) | 异步 API 调用，支持系统代理 |
| Token 计算 | tiktoken | 上下文 Token 截断控制 |

---

## 项目结构

```
ask-model/
├── app/
│   ├── __init__.py
│   ├── config.py                 # 配置管理（pydantic-settings，读取 .env）
│   ├── query_rewrite.py          # Query 改写 / 意图理解
│   ├── reranker.py               # 重排 + 上下文压缩（LLM 精排 + Token 截断）
│   ├── prompt_builder.py         # Prompt 工程（角色 + 上下文 + 记忆融合）
│   ├── llm.py                    # LLM 客户端（DeepSeek，流式/非流式）
│   ├── postprocessor.py          # 结果后处理（引用溯源 + 敏感词 + 格式）
│   ├── pipeline.py               # RAG 主流程编排
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py         # 短期记忆（滑动窗口，最近 N 轮完整对话）
│   │   ├── mid_term.py           # 中期记忆（LLM 滚动摘要压缩）
│   │   ├── long_term.py          # 长期记忆（Chroma 向量持久存储）
│   │   └── memory_manager.py     # 三层记忆统一管理器
│   └── retriever/
│       ├── __init__.py
│       ├── bailian_retriever.py  # 阿里百炼远程知识库检索（SSE 流式解析）
│       ├── local_retriever.py    # 本地 Chroma 向量库检索
│       └── hybrid_retriever.py   # 多路召回混合检索器
├── main.py                       # FastAPI 服务入口
├── requirements.txt              # Python 依赖
├── .env                          # 运行时配置（需手动创建）
├── .env.example                  # 配置模板
├── Dockerfile                    # Docker 镜像构建
└── docker-compose.yml            # Docker Compose 编排
```

---

## 系统架构

```
用户 Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI (main.py)                        │
│  /chat  /chat/stream  /documents/upload  /session/*          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                   RAG Pipeline (pipeline.py)                  │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Query 改写  │→ │  多路召回    │→ │  重排+上下文压缩   │ │
│  │(query_rewrite)│  │(hybrid_     │  │  (reranker.py)      │ │
│  │             │  │ retriever)   │  │                     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
│         │                  │                     │           │
│         ▼                  ▼                     ▼           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │Prompt 工程  │→ │  LLM 调用   │→ │  结果后处理         │ │
│  │(prompt_     │  │ (llm.py)    │  │  (postprocessor.py) │ │
│  │ builder.py) │  │              │  │                     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Memory Manager (memory_manager.py)          │   │
│  │  短期窗口 ←→ 中期摘要 ←→ 长期 Chroma 向量记忆        │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
    │                        │
    ▼                        ▼
┌─────────────────┐   ┌──────────────────┐
│  阿里百炼 API   │   │  ChromaDB 本地   │
│  (远程知识库)   │   │  (向量持久存储)  │
└─────────────────┘   └──────────────────┘
```

---

## 完整 RAG 处理流程

```
用户 Query
    │
    ▼
┌─────────────────────────────┐
│  1. Query 改写 / 意图理解     │  融合历史对话，生成 1~3 条改写检索词
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. 多路召回                  │  并发：阿里百炼远程知识库 + 本地 Chroma
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. 重排 + 上下文压缩         │  Score 过滤 → LLM 精排 → Token 截断
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. Prompt 工程               │  企业角色 + 知识库上下文 + 三层记忆融合
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  5. 调用 LLM（DeepSeek Pro）  │  支持流式 / 非流式响应
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  6. 结果后处理                │  引用溯源 + 敏感词过滤 + Markdown 格式整理
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  7. 返回答案 + 更新三层记忆    │  短期窗口写入 → 溢出压缩 → 长期实体提取
└─────────────────────────────┘
```

---

## 模块详解

### Query 改写 (`query_rewrite.py`)

- 融合中期摘要和短期对话历史，构建 LLM 输入
- 使用 Few-shot 示例引导 LLM 输出格式
- 输出 1~3 条检索词，用 `|||` 分隔
- **降级策略**：LLM 不可用时直接返回原始 query

### 多路召回 (`retriever/hybrid_retriever.py`)

| 路径 | 实现 | 说明 |
|------|------|------|
| 远程 | `BailianRetriever` | 调用阿里百炼 API，强制 SSE 流式，解析 tool/assistant 角色事件 |
| 本地 | `LocalRetriever` | ChromaDB + text2vec-base-chinese，支持离线模型加载 |

- 并发执行所有 query 的远程检索（`asyncio.gather`）
- 本地检索同步执行（Chroma 本地操作无需 await）
- 合并去重（按内容前100字符），按 score 降序排序

### 重排 (`reranker.py`)

三阶段处理：
1. **Score 过滤**：移除低质量候选（score < 0.5）
2. **LLM 精排**：调用 DeepSeek 对候选文档按相关性打分（0-10）
3. **Token 截断**：按 `MAX_CONTEXT_TOKENS` 限制截断，防止上下文溢出

### Prompt 工程 (`prompt_builder.py`)

构建的 messages 结构：
```
[system] 企业角色定义 + 知识库参考内容（[来源1]、[来源2]...）
[system] 中期摘要记忆（如有）
[system] 长期记忆召回（相关历史，如有）
[user/assistant] 短期对话历史（最近 N 轮）
[user] 当前用户问题
```

### LLM 客户端 (`llm.py`)

- 对接 DeepSeek OpenAI 兼容 API
- 支持非流式（`chat`）和流式（`chat_stream`）两种模式
- httpx AsyncClient 启用 `trust_env=True`，自动读取系统代理环境变量

### 结果后处理 (`postprocessor.py`)

| 阶段 | 说明 |
|------|------|
| 敏感词过滤 | 对 `password`、`api_key`、`token` 等关键词脱敏（首尾保留，中间用 `***`） |
| 引用溯源 | 解析 `[来源X]` 标注，关联实际来源名称和链接；无引用时自动追加参考来源列表 |
| 格式整理 | 清理连续空行、行尾空白，规范化 Markdown 标题格式 |

---

## 三级对话记忆机制

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Manager                           │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ L1 短期窗口记忆 (ShortTermMemory)                  │    │
│  │ 存储：最近 N 轮完整问答 (default: 6 轮)            │    │
│  │ 触发：每次问答实时更新                               │    │
│  │ 生命周期：会话内                                     │    │
│  └────────────────────────┬───────────────────────────┘    │
│                           │ 溢出                            │
│                           ▼                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ L2 中期摘要记忆 (MidTermMemory)                    │    │
│  │ 存储：LLM 压缩的滚动摘要 (max 500 tokens)          │    │
│  │ 触发：窗口溢出后自动调用 LLM 生成摘要               │    │
│  │ 生命周期：会话内，滚动更新                           │    │
│  └────────────────────────┬───────────────────────────┘    │
│                           │ 提取关键实体/结论               │
│                           ▼                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ L3 长期向量记忆 (LongTermMemory)                   │    │
│  │ 存储：关键实体、业务主题、核心结论                   │    │
│  │ 触发：从溢出对话中规则提取（引号内容、驼峰命名等）   │    │
│  │ 生命周期：跨会话，ChromaDB 持久化                    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**记忆使用方式：**
- 每次问答前调用 `build_context(query)` 融合三层记忆
- 长期记忆以当前 query 进行语义相似度召回（top_k=5）
- 三层记忆分别注入 Prompt 的不同位置，形成完整上下文

---

## 配置说明

所有配置通过 `.env` 文件管理，由 `pydantic-settings` 类型安全加载：

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `BAILIAN_API_URL` | str | **必填** | 阿里百炼知识库 API 地址 |
| `BAILIAN_API_KEY` | str | **必填** | 阿里百炼 API Key |
| `BAILIAN_AGENT_ID` | str | **必填** | 百炼 Agent ID |
| `DEEPSEEK_API_URL` | str | **必填** | DeepSeek API 完整端点（含 `/v1/chat/completions`） |
| `DEEPSEEK_API_KEY` | str | **必填** | DeepSeek API Key |
| `DEEPSEEK_MODEL` | str | **必填** | 模型名称（如 `deepseek-v4-pro`） |
| `CHROMA_PERSIST_DIR` | str | `./chroma_data` | ChromaDB 持久化目录 |
| `CHROMA_COLLECTION` | str | `enterprise_knowledge` | Chroma 集合名称 |
| `EMBEDDING_MODEL` | str | 本地路径或 HF 模型名 | text2vec-base-chinese 模型路径 |
| `SHORT_TERM_WINDOW_SIZE` | int | `6` | 短期记忆保留轮数 |
| `MID_TERM_SUMMARY_THRESHOLD` | int | `10` | 触发中期摘要的溢出轮数 |
| `MID_TERM_MAX_SUMMARY_TOKENS` | int | `500` | 摘要最大 Token 数 |
| `LONG_TERM_TOP_K` | int | `5` | 长期记忆召回条数 |
| `RETRIEVE_TOP_K` | int | `10` | 多路召回数量 |
| `RERANK_TOP_K` | int | `5` | 重排后保留数量 |
| `MAX_CONTEXT_TOKENS` | int | `3000` | 上下文最大 Token 数 |
| `HOST` | str | `0.0.0.0` | 服务监听地址 |
| `PORT` | int | `8010` | 服务监听端口 |

---

## 启动方式

### 本地开发

```bash
# 1. 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建并编辑配置文件
cp .env.example .env
# 编辑 .env，填入必要配置（见配置说明）

# 4. 启动服务
python main.py
# 服务默认运行在 http://0.0.0.0:8010
```

### Docker 部署

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

### PyCharm 调试

项目已兼容 PyCharm Debug 模式，自动检测 `pydevd` 并禁用 `reload`，避免与 WatchFiles reloader 冲突。直接配置 Run/Debug Configuration 指向 `main.py` 即可。

---

## API 接口文档

### 健康检查

```
GET /health
```

**响应：**
```json
{
  "status": "ok",
  "service": "enterprise-knowledge-qa"
}
```

---

### 非流式问答

```
POST /chat
Content-Type: application/json
```

**请求体：**
```json
{
  "query": "返还单合并的场景有哪些",
  "session_id": "user-001"
}
```

**响应：**
```json
{
  "answer": "返还单合并主要有以下场景：...",
  "sources": [
    {"index": 1, "source": "返还单全生命周期分析", "origin": "bailian"}
  ],
  "session_id": "user-001",
  "trace": {
    "rewritten_queries": ["返还单合并场景", "返还单合并规则"],
    "retrieved_count": 8,
    "reranked_count": 5,
    "elapsed_ms": 3200
  }
}
```

---

### 流式问答（SSE）

```
POST /chat/stream
Content-Type: application/json
```

**请求体：**
```json
{
  "query": "返还单合并的场景有哪些",
  "session_id": "user-001"
}
```

**SSE 事件流：**
```
event: status
data: "正在理解您的问题..."

event: status
data: "正在检索知识库..."

event: status
data: "正在生成回答..."

event: chunk
data: "返还单合并"

event: chunk
data: "主要有以下场景"

event: done
data: {"sources": [...], "session_id": "user-001", "trace": {...}}
```

---

### 上传文档到本地向量库

```
POST /documents/upload
Content-Type: application/json
```

**请求体：**
```json
{
  "documents": ["文档内容1", "文档内容2"],
  "source": "confluence"
}
```

**响应：**
```json
{
  "message": "成功上传 2 条文档",
  "total": 150
}
```

---

### 清除会话记忆

```
POST /session/clear
Content-Type: application/json
```

**请求体：**
```json
{
  "session_id": "user-001"
}
```

**响应：**
```json
{
  "message": "会话 user-001 记忆已清除"
}
```

---

### 查看会话记忆状态

```
GET /session/{session_id}/memory
```

**响应：**
```json
{
  "session_id": "user-001",
  "short_term": {
    "total_turns": 3,
    "window_size": 6
  },
  "mid_term": {
    "summary_count": 1
  },
  "long_term": {
    "total_records": 12
  }
}
```

---

## 使用前需要做的事

### 1. 配置 DeepSeek API

编辑 `.env`，填入 DeepSeek API Key 和完整端点：

```
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-v4-pro
```

> **注意**：`DEEPSEEK_API_URL` 必须是完整端点（含 `/v1/chat/completions`），不能只填 base URL。

### 2. 配置百炼知识库

```
BAILIAN_API_URL=your-bailian-api-url
BAILIAN_API_KEY=your-bailian-api-key
BAILIAN_AGENT_ID=your-agent-id
```

> **注意**：百炼 API 强制要求 `stream=True`，系统已自动处理。

### 3. 配置本地 Embedding 模型（可选）

如需使用本地 Chroma 向量库，需准备 `text2vec-base-chinese` 模型：

```bash
# 方式一：使用本地已下载的模型（推荐）
# 在 .env 中设置本地路径：
EMBEDDING_MODEL=/path/to/shibing624-text2vec-base-chinese

# 方式二：让系统自动从 HuggingFace 下载（需网络）
EMBEDDING_MODEL=shibing624/text2vec-base-chinese
```

### 4. 设置系统代理（如需要）

如果 DeepSeek API 需要代理访问，在环境变量中设置：

```bash
export HTTPS_PROXY=http://your-proxy:port
```

httpx 客户端已启用 `trust_env=True`，会自动读取系统代理配置。

### 5. 上传本地文档（可选）

如需使用本地知识库，启动后通过 API 上传文档：

```bash
curl -X POST http://localhost:8010/documents/upload \
  -H "Content-Type: application/json" \
  -d '{"documents": ["文档内容"], "source": "confluence"}'
```

---

## 已知限制与注意事项

| 项目 | 说明 |
|------|------|
| **百炼 API stream 参数** | 强制要求 `stream=True`，不支持非流式调用 |
| **DeepSeek API URL** | 必须填完整端点 `https://api.deepseek.com/v1/chat/completions` |
| **Embedding 模型** | 使用 `local_files_only=True` 时，`EMBEDDING_MODEL` 必须为本地绝对路径 |
| **HuggingFaceEmbeddings API** | 需调用 `embed_documents()` / `embed_query()`，不能直接调用对象 |
| **Query 改写降级** | LLM 不可用时自动降级为原始 query，不影响主流程 |
| **长期记忆提取** | 当前使用简单规则（引号内容、驼峰命名），可按需扩展 NER 提取逻辑 |
| **敏感词过滤** | 默认内置词表，生产环境建议从数据库动态加载 |
| **Chroma 持久化** | `./chroma_data` 目录会自动创建，Docker 部署需挂载为 Volume |
