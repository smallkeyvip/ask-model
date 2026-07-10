# ASK 企业知识问答系统

一个面向单企业内部使用的邀请制知识问答应用，不引入租户实体或多租户管理。系统将阿里百炼远程知识库、本地 Chroma 向量库和 DeepSeek 兼容接口组合为 RAG 流程，并提供登录、问答、会话管理、成员邀请、账号启停和文档入库页面。

## 核心能力

- 仅初始管理员和持有有效邀请链接的用户可以进入系统。
- 普通成员可以创建独立对话、进行流式问答、查看来源和删除自己的会话。
- 管理员可以邀请成员、查看邀请状态、启停账号并上传文本知识。
- 问答流程包含输入预处理、查询改写、远程与本地并发检索、RRF 融合、LLM 重排、Prompt 加固和结果脱敏。
- 用户只能访问自己的聊天会话与个人长期记忆；管理员权限不会绕过会话所有权。
- Docker 镜像以非 root 用户运行，构建上下文不包含 `.env` 和持久化数据。

## 快速开始

### 1. 准备环境

- Python 3.11
- 可访问的 DeepSeek 兼容聊天接口
- 阿里百炼应用和知识库接口
- 首次下载默认 Embedding 模型时需要访问 Hugging Face；也可以配置本地模型目录

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，至少填写以下配置：

```dotenv
BAILIAN_API_KEY=your-bailian-api-key
BAILIAN_AGENT_ID=your-bailian-agent-id
DEEPSEEK_API_KEY=your-deepseek-api-key

BOOTSTRAP_ADMIN_EMAIL=admin@example.com
BOOTSTRAP_ADMIN_PASSWORD=replace-with-a-long-random-password
```

`BOOTSTRAP_ADMIN_PASSWORD` 至少需要 10 个字符。初始管理员只会在用户表为空时创建；首次成功启动后，应从 `.env` 中删除该明文密码。

### 2. 启动应用

```bash
python main.py
```

浏览器打开 `http://localhost:8010`，使用初始管理员账号登录。管理员可以在 **系统管理 > 成员邀请** 中生成一次性邀请链接。

服务启动时会初始化 Embedding 模型和 Chroma，首次启动可能需要较长时间。可用以下端点检查状态：

```bash
curl http://localhost:8010/live
curl http://localhost:8010/ready
```

## 前端功能

### 成员工作区

1. 受邀用户打开管理员提供的 `/invite?token=...` 链接。
2. 用户填写姓名和密码完成一次性激活，随后自动登录。
3. 用户创建或选择对话，在问答区提交企业业务问题。
4. 回答通过 SSE 返回，最终内容以服务端后处理后的 `done.answer` 为准。
5. 用户可以查看来源、切换自己的历史会话或删除会话及其个人长期记忆。

### 管理工作区

- **成员邀请**：按邮箱和角色生成限时、一次性邀请链接。
- **文档入库**：上传 UTF-8 文本、Markdown、CSV、JSON，或直接粘贴文本。
- **成员管理**：查看成员并启停账号；停用账号会立即撤销其登录会话。

浏览器会将文件读取为文本，再通过 JSON 发送给后端。PDF、Office 文档、图片 OCR 和二进制文件解析不在当前范围内。

## 系统架构

```text
浏览器
  -> FastAPI 静态前端与邀请制认证
  -> 用户/角色/聊天会话所有权校验
  -> 输入过滤与业务相关性判断
  -> Query 改写
  -> 并发检索
       -> 阿里百炼远程知识库
       -> 本地 Chroma + HuggingFace Embedding
  -> RRF 去重融合与来源覆盖
  -> 严格 JSON LLM 重排与 Token 截断
  -> 防提示注入的知识上下文
  -> DeepSeek 兼容接口生成
  -> 引用处理与敏感值脱敏
  -> SSE / JSON 响应
```

认证数据和聊天会话元数据保存在 SQLite。短期与中期消息保存在当前服务进程，个人长期记忆和本地知识保存在 Chroma。

## 项目结构

```text
ask-model/
├── app/
│   ├── auth.py                    # 用户、邀请、登录令牌和聊天会话所有权
│   ├── config.py                  # 环境配置
│   ├── input_preprocessor.py      # 输入安全、业务判断和指代消解
│   ├── pipeline.py                # RAG 主流程与流式一致性
│   ├── prompt_builder.py          # Prompt 与不可信知识上下文
│   ├── query_rewrite.py           # 查询改写
│   ├── reranker.py                # 严格重排和 Token 控制
│   ├── postprocessor.py           # 引用、格式和敏感值处理
│   ├── memory/                    # 短期、中期和长期记忆
│   └── retriever/                 # 百炼、本地和混合检索器
├── web/
│   ├── index.html                 # 登录、问答和管理页面
│   ├── app.js                     # 前端状态、API 和 SSE
│   └── styles.css                 # 响应式样式
├── main.py                        # FastAPI 路由与生命周期
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 配置说明

所有配置从环境变量或项目根目录的 `.env` 加载。

### 模型与知识库

| 变量 | 默认值 | 说明 |
|---|---|---|
| `BAILIAN_API_URL` | 示例百炼地址 | 百炼知识库聊天端点 |
| `BAILIAN_API_KEY` | 无 | 百炼 API 密钥，必填 |
| `BAILIAN_AGENT_ID` | 无 | 百炼应用 ID，必填 |
| `DEEPSEEK_API_URL` | DeepSeek chat completions | 完整聊天端点 |
| `DEEPSEEK_API_KEY` | 无 | LLM API 密钥，必填 |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | 服务端可用模型名 |
| `EMBEDDING_MODEL` | `shibing624/text2vec-base-chinese` | Hugging Face 模型名或本地目录 |
| `EMBEDDING_LOCAL_FILES_ONLY` | `false` | `true` 时禁止联网下载模型 |

### 存储与检索

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Chroma 持久化目录 |
| `CHROMA_COLLECTION` | `enterprise_knowledge` | 企业知识集合名 |
| `RETRIEVE_TOP_K` | `10` | 融合检索返回数量 |
| `RERANK_TOP_K` | `5` | 重排后保留数量 |
| `MAX_CONTEXT_TOKENS` | `3000` | 知识上下文 Token 上限 |
| `SHORT_TERM_WINDOW_SIZE` | `6` | 短期记忆窗口轮数 |
| `MID_TERM_SUMMARY_THRESHOLD` | `10` | 触发中期摘要的溢出轮数 |
| `MID_TERM_MAX_SUMMARY_TOKENS` | `500` | 单次摘要 Token 上限 |
| `LONG_TERM_TOP_K` | `5` | 个人长期记忆召回数量 |
| `SESSION_TTL_SECONDS` | `3600` | 内存会话空闲归档时间 |

### 认证与安全

| 变量 | 默认值 | 说明 |
|---|---|---|
| `AUTH_DB_PATH` | `./data/auth.db` | SQLite 认证数据库 |
| `AUTH_COOKIE_NAME` | `ask_model_session` | HttpOnly 会话 Cookie 名 |
| `AUTH_SESSION_TTL_SECONDS` | `604800` | 登录令牌有效期，默认 7 天 |
| `INVITATION_TTL_SECONDS` | `604800` | 邀请有效期，默认 7 天 |
| `AUTH_COOKIE_SECURE` | `false` | HTTPS 生产环境必须设为 `true` |
| `BOOTSTRAP_ADMIN_EMAIL` | 空 | 空用户表的初始管理员邮箱 |
| `BOOTSTRAP_ADMIN_PASSWORD` | 空 | 初始管理员密码 |
| `ALLOWED_ORIGINS` | 本机 8010 两个来源 | 逗号分隔的 CORS 来源 |
| `MAX_ACTIVE_CHAT_SESSIONS` | `20` | 每位用户最多保存的聊天会话数 |

密码使用带随机盐的 `scrypt` 保存。邀请令牌和登录令牌只保存 SHA-256 摘要，原始邀请令牌只在创建响应中出现一次。

### 上传与输入

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAX_UPLOAD_DOCUMENTS` | `20` | 单次入库文档数上限 |
| `MAX_UPLOAD_CHARS_PER_DOCUMENT` | `100000` | 单文档字符上限 |
| `DOCUMENT_CHUNK_SIZE` | `1200` | 文档分块字符数 |
| `DOCUMENT_CHUNK_OVERLAP` | `150` | 相邻分块重叠字符数 |
| `ENABLE_INPUT_PREPROCESS` | `true` | 启用敏感词、业务判断和指代消解 |
| `SENSITIVE_KEYWORDS_EXTRA_FILE` | `./sensitive_words_extra.json` | 扩展敏感规则文件 |
| `BUSINESS_DOMAIN_KEYWORDS` | 空 | 逗号分隔的业务关键词 |

`DOCUMENT_CHUNK_OVERLAP` 必须小于 `DOCUMENT_CHUNK_SIZE`。

### 服务

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8010` | 本地和 Compose 暴露端口 |

## Docker 部署

准备 `.env` 后执行：

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f ask-model
```

Compose 使用 `chroma_data` 和 `auth_data` 命名卷，避免非 root 容器访问宿主机 bind mount 时发生 UID 权限冲突。普通停止不会删除数据：

```bash
docker compose down
```

`docker compose down -v` 会永久删除认证数据库和本地向量数据，不应在生产环境随意执行。

生产环境应在应用前配置 HTTPS 反向代理，并设置：

```dotenv
AUTH_COOKIE_SECURE=true
ALLOWED_ORIGINS=https://knowledge.example.com
```

反向代理需关闭 SSE 响应缓冲，或至少确保 `/chat/stream` 可以持续转发事件。

## API

除健康检查、登录和邀请激活外，业务接口都需要登录 Cookie。前端已封装完整流程；直接调用 API 时应先保存 Cookie。

### 认证示例

```bash
curl -c cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"YOUR_PASSWORD"}' \
  http://localhost:8010/api/auth/login
```

连续登录失败会触发单进程窗口限速，并返回 HTTP `429`。

### 创建邀请

管理员接口：

```bash
curl -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"email":"member@example.com","role":"user"}' \
  http://localhost:8010/api/admin/invitations
```

响应中的 `invite_url` 只包含本次邀请的原始令牌。再次邀请相同邮箱会使之前尚未使用的邀请失效。

### 创建会话并问答

```bash
curl -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"title":"返还单规则"}' \
  http://localhost:8010/api/sessions
```

使用响应中的会话 `id`：

```bash
curl -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"query":"返还单的处理规则是什么？","session_id":"SESSION_ID"}' \
  http://localhost:8010/chat
```

流式问答：

```bash
curl -N -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"query":"返还单的处理规则是什么？","session_id":"SESSION_ID"}' \
  http://localhost:8010/chat/stream
```

SSE 可能返回 `status`、`chunk`、`blocked`、`error` 和 `done`。`done` 事件包含完整的最终 `answer`、来源和检索追踪信息。

### 管理员文档入库

```bash
curl -b cookies.txt \
  -H 'Content-Type: application/json' \
  -d '{"documents":["企业规则正文"],"source":"运营制度"}' \
  http://localhost:8010/documents/upload
```

### 路由速查

| 方法与路径 | 权限 | 用途 |
|---|---|---|
| `GET /live` | 公开 | 进程存活检查 |
| `GET /ready` | 公开 | Pipeline、SQLite 和本地文档状态 |
| `GET /health` | 公开 | 兼容健康检查 |
| `POST /api/auth/login` | 公开 | 登录并设置 Cookie |
| `POST /api/auth/logout` | 当前 Cookie | 注销并撤销令牌 |
| `GET /api/auth/me` | 成员 | 当前用户 |
| `POST /api/auth/invitations/accept` | 邀请令牌 | 激活账号 |
| `GET/POST /api/admin/invitations` | 管理员 | 查看和创建邀请 |
| `GET /api/admin/users` | 管理员 | 查看成员 |
| `PATCH /api/admin/users/{user_id}` | 管理员 | 启停成员 |
| `GET/POST /api/sessions` | 成员 | 列出和创建自己的会话 |
| `GET /api/sessions/{id}/messages` | 会话所有者 | 读取当前进程中的消息 |
| `DELETE /api/sessions/{id}` | 会话所有者 | 删除会话和个人记忆 |
| `POST /chat` | 会话所有者 | 非流式问答 |
| `POST /chat/stream` | 会话所有者 | SSE 问答 |
| `POST /documents/upload` | 管理员 | 文本文档分块入库 |

## 安全与运维边界

- 当前认证数据库为单实例 SQLite，登录限速和短期会话为进程内状态。不要直接运行多个 Uvicorn worker 或多个副本。
- 如果要横向扩容，应将认证会话、限速、聊天状态和任务协调迁移到共享数据库或 Redis。
- 聊天会话元数据会持久化，但短期和中期消息不会在服务重启后恢复；个人长期记忆仍保存在 Chroma。
- 管理员上传的知识对所有受邀用户可检索；用户长期记忆按用户 ID 过滤。
- `SameSite=Lax` 和 JSON 请求降低跨站请求风险，但生产环境仍必须使用 HTTPS、受控来源和可信反向代理。
- 应同时备份认证数据卷和 Chroma 数据卷，并定期测试恢复流程。
- 系统没有邮件发送、密码重置、邀请撤销按钮、审计日志和共享配额服务。

## 开发验证

```bash
python3 -m py_compile main.py app/*.py app/memory/*.py app/retriever/*.py
node --check web/app.js
git diff --check
```

真实问答回归还需要有效的百炼、DeepSeek 和 Embedding 配置。
