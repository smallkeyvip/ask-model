"""
全局配置管理 - 通过 pydantic-settings 从环境变量/.env 加载
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 阿里百炼知识库 ──────────────────────────────────────────────
    BAILIAN_API_URL: str
    BAILIAN_API_KEY: str
    BAILIAN_AGENT_ID: str

    # ── DeepSeek LLM ────────────────────────────────────────────────
    DEEPSEEK_API_URL: str
    DEEPSEEK_API_KEY: str
    DEEPSEEK_MODEL: str

    # ── Chroma 本地向量库 ───────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION: str = "enterprise_knowledge"

    # ── Embedding ───────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "shibing624/text2vec-base-chinese"
    EMBEDDING_LOCAL_FILES_ONLY: bool = False

    # ── 记忆层参数 ──────────────────────────────────────────────────
    SHORT_TERM_WINDOW_SIZE: int = 6          # 短期窗口保留最近 N 轮
    MID_TERM_SUMMARY_THRESHOLD: int = 10     # 窗口溢出 N 轮后触发摘要
    MID_TERM_MAX_SUMMARY_TOKENS: int = 500   # 摘要最大 token
    LONG_TERM_TOP_K: int = 5                 # 长期记忆召回条数

    # ── 会话管理 ────────────────────────────────────────────────────
    SESSION_TTL_SECONDS: int = 3600          # 会话无活动超时秒数（过期后归档记忆并释放）

    # ── 检索 & 重排 ─────────────────────────────────────────────────
    RETRIEVE_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    MAX_CONTEXT_TOKENS: int = 3000

    # ── 输入预处理 ────────────────────────────────────────────────────
    ENABLE_INPUT_PREPROCESS: bool = True    # 是否启用输入预处理（敏感词+业务判断+指代消解）
    SENSITIVE_KEYWORDS_EXTRA_FILE: str = "./sensitive_words_extra.json"
    BUSINESS_DOMAIN_KEYWORDS: str = ""      # 业务领域关键词（逗号分隔，留空使用内置默认）

    # ── 认证与授权 ─────────────────────────────────────────────────────
    AUTH_DB_PATH: str = "./data/auth.db"
    AUTH_COOKIE_NAME: str = "ask_model_session"
    AUTH_SESSION_TTL_SECONDS: int = 604800
    INVITATION_TTL_SECONDS: int = 604800
    AUTH_COOKIE_SECURE: bool = False
    BOOTSTRAP_ADMIN_EMAIL: str = ""
    BOOTSTRAP_ADMIN_PASSWORD: str = ""

    # ── API 安全边界 ──────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = "http://localhost:8010,http://127.0.0.1:8010"
    MAX_ACTIVE_CHAT_SESSIONS: int = 20
    MAX_UPLOAD_DOCUMENTS: int = 20
    MAX_UPLOAD_CHARS_PER_DOCUMENT: int = 100000
    DOCUMENT_CHUNK_SIZE: int = 1200
    DOCUMENT_CHUNK_OVERLAP: int = 150

    # ── 服务 ────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8010

    @property
    def allowed_origins(self) -> list[str]:
        """将逗号分隔的 CORS 配置解析为可直接传给 FastAPI 的来源列表。

        Returns:
            去除首尾空白并丢弃空项后的来源字符串列表；配置为空时返回空列表。
        """
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """加载并缓存当前进程使用的全局配置实例。

    首次调用时由 ``pydantic-settings`` 合并环境变量与 ``.env``；后续调用返回
    同一个 ``Settings`` 对象，避免重复解析配置。

    Returns:
        当前进程共享的配置实例。

    Raises:
        pydantic.ValidationError: 必填配置缺失或环境变量无法转换为声明类型。
    """
    return Settings()  # type: ignore[call-arg]
