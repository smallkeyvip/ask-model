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
    EMBEDDING_MODEL: str = "/Users/enmonster/ai/model/shibing624-text2vec-base-chinese"

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

    # ── 服务 ────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8010


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
