"""
企业知识问答服务 - FastAPI 入口
支持非流式问答、SSE 流式问答、会话管理、本地知识库文档上传
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.pipeline import RAGPipeline

# ── 日志配置 ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 全局实例 ─────────────────────────────────────────────────────────
pipeline: RAGPipeline = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("正在初始化 RAG Pipeline...")
    pipeline = RAGPipeline()
    logger.info("RAG Pipeline 初始化完成 ✓")
    try:
        yield
    finally:
        await pipeline.aclose()
        logger.info("服务关闭")


app = FastAPI(
    title="企业知识问答服务",
    description="基于 RAG 的企业级知识问答系统，支持阿里百炼知识库 + 本地向量库多路召回",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    stream: bool = Field(False, description="是否流式返回")


class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str
    trace: dict


class UploadDocsRequest(BaseModel):
    documents: list[str] = Field(..., description="文档内容列表")
    source: str = Field("manual", description="文档来源标识")


class SessionManageRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")


# ── API 路由 ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "enterprise-knowledge-qa"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """非流式问答"""
    session_id = req.session_id or str(uuid.uuid4())
    try:
        result = await pipeline.ask(req.query, session_id=session_id)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"问答异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答服务异常: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式问答"""
    session_id = req.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            async for event in pipeline.ask_stream(req.query, session_id=session_id):
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }
        except Exception as e:
            logger.error(f"流式问答异常: {e}", exc_info=True)
            yield {"event": "error", "data": json.dumps({"error": str(e)}, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@app.post("/documents/upload")
async def upload_documents(req: UploadDocsRequest):
    """上传文档到本地 Chroma 向量库"""
    try:
        metadatas = [{"source": req.source} for _ in req.documents]
        count = pipeline.retriever.local.add_documents(
            documents=req.documents,
            metadatas=metadatas,
        )
        return {"message": f"成功上传 {count} 条文档", "total": pipeline.retriever.local.count}
    except Exception as e:
        logger.error(f"文档上传异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@app.post("/session/clear")
async def clear_session(req: SessionManageRequest):
    """清除指定会话的对话记忆"""
    pipeline.clear_session(req.session_id)
    return {"message": f"会话 {req.session_id} 记忆已清除"}


@app.get("/session/{session_id}/memory")
async def get_session_memory(session_id: str):
    """查看会话的记忆状态"""
    memory = pipeline.get_memory(session_id)
    return {
        "session_id": session_id,
        "short_term": {
            "total_turns": memory.short_term.total_turns,
            "window_size": memory.short_term.window_size,
        },
        "mid_term": {
            "summary_count": len(memory.mid_term.summaries),
        },
        "long_term": {
            "total_records": memory.long_term.count,
        },
    }


# ── 启动 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    s = get_settings()
    # PyCharm 调试器下禁用 reload，避免 WatchFiles reloader 与 pydevd 冲突
    is_debugging = "PYDEVD_LOAD_VALUES_ASYNC" in os.environ or "pydevd" in sys.modules
    uvicorn.run(
        "main:app",
        host=s.HOST,
        port=s.PORT,
        reload=not is_debugging,
        log_level="info",
    )
