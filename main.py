"""企业知识问答服务：邀请制认证、管理端与 RAG API。"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.auth import (
    AuthError,
    AuthStore,
    AuthUser,
    LoginAttemptLimiter,
    LoginRateLimitError,
)
from app.config import get_settings
from app.pipeline import RAGPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings

pipeline: RAGPipeline = None  # type: ignore[assignment]
auth_store: AuthStore = None  # type: ignore[assignment]
login_limiter = LoginAttemptLimiter()
WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, auth_store
    s = settings()
    logger.info("正在初始化认证存储与 RAG Pipeline...")
    auth_store = AuthStore(
        db_path=s.AUTH_DB_PATH,
        session_ttl_seconds=s.AUTH_SESSION_TTL_SECONDS,
        invitation_ttl_seconds=s.INVITATION_TTL_SECONDS,
        max_active_chat_sessions=s.MAX_ACTIVE_CHAT_SESSIONS,
    )
    if auth_store.bootstrap_admin(s.BOOTSTRAP_ADMIN_EMAIL, s.BOOTSTRAP_ADMIN_PASSWORD):
        logger.info("已创建初始管理员账号")
    elif auth_store.user_count() == 0:
        logger.warning("系统尚无管理员，请配置 BOOTSTRAP_ADMIN_EMAIL 和 BOOTSTRAP_ADMIN_PASSWORD")
    pipeline = RAGPipeline()
    app.state.auth_store = auth_store
    app.state.pipeline = pipeline
    logger.info("服务初始化完成")
    try:
        yield
    finally:
        await pipeline.aclose()
        auth_store.close()
        logger.info("服务关闭")


app = FastAPI(
    title="企业知识问答服务",
    description="邀请制企业知识库问答与文档管理",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
        "script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    )
    private_paths = ("/api/", "/chat", "/session/", "/documents/")
    if request.url.path == "/invite" or request.url.path.startswith(private_paths):
        response.headers["Cache-Control"] = "no-store"
    return response


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)


class AcceptInvitationRequest(BaseModel):
    token: str = Field(..., min_length=20, max_length=200)
    display_name: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=10, max_length=128)


class CreateInvitationRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    role: Literal["user", "admin"] = "user"


class UserStatusRequest(BaseModel):
    active: bool


class CreateChatSessionRequest(BaseModel):
    title: str = Field("新对话", max_length=80)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, max_length=64)
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str
    trace: dict


class UploadDocsRequest(BaseModel):
    documents: list[str] = Field(..., min_length=1)
    source: str = Field("manual", min_length=1, max_length=120)


class SessionManageRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)


def _set_auth_cookie(response: Response, token: str) -> None:
    s = settings()
    response.set_cookie(
        key=s.AUTH_COOKIE_NAME,
        value=token,
        max_age=s.AUTH_SESSION_TTL_SECONDS,
        httponly=True,
        secure=s.AUTH_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _delete_auth_cookie(response: Response) -> None:
    response.delete_cookie(settings().AUTH_COOKIE_NAME, path="/")


def current_user(request: Request) -> AuthUser:
    token = request.cookies.get(settings().AUTH_COOKIE_NAME, "")
    user = request.app.state.auth_store.resolve_session(token)
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录。")
    return user


def admin_user(user: AuthUser = Depends(current_user)) -> AuthUser:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限。")
    return user


def _auth_error(error: AuthError, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(error))


def _ensure_chat_session(user: AuthUser, session_id: Optional[str]) -> dict:
    try:
        if session_id:
            return auth_store.require_chat_session(user, session_id)
        return auth_store.create_chat_session(user)
    except AuthError as error:
        raise _auth_error(error, 404 if session_id else 400) from error


def _document_chunks(documents: list[str], source: str) -> tuple[list[str], list[dict]]:
    s = settings()
    if len(documents) > s.MAX_UPLOAD_DOCUMENTS:
        raise HTTPException(status_code=413, detail="单次上传文档数量超过限制。")
    if s.DOCUMENT_CHUNK_OVERLAP >= s.DOCUMENT_CHUNK_SIZE:
        raise HTTPException(status_code=500, detail="文档分块配置无效。")
    chunks: list[str] = []
    metadatas: list[dict] = []
    for document_index, document in enumerate(documents):
        content = document.strip()
        if not content:
            continue
        if len(content) > s.MAX_UPLOAD_CHARS_PER_DOCUMENT:
            raise HTTPException(status_code=413, detail="单个文档内容超过限制。")
        start = 0
        chunk_index = 0
        while start < len(content):
            end = min(len(content), start + s.DOCUMENT_CHUNK_SIZE)
            chunks.append(content[start:end])
            metadatas.append(
                {
                    "source": source,
                    "document_index": document_index,
                    "chunk_index": chunk_index,
                }
            )
            if end == len(content):
                break
            start = end - s.DOCUMENT_CHUNK_OVERLAP
            chunk_index += 1
    if not chunks:
        raise HTTPException(status_code=400, detail="没有可入库的文档内容。")
    return chunks, metadatas


@app.exception_handler(AuthError)
async def handle_auth_error(_: Request, error: AuthError):
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.get("/live")
async def live():
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    try:
        count = request.app.state.pipeline.retriever.local.count
        request.app.state.auth_store.user_count()
        return {"status": "ready", "local_documents": count}
    except Exception:
        raise HTTPException(status_code=503, detail="服务尚未就绪。")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "enterprise-knowledge-qa"}


@app.post("/api/auth/login")
async def login(payload: LoginRequest, request: Request):
    client_host = request.client.host if request.client else "unknown"
    attempt_key = f"{client_host}:{payload.email.strip().casefold()}"
    try:
        login_limiter.check(attempt_key)
        user, token = auth_store.authenticate(payload.email, payload.password)
    except LoginRateLimitError as error:
        raise _auth_error(error, 429) from error
    except AuthError as error:
        login_limiter.record_failure(attempt_key)
        raise _auth_error(error, 401) from error
    login_limiter.reset(attempt_key)
    response = JSONResponse({"user": user.as_dict()})
    _set_auth_cookie(response, token)
    return response


@app.post("/api/auth/logout")
async def logout(request: Request):
    auth_store.logout(request.cookies.get(settings().AUTH_COOKIE_NAME, ""))
    response = JSONResponse({"message": "已退出登录。"})
    _delete_auth_cookie(response)
    return response


@app.get("/api/auth/me")
async def me(user: AuthUser = Depends(current_user)):
    return {"user": user.as_dict()}


@app.post("/api/auth/invitations/accept")
async def accept_invitation(payload: AcceptInvitationRequest):
    try:
        user, session_token = auth_store.accept_invitation(
            payload.token,
            payload.display_name,
            payload.password,
        )
    except AuthError as error:
        raise _auth_error(error) from error
    response = JSONResponse({"user": user.as_dict()})
    _set_auth_cookie(response, session_token)
    return response


@app.get("/api/admin/invitations")
async def list_invitations(user: AuthUser = Depends(admin_user)):
    return {"invitations": auth_store.list_invitations()}


@app.post("/api/admin/invitations")
async def create_invitation(
    payload: CreateInvitationRequest,
    user: AuthUser = Depends(admin_user),
):
    try:
        invitation, token = auth_store.create_invitation(user, payload.email, payload.role)
    except AuthError as error:
        raise _auth_error(error) from error
    invitation["invite_url"] = f"/invite?token={token}"
    return invitation


@app.get("/api/admin/users")
async def list_users(user: AuthUser = Depends(admin_user)):
    return {"users": auth_store.list_users()}


@app.patch("/api/admin/users/{user_id}")
async def update_user_status(
    user_id: str,
    payload: UserStatusRequest,
    user: AuthUser = Depends(admin_user),
):
    if user_id == user.id and not payload.active:
        raise HTTPException(status_code=400, detail="不能停用当前管理员账号。")
    try:
        auth_store.set_user_active(user_id, payload.active)
    except AuthError as error:
        raise _auth_error(error, 404) from error
    return {"message": "用户状态已更新。"}


@app.get("/api/sessions")
async def list_chat_sessions(user: AuthUser = Depends(current_user)):
    return {"sessions": auth_store.list_chat_sessions(user)}


@app.post("/api/sessions")
async def create_chat_session(
    payload: CreateChatSessionRequest,
    user: AuthUser = Depends(current_user),
):
    try:
        return auth_store.create_chat_session(user, payload.title)
    except AuthError as error:
        raise _auth_error(error) from error


@app.get("/api/sessions/{session_id}/messages")
async def get_chat_messages(session_id: str, user: AuthUser = Depends(current_user)):
    try:
        auth_store.require_chat_session(user, session_id)
    except AuthError as error:
        raise _auth_error(error, 404) from error
    memory = pipeline.get_memory(session_id, user.id)
    return {
        "messages": [
            {"role": turn.role, "content": turn.content}
            for turn in memory.short_term.get_all_turns()
        ]
    }


@app.delete("/api/sessions/{session_id}")
async def delete_chat_session(session_id: str, user: AuthUser = Depends(current_user)):
    try:
        auth_store.require_chat_session(user, session_id)
        pipeline.clear_session(session_id, user.id, include_long_term=True)
        auth_store.delete_chat_session(user, session_id)
    except AuthError as error:
        raise _auth_error(error, 404) from error
    return {"message": "对话已删除。"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, user: AuthUser = Depends(current_user)):
    chat_session = _ensure_chat_session(user, payload.session_id)
    try:
        result = await pipeline.ask(
            payload.query,
            session_id=chat_session["id"],
            user_id=user.id,
        )
        title = payload.query[:40] if chat_session["title"] == "新对话" else None
        auth_store.touch_chat_session(user, chat_session["id"], title=title)
        return ChatResponse(**result)
    except Exception:
        logger.exception("非流式问答失败")
        raise HTTPException(status_code=500, detail="问答服务暂时不可用，请稍后重试。")


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest, user: AuthUser = Depends(current_user)):
    chat_session = _ensure_chat_session(user, payload.session_id)
    title = payload.query[:40] if chat_session["title"] == "新对话" else None
    auth_store.touch_chat_session(user, chat_session["id"], title=title)

    async def event_generator():
        try:
            async for event in pipeline.ask_stream(
                payload.query,
                session_id=chat_session["id"],
                user_id=user.id,
            ):
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("流式问答失败")
            yield {
                "event": "error",
                "data": json.dumps("问答服务暂时不可用，请稍后重试。", ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@app.post("/documents/upload")
async def upload_documents(
    payload: UploadDocsRequest,
    _: AuthUser = Depends(admin_user),
):
    chunks, metadatas = _document_chunks(payload.documents, payload.source.strip())
    try:
        count = await asyncio.to_thread(
            pipeline.retriever.local.add_documents,
            chunks,
            None,
            metadatas,
        )
        return {
            "message": f"成功写入 {count} 个文档分块。",
            "chunks": count,
            "total": pipeline.retriever.local.count,
        }
    except Exception:
        logger.exception("文档上传失败")
        raise HTTPException(status_code=500, detail="文档上传失败，请检查模型与数据配置。")


@app.post("/session/clear")
async def clear_session(
    payload: SessionManageRequest,
    user: AuthUser = Depends(current_user),
):
    try:
        auth_store.require_chat_session(user, payload.session_id)
        pipeline.clear_session(payload.session_id, user.id, include_long_term=True)
        auth_store.delete_chat_session(user, payload.session_id)
    except AuthError as error:
        raise _auth_error(error, 404) from error
    return {"message": "会话记忆已清除。"}


@app.get("/session/{session_id}/memory")
async def get_session_memory(session_id: str, user: AuthUser = Depends(current_user)):
    try:
        auth_store.require_chat_session(user, session_id)
    except AuthError as error:
        raise _auth_error(error, 404) from error
    memory = pipeline.get_memory(session_id, user.id)
    return {
        "session_id": session_id,
        "short_term": {"message_count": memory.short_term.total_turns},
        "mid_term": {"summary_count": len(memory.mid_term.summaries)},
        "long_term": {"total_records": memory.long_term.count},
    }


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
async def frontend_index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/invite", include_in_schema=False)
async def frontend_invite():
    return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import os

    s = settings()
    is_debugging = "PYDEVD_LOAD_VALUES_ASYNC" in os.environ or "pydevd" in sys.modules
    uvicorn.run(
        "main:app",
        host=s.HOST,
        port=s.PORT,
        reload=not is_debugging,
        log_level="info",
    )
