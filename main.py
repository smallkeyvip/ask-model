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
    """在 FastAPI 生命周期内初始化并释放认证存储与 RAG Pipeline。

    Args:
        app: 当前 FastAPI 应用实例，用于暴露已初始化的共享组件。

    Yields:
        初始化完成后的控制权；应用开始接收请求前，认证数据库、可选初始管理员和
        RAG Pipeline 均已准备并写入 ``app.state``。

    Raises:
        Exception: 配置、认证数据库或 Pipeline 初始化失败时阻止服务启动；关闭阶段
            的资源释放异常也会向服务器生命周期管理器传播。

    Notes:
        退出时依次关闭 Pipeline 网络资源和 SQLite 连接，不论服务正常停止还是
        生命周期内部抛出异常。
    """
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
    """为每个 HTTP 响应附加浏览器安全策略，并禁止私有页面缓存。

    Args:
        request: 当前入站请求，用于判断路径是否包含私有数据。
        call_next: FastAPI 提供的下游 ASGI 请求处理函数。

    Returns:
        下游生成的原响应对象，原地增加内容嗅探、点击劫持、来源、权限和 CSP
        响应头；邀请页及 API、聊天、会话、文档路径额外设置 ``no-store``。

    Raises:
        Exception: 下游请求处理异常原样传播，不在中间件中转换。
    """
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
    """把原始登录令牌设置为应用配置的受保护会话 Cookie。

    Args:
        response: 需要附加 ``Set-Cookie`` 头的响应对象。
        token: 认证存储刚创建、未持久化明文的原始会话令牌。

    Notes:
        Cookie 始终为 HttpOnly、SameSite=Lax、全站路径；Secure 与有效期由环境配置。
    """
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
    """在响应中删除应用配置的全站认证 Cookie。

    Args:
        response: 需要附加过期 Cookie 头的响应对象。
    """
    response.delete_cookie(settings().AUTH_COOKIE_NAME, path="/")


def current_user(request: Request) -> AuthUser:
    """FastAPI 依赖：把请求 Cookie 解析为当前启用用户。

    Args:
        request: 当前请求，包含 Cookie 和应用级 ``AuthStore``。

    Returns:
        有效登录会话对应的 ``AuthUser``；解析过程会更新会话最后访问时间。

    Raises:
        HTTPException: Cookie 缺失、令牌无效/过期或账号停用时返回 HTTP 401。
    """
    token = request.cookies.get(settings().AUTH_COOKIE_NAME, "")
    user = request.app.state.auth_store.resolve_session(token)
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录。")
    return user


def admin_user(user: AuthUser = Depends(current_user)) -> AuthUser:
    """FastAPI 依赖：在已登录基础上强制要求管理员角色。

    Args:
        user: :func:`current_user` 已验证的当前用户。

    Returns:
        原 ``AuthUser``，供管理员路由继续使用。

    Raises:
        HTTPException: 当前角色不是管理员时返回 HTTP 403。
    """
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限。")
    return user


def _auth_error(error: AuthError, status_code: int = 400) -> HTTPException:
    """把可公开的认证业务错误转换为指定状态的 FastAPI 异常。

    Args:
        error: 消息已设计为可安全返回客户端的 ``AuthError``。
        status_code: 需要映射的 HTTP 状态码，默认 400。

    Returns:
        以业务错误文本作为 ``detail`` 的 ``HTTPException``。
    """
    return HTTPException(status_code=status_code, detail=str(error))


def _ensure_chat_session(user: AuthUser, session_id: Optional[str]) -> dict:
    """获取用户已有会话，或在请求未指定 ID 时创建一个新会话。

    Args:
        user: 当前已认证用户，用于会话所有权和配额检查。
        session_id: 客户端指定的会话 ID；空值表示自动创建“新对话”。

    Returns:
        已验证或新建的聊天会话元数据字典。

    Raises:
        HTTPException: 指定会话不存在/无权访问时返回 404；自动创建因配额等业务
            原因失败时返回 400。
    """
    try:
        if session_id:
            return auth_store.require_chat_session(user, session_id)
        return auth_store.create_chat_session(user)
    except AuthError as error:
        raise _auth_error(error, 404 if session_id else 400) from error


def _document_chunks(documents: list[str], source: str) -> tuple[list[str], list[dict]]:
    """校验上传文本并按配置的大小与重叠量切分为可入库片段。

    Args:
        documents: 单次请求中的原始文本文档列表。
        source: 写入每个片段元数据的来源标签。

    Returns:
        等长的正文片段列表和元数据列表。元数据记录来源、原文档下标与片段下标；
        纯空白文档被跳过，相邻非末片段按配置字符数重叠。

    Raises:
        HTTPException: 文档数量或单篇字符数超限时返回 413；没有有效正文时返回
            400；重叠量大于等于片段大小的服务端配置错误返回 500。
    """
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
    """把未在路由内单独映射的认证业务错误转为 JSON 400 响应。

    Args:
        _: FastAPI 传入但本处理器不需要读取的请求对象。
        error: 消息可安全展示的认证业务异常。

    Returns:
        ``{"detail": ...}`` 结构的 HTTP 400 ``JSONResponse``。
    """
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.get("/live")
async def live():
    """返回不依赖外部组件的进程存活探针响应。

    Returns:
        固定的 ``{"status": "ok"}``，用于容器判断 ASGI 进程是否可响应。
    """
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    """检查本地知识库与认证数据库是否可以响应基础查询。

    Args:
        request: 当前请求，用于从 ``app.state`` 访问初始化后的组件。

    Returns:
        就绪状态及本地知识库当前文档数。

    Raises:
        HTTPException: 任一组件未初始化或查询失败时返回 HTTP 503，并隐藏内部细节。
    """
    try:
        count = request.app.state.pipeline.retriever.local.count
        request.app.state.auth_store.user_count()
        return {"status": "ready", "local_documents": count}
    except Exception:
        raise HTTPException(status_code=503, detail="服务尚未就绪。")


@app.get("/health")
async def health():
    """返回兼容旧部署检查的服务健康标识。

    Returns:
        固定状态和服务名；该端点不验证数据库或模型依赖。
    """
    return {"status": "ok", "service": "enterprise-knowledge-qa"}


@app.post("/api/auth/login")
async def login(payload: LoginRequest, request: Request):
    """验证邮箱密码、应用失败窗口限速并设置登录 Cookie。

    Args:
        payload: 经 Pydantic 校验的邮箱和密码请求体。
        request: 当前请求，用于读取客户端地址并构造“IP:邮箱”限速键。

    Returns:
        含公开用户资料的 JSON 响应，并附加 HttpOnly 会话 Cookie。

    Raises:
        HTTPException: 达到失败上限时返回 429；认证业务失败时记录一次失败并返回
            401。成功后会清除该限速键的历史失败记录。
    """
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
    """撤销当前数据库会话并要求浏览器删除认证 Cookie。

    Args:
        request: 当前请求，用于读取可能为空的会话 Cookie。

    Returns:
        退出成功消息的 JSON 响应。无 Cookie 或令牌已失效时仍幂等成功。
    """
    auth_store.logout(request.cookies.get(settings().AUTH_COOKIE_NAME, ""))
    response = JSONResponse({"message": "已退出登录。"})
    _delete_auth_cookie(response)
    return response


@app.get("/api/auth/me")
async def me(user: AuthUser = Depends(current_user)):
    """返回当前有效登录用户的公开资料。

    Args:
        user: 由认证依赖解析的当前用户。

    Returns:
        ``{"user": ...}`` 响应；无有效会话时依赖会先返回 HTTP 401。
    """
    return {"user": user.as_dict()}


@app.post("/api/auth/invitations/accept")
async def accept_invitation(payload: AcceptInvitationRequest):
    """消费一次性邀请、创建账号并自动建立登录会话。

    Args:
        payload: 邀请令牌、显示名和新密码组成的校验后请求体。

    Returns:
        含新用户公开资料的 JSON 响应，并附加新的认证 Cookie。

    Raises:
        HTTPException: 邀请或用户资料业务校验失败时返回 HTTP 400。
    """
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
    """列出最近邀请及实时计算的状态，仅允许管理员访问。

    Args:
        user: 管理员依赖注入值；参数本身无需用于查询。

    Returns:
        ``{"invitations": [...]}``，不包含任何令牌摘要或原始令牌。
    """
    return {"invitations": auth_store.list_invitations()}


@app.post("/api/admin/invitations")
async def create_invitation(
    payload: CreateInvitationRequest,
    user: AuthUser = Depends(admin_user),
):
    """为未注册邮箱创建管理员授权的一次性邀请链接。

    Args:
        payload: 被邀请邮箱和目标角色。
        user: 经管理员依赖验证的邀请创建者。

    Returns:
        邀请元数据以及仅本次响应可见的相对 ``invite_url``。

    Raises:
        HTTPException: 角色、邮箱、权限或重复注册等业务错误返回 HTTP 400。
    """
    try:
        invitation, token = auth_store.create_invitation(user, payload.email, payload.role)
    except AuthError as error:
        raise _auth_error(error) from error
    invitation["invite_url"] = f"/invite?token={token}"
    return invitation


@app.get("/api/admin/users")
async def list_users(user: AuthUser = Depends(admin_user)):
    """列出用户公开资料，仅允许管理员访问。

    Args:
        user: 管理员依赖注入值；参数本身无需用于查询。

    Returns:
        ``{"users": [...]}``，不含密码或登录会话数据。
    """
    return {"users": auth_store.list_users()}


@app.patch("/api/admin/users/{user_id}")
async def update_user_status(
    user_id: str,
    payload: UserStatusRequest,
    user: AuthUser = Depends(admin_user),
):
    """启用或停用指定用户，并阻止管理员停用自己。

    Args:
        user_id: URL 路径中的目标用户 ID。
        payload: 包含目标 ``active`` 状态的请求体。
        user: 执行操作的当前管理员。

    Returns:
        状态更新成功消息；停用账号会在存储层立即撤销其所有登录会话。

    Raises:
        HTTPException: 尝试停用自己时返回 400；目标用户不存在时返回 404。
    """
    if user_id == user.id and not payload.active:
        raise HTTPException(status_code=400, detail="不能停用当前管理员账号。")
    try:
        auth_store.set_user_active(user_id, payload.active)
    except AuthError as error:
        raise _auth_error(error, 404) from error
    return {"message": "用户状态已更新。"}


@app.get("/api/sessions")
async def list_chat_sessions(user: AuthUser = Depends(current_user)):
    """列出当前用户自己的聊天会话，按最近活动时间排序。

    Args:
        user: 当前已认证用户。

    Returns:
        ``{"sessions": [...]}`` 会话元数据响应。
    """
    return {"sessions": auth_store.list_chat_sessions(user)}


@app.post("/api/sessions")
async def create_chat_session(
    payload: CreateChatSessionRequest,
    user: AuthUser = Depends(current_user),
):
    """在当前用户配额内创建一个独立聊天会话。

    Args:
        payload: 包含候选标题的校验后请求体。
        user: 当前已认证用户，作为新会话所有者。

    Returns:
        新会话 ID、规范化标题及时间戳。

    Raises:
        HTTPException: 达到活跃会话上限等业务错误返回 HTTP 400。
    """
    try:
        return auth_store.create_chat_session(user, payload.title)
    except AuthError as error:
        raise _auth_error(error) from error


@app.get("/api/sessions/{session_id}/messages")
async def get_chat_messages(session_id: str, user: AuthUser = Depends(current_user)):
    """返回当前用户指定会话仍驻留进程内的短期消息。

    Args:
        session_id: 需要读取的聊天会话 ID。
        user: 当前已认证用户，用于所有权校验和内存隔离。

    Returns:
        仅含角色和正文的 ``messages`` 列表；来源元数据不在该响应中返回。

    Raises:
        HTTPException: 会话不存在或不属于当前用户时返回 HTTP 404。

    Notes:
        SQLite 只保存会话元数据，消息来自当前进程短期记忆，进程重启后可能为空。
    """
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
    """删除当前用户的会话元数据及其全部三层记忆。

    Args:
        session_id: 需要删除的聊天会话 ID。
        user: 当前已认证用户，用于所有权和记忆隔离。

    Returns:
        删除成功消息。

    Raises:
        HTTPException: 会话不存在或不属于当前用户时返回 HTTP 404。
        Exception: Pipeline/Chroma 清理失败时原样传播，由全局服务器错误处理接管。
    """
    try:
        auth_store.require_chat_session(user, session_id)
        pipeline.clear_session(session_id, user.id, include_long_term=True)
        auth_store.delete_chat_session(user, session_id)
    except AuthError as error:
        raise _auth_error(error, 404) from error
    return {"message": "对话已删除。"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, user: AuthUser = Depends(current_user)):
    """执行一次非流式 RAG 问答并更新聊天会话活动信息。

    Args:
        payload: 用户问题、可选会话 ID及兼容保留的 ``stream`` 标志。
        user: 当前已认证用户。

    Returns:
        经 ``ChatResponse`` 校验的回答、来源、会话 ID 和流程跟踪信息。

    Raises:
        HTTPException: 指定会话无权访问时返回 404，自动创建失败时返回 400；
            Pipeline 或会话更新时间的其他异常会记录内部堆栈并统一返回 HTTP 500。

    Notes:
        新会话的默认标题会在首次问题成功处理后更新为问题前 40 个字符。
    """
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
    """建立 SSE 流式问答响应，并在开始生成前更新会话活动时间。

    Args:
        payload: 用户问题与可选聊天会话 ID。
        user: 当前已认证用户。

    Returns:
        包装嵌套事件生成器的 ``EventSourceResponse``。Pipeline 事件类型映射为 SSE
        ``event``，数据以不转义中文的 JSON 编码。

    Raises:
        HTTPException: 在响应建立前，会话校验/创建失败会映射为 404 或 400；
            会话活动时间更新的异常也会在建立 SSE 前传播。
    """
    chat_session = _ensure_chat_session(user, payload.session_id)
    title = payload.query[:40] if chat_session["title"] == "新对话" else None
    auth_store.touch_chat_session(user, chat_session["id"], title=title)

    async def event_generator():
        """把 Pipeline 异步事件转换为 SSE 字典，并隔离生成期错误。

        Yields:
            含 ``event`` 与 JSON ``data`` 的 SSE 事件字典。

        Notes:
            客户端断开产生的 ``CancelledError`` 原样抛出以终止工作；其他异常记录
            服务端堆栈并产生一个不泄露内部信息的 ``error`` 事件。
        """
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
    """校验管理员上传文本、分块并在线程池写入本地向量库。

    Args:
        payload: 文档正文列表和来源标签。
        _: 管理员依赖注入值，仅用于授权，不在函数体内读取。

    Returns:
        成功写入的片段数、成功消息和 collection 当前总文档数。

    Raises:
        HTTPException: 分块前的数量、大小、内容或配置问题返回 400/413/500；
            嵌入或 Chroma 写入失败会记录堆栈并统一返回 HTTP 500。
    """
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
    """通过兼容端点删除当前用户会话及其全部记忆。

    Args:
        payload: 包含目标会话 ID 的请求体。
        user: 当前已认证用户，用于所有权校验。

    Returns:
        会话记忆清除成功消息。

    Raises:
        HTTPException: 会话不存在或无权访问时返回 HTTP 404。
        Exception: Pipeline/Chroma 清理失败时原样传播。
    """
    try:
        auth_store.require_chat_session(user, payload.session_id)
        pipeline.clear_session(payload.session_id, user.id, include_long_term=True)
        auth_store.delete_chat_session(user, payload.session_id)
    except AuthError as error:
        raise _auth_error(error, 404) from error
    return {"message": "会话记忆已清除。"}


@app.get("/session/{session_id}/memory")
async def get_session_memory(session_id: str, user: AuthUser = Depends(current_user)):
    """返回当前用户会话三层记忆的数量摘要。

    Args:
        session_id: 需要检查的聊天会话 ID。
        user: 当前已认证用户，用于所有权与记忆隔离。

    Returns:
        会话 ID、短期消息数、中期摘要数和当前用户长期记录总数。

    Raises:
        HTTPException: 会话不存在或无权访问时返回 HTTP 404。
        Exception: 记忆管理器或 Chroma 计数查询失败。
    """
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
    """返回单页应用入口文件，不把静态页面路由加入 OpenAPI。

    Returns:
        指向 ``web/index.html`` 的 ``FileResponse``。
    """
    return FileResponse(WEB_DIR / "index.html")


@app.get("/invite", include_in_schema=False)
async def frontend_invite():
    """为邀请链接返回同一单页应用入口，由前端根据路径显示激活表单。

    Returns:
        指向 ``web/index.html`` 的 ``FileResponse``。
    """
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
