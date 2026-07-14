"""
RAG 主流程编排：串联所有模块完成一次完整的知识问答
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, List, Tuple

from app.config import get_settings
from app.input_preprocessor import InputPreprocessor
from app.memory.memory_manager import MemoryManager
from app.query_rewrite import QueryRewriter
from app.retriever.hybrid_retriever import HybridRetriever
from app.reranker import Reranker
from app.prompt_builder import PromptBuilder
from app.llm import LLMClient
from app.postprocessor import PostProcessor
from app.sensitive_words import get_default_sensitive_config

logger = logging.getLogger(__name__)
settings = get_settings


class RAGPipeline:
    """
    RAG 主流程：
    1. Query 改写 / 意图理解
    2. 多路召回（百炼远程 + 本地 Chroma）
    3. 重排 + 上下文压缩
    4. Prompt 工程
    5. 调用 LLM（DeepSeek Pro）
    6. 结果后处理
    7. 同步更新三层对话记忆
    """

    # 节流间隔：两次 session 过期扫描之间最小间隔（秒）
    _EVICT_THROTTLE_SECONDS = 60

    def __init__(self):
        """根据全局配置装配 RAG 各阶段及会话级内存状态。

        创建共享 LLM 客户端并注入改写、重排和预处理组件，同时初始化远程/本地
        检索器、Prompt 构建器和后处理器。会话记忆及锁按需创建，不在构造时预分配。

        Raises:
            Exception: 配置缺失、Chroma 初始化或检索组件构造失败。
        """
        s = settings()
        sensitive_config = get_default_sensitive_config(
            s.SENSITIVE_KEYWORDS_EXTRA_FILE
        )
        self.llm = LLMClient()
        self.rewriter = QueryRewriter(llm_client=self.llm)
        self.retriever = HybridRetriever()
        self.reranker = Reranker(
            top_k=s.RERANK_TOP_K,
            max_context_tokens=s.MAX_CONTEXT_TOKENS,
            llm_client=self.llm,
        )
        self.prompt_builder = PromptBuilder()
        self.postprocessor = PostProcessor(
            sensitive_words=sensitive_config["output_sensitive_words"]
        )
        self.preprocessor = InputPreprocessor(
            sensitive_config=sensitive_config,
            llm_client=self.llm,
        )

        # 会话级记忆管理（按 session_id 隔离）
        # value = (MemoryManager, last_access_timestamp)
        self._sessions: Dict[str, Tuple[MemoryManager, float]] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._last_evict_time: float = 0.0

    # ── 会话管理 ─────────────────────────────────────────────────────

    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        """构造同时隔离用户与聊天会话的进程内状态键。

        Args:
            user_id: 会话所属用户标识。
            session_id: 聊天会话标识。

        Returns:
            ``user_id:session_id`` 形式的复合键。
        """
        return f"{user_id}:{session_id}"

    def get_memory(
        self,
        session_id: str,
        user_id: str = "",
    ) -> MemoryManager:
        """获取或创建会话记忆，并按节流策略清理空闲会话。

        Args:
            session_id: 聊天会话标识。
            user_id: 所有者标识；与会话 ID 共同决定隔离键。

        Returns:
            复用或新建的 ``MemoryManager``。复用时会把最后访问时间续期到当前时间。

        Raises:
            Exception: 过期记忆归档或新记忆管理器的 Chroma 初始化失败。

        Notes:
            全量过期扫描最多每 60 秒触发一次；新会话会同时准备对应的异步锁。
        """
        now = time.time()
        session_key = self._session_key(user_id, session_id)

        # 节流：避免每次请求都全量扫描
        if now - self._last_evict_time > self._EVICT_THROTTLE_SECONDS:
            self._evict_expired(now)
            self._last_evict_time = now

        if session_key in self._sessions:
            memory, _ = self._sessions[session_key]
            self._sessions[session_key] = (memory, now)  # 续期
            return memory

        memory = MemoryManager(
            session_id=session_id,
            user_id=user_id,
            llm_client=self.llm,
        )
        self._sessions[session_key] = (memory, now)
        self._session_locks.setdefault(session_key, asyncio.Lock())
        logger.info(f"[Pipeline] 创建新会话记忆: {session_id}")
        return memory

    def _get_session_lock(self, user_id: str, session_id: str) -> asyncio.Lock:
        """返回指定用户会话的独占异步锁，不存在时原子式创建字典项。

        Args:
            user_id: 会话所属用户标识。
            session_id: 聊天会话标识。

        Returns:
            同一进程中该复合会话共享的 ``asyncio.Lock``。
        """
        key = self._session_key(user_id, session_id)
        return self._session_locks.setdefault(key, asyncio.Lock())

    def _evict_expired(self, now: float) -> None:
        """归档并释放在给定时间点已经超过空闲 TTL 的会话。

        Args:
            now: 本次扫描统一使用的 Unix 时间戳，避免循环内时间漂移。

        Notes:
            每个过期会话先归档短期和中期内容，再删除内存管理器及对应锁；
            没有过期项时不修改状态。

        Raises:
            Exception: 长期记忆批量归档失败；此时该会话不会被继续删除。
        """
        ttl = settings().SESSION_TTL_SECONDS
        expired_ids = [
            sid for sid, (_, last_access) in self._sessions.items()
            if now - last_access > ttl
        ]
        if not expired_ids:
            return

        for sid in expired_ids:
            memory, last_access = self._sessions[sid]
            self._archive_to_long_term(memory)
            del self._sessions[sid]
            self._session_locks.pop(sid, None)
            idle_minutes = int((now - last_access) / 60)
            logger.info(f"[Pipeline] 会话 {sid} 已过期（空闲 {idle_minutes} 分钟），记忆已归档并释放")

    def _archive_to_long_term(self, memory: MemoryManager) -> None:
        """把一个会话尚存的短期对话和全部中期摘要批量写入长期记忆。

        Args:
            memory: 即将过期或释放的会话记忆管理器。

        Notes:
            短期消息合并成一条 ``session_archive``，每个中期摘要形成一条
            ``summary_archive``；两者都使用该会话 ID。没有内容时不访问 Chroma。

        Raises:
            Exception: Chroma 批量写入失败。
        """
        items: List[dict] = []

        # ── 归档短期记忆中的全部对话 ──
        short_turns = memory.short_term.get_all_turns()
        if short_turns:
            conversation = "\n".join(
                f"{'用户' if t.role == 'user' else '助手'}：{t.content}"
                for t in short_turns
            )
            items.append({
                "content": f"[会话归档 {memory.session_id}]\n{conversation}",
                "category": "session_archive",
                "entities": [],
            })

        # ── 归档中期摘要（已经是 LLM 压缩后的，直接存入）─────
        for summary in memory.mid_term.summaries:
            items.append({
                "content": f"[会话摘要归档 {memory.session_id}]\n{summary.text}",
                "category": "summary_archive",
                "entities": [],
            })

        if items:
            memory.long_term.store_batch(items, session_id=memory.session_id)
            short_count = len(short_turns)
            mid_count = len(memory.mid_term.summaries)
            logger.info(
                f"[Pipeline] 会话 {memory.session_id} 归档完成："
                f"短期 {short_count} 条对话 + 中期 {mid_count} 条摘要 → 长期记忆"
            )

    def clear_session(
        self,
        session_id: str,
        user_id: str = "",
        include_long_term: bool = True,
    ) -> None:
        """清除指定用户会话的内存状态，并按需删除其长期记录。

        Args:
            session_id: 需要清除的聊天会话标识。
            user_id: 会话所有者标识。
            include_long_term: 是否同时删除该用户、该会话的持久记忆。

        Notes:
            活跃会话会先调用 ``MemoryManager.clear_all`` 再移除管理器和锁；会话
            不在内存但要求删除长期记录时，会临时创建管理器执行所有权受限删除。

        Raises:
            Exception: Chroma 初始化或删除失败。
        """
        session_key = self._session_key(user_id, session_id)
        if session_key in self._sessions:
            self._sessions[session_key][0].clear_all(include_long_term=include_long_term)
            del self._sessions[session_key]
            self._session_locks.pop(session_key, None)
            logger.info(f"[Pipeline] 会话已清除: {session_id}")
        elif include_long_term:
            memory = MemoryManager(
                session_id=session_id,
                user_id=user_id,
                llm_client=self.llm,
            )
            memory.long_term.clear_session(session_id)

    async def aclose(self) -> None:
        """依次关闭预处理器自有资源、远程检索连接和共享 LLM 连接池。

        预处理器使用本 Pipeline 注入的共享 LLM，因此不会重复关闭它；各组件的关闭
        方法均允许在未初始化或已关闭时调用。
        """
        await self.preprocessor.aclose()
        await self.retriever.aclose()
        await self.llm.aclose()

    # ── 非流式问答 ───────────────────────────────────────────────────
    async def ask(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = "",
    ) -> dict:
        """串行执行一次非流式 RAG 问答并在成功后记录完整对话。

        Args:
            query: 用户原始问题；启用预处理时可能被指代消解结果替换。
            session_id: 聊天会话标识。
            user_id: 会话所有者标识，用于锁、内存和长期记忆隔离。

        Returns:
            含 ``answer``、``sources``、``session_id`` 和 ``trace`` 的字典。
            ``trace`` 记录改写查询、召回/重排数量和耗时。预处理拦截时立即返回
            无来源回答及 ``blocked`` 跟踪信息，不调用检索和生成，也不写入记忆。

        Raises:
            Exception: 记忆、改写、检索、重排、LLM、后处理或持久写入失败。
                本层不吞掉异常，由 HTTP 路由统一转换为服务错误。

        Notes:
            同一用户会话的整个流程由独占锁串行化；只有最终回答成功后才追加问答记录。
        """
        lock = self._get_session_lock(user_id, session_id)
        async with lock:
            start_time = time.time()
            memory = self.get_memory(session_id, user_id)

            if settings().ENABLE_INPUT_PREPROCESS:
                pre_result = await self.preprocessor.preprocess(
                    query,
                    history_messages=memory.short_term.get_window_as_messages(),
                )
                if pre_result.blocked:
                    return {
                        "answer": pre_result.reason,
                        "sources": [],
                        "session_id": session_id,
                        "trace": {"blocked": True, "reason": pre_result.reason},
                    }
                query = pre_result.processed_query

            # 当前问题不提前写入历史，避免在改写和最终 Prompt 中重复。
            memory_context = await asyncio.to_thread(memory.build_context, query)
            rewritten_queries = await self.rewriter.rewrite(
                query,
                memory_context["short_term_messages"],
                memory_context["mid_term_summary"],
            )
            retrieved_docs = await self.retriever.retrieve(
                rewritten_queries,
                top_k=settings().RETRIEVE_TOP_K,
            )
            reranked_docs = await self.reranker.rerank(query, retrieved_docs)
            messages = self.prompt_builder.build_messages(query, reranked_docs, memory_context)
            raw_answer = await self.llm.chat(messages)
            result = self.postprocessor.process(raw_answer, reranked_docs)

            await memory.record_exchange(
                query,
                result["answer"],
                sources=result["sources"],
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"[Pipeline] 会话={session_id}, 耗时={elapsed_ms}ms")
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "session_id": session_id,
                "trace": {
                    "rewritten_queries": rewritten_queries,
                    "retrieved_count": len(retrieved_docs),
                    "reranked_count": len(reranked_docs),
                    "elapsed_ms": elapsed_ms,
                },
            }

    # ── 流式问答 ─────────────────────────────────────────────────────
    async def ask_stream(
        self,
        query: str,
        session_id: str = "default",
        user_id: str = "",
    ) -> AsyncGenerator[dict, None]:
        """执行流式问答，并以结构化事件报告阶段、正文和最终结果。

        Args:
            query: 用户原始问题；启用预处理时可能替换为消解后的完整问题。
            session_id: 聊天会话标识。
            user_id: 会话所有者标识，用于锁与三层记忆隔离。

        Yields:
            ``{"type": ..., "data": ...}`` 事件：``status`` 报告处理阶段，
            ``chunk`` 分段输出完成后处理的回答，``blocked`` 报告输入拦截，
            ``done`` 携带最终回答/来源/跟踪信息，``error`` 携带可展示的失败消息。

        Notes:
            为保证脱敏和引用一致性，底层 LLM 增量会先完整缓冲和后处理，再按最多
            120 个字符切块。普通异常被记录并转成 ``error`` 事件；任务取消异常原样
            抛出。预处理拦截会依次产出 ``blocked`` 与 ``done``，且不写入记忆。
        """
        lock = self._get_session_lock(user_id, session_id)
        async with lock:
            start_time = time.time()
            memory = self.get_memory(session_id, user_id)
            full_answer = ""
            try:
                if settings().ENABLE_INPUT_PREPROCESS:
                    pre_result = await self.preprocessor.preprocess(
                        query,
                        history_messages=memory.short_term.get_window_as_messages(),
                    )
                    if pre_result.blocked:
                        yield {"type": "blocked", "data": pre_result.reason}
                        yield {
                            "type": "done",
                            "data": {
                                "answer": pre_result.reason,
                                "sources": [],
                                "session_id": session_id,
                                "trace": {"blocked": True},
                            },
                        }
                        return
                    query = pre_result.processed_query

                memory_context = await asyncio.to_thread(memory.build_context, query)
                yield {"type": "status", "data": "正在理解您的问题..."}
                rewritten_queries = await self.rewriter.rewrite(
                    query,
                    memory_context["short_term_messages"],
                    memory_context["mid_term_summary"],
                )
                yield {"type": "status", "data": "正在检索知识库..."}
                retrieved_docs = await self.retriever.retrieve(
                    rewritten_queries,
                    top_k=settings().RETRIEVE_TOP_K,
                )
                reranked_docs = await self.reranker.rerank(query, retrieved_docs)
                messages = self.prompt_builder.build_messages(query, reranked_docs, memory_context)
                yield {"type": "status", "data": "正在生成并校验回答..."}

                # 严格模式：原始模型输出先完整后处理，再发送给客户端。
                async for chunk in self.llm.chat_stream(messages):
                    full_answer += chunk
                result = self.postprocessor.process(full_answer, reranked_docs)
                for offset in range(0, len(result["answer"]), 120):
                    yield {"type": "chunk", "data": result["answer"][offset:offset + 120]}

                await memory.record_exchange(
                    query,
                    result["answer"],
                    sources=result["sources"],
                )
                elapsed_ms = int((time.time() - start_time) * 1000)
                yield {
                    "type": "done",
                    "data": {
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "session_id": session_id,
                        "trace": {
                            "rewritten_queries": rewritten_queries,
                            "retrieved_count": len(retrieved_docs),
                            "reranked_count": len(reranked_docs),
                            "elapsed_ms": elapsed_ms,
                        },
                    },
                }
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(f"[Pipeline-流式] 会话={session_id}, 处理失败")
                yield {"type": "error", "data": "问答服务暂时不可用，请稍后重试。"}
