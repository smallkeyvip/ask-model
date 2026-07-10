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
        s = settings()
        self.llm = LLMClient()
        self.rewriter = QueryRewriter(llm_client=self.llm)
        self.retriever = HybridRetriever()
        self.reranker = Reranker(
            top_k=s.RERANK_TOP_K,
            max_context_tokens=s.MAX_CONTEXT_TOKENS,
            llm_client=self.llm,
        )
        self.prompt_builder = PromptBuilder()
        self.postprocessor = PostProcessor()
        self.preprocessor = InputPreprocessor(
            business_keywords=s.BUSINESS_DOMAIN_KEYWORDS or None,
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
        return f"{user_id}:{session_id}"

    def get_memory(
        self,
        session_id: str,
        user_id: str = "",
    ) -> MemoryManager:
        """获取或创建会话级记忆管理器，内部自动清理过期会话"""
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
        key = self._session_key(user_id, session_id)
        return self._session_locks.setdefault(key, asyncio.Lock())

    def _evict_expired(self, now: float) -> None:
        """扫描并清理过期会话，过期前将记忆归档到长期存储"""
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
        """将短期对话和中期摘要归档到长期 Chroma 记忆"""
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
        """清除指定会话的短期/中期记忆并释放会话"""
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
        """释放 Pipeline 持有的异步网络资源。"""
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
        """
        完整的 RAG 问答流程（非流式）。

        返回:
            {
                "answer": str,
                "sources": list,
                "session_id": str,
                "trace": {
                    "rewritten_queries": list,
                    "retrieved_count": int,
                    "reranked_count": int,
                    "elapsed_ms": int,
                }
            }
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
        """
        流式 RAG 问答：检索阶段非流式，LLM 生成阶段流式输出。
        每个 yield: {"type": "chunk|done|error", "data": ...}
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
