"""
记忆管理器：统一编排三层记忆的读写、溢出压缩、长期记忆提取
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.config import get_settings
from app.memory.short_term import ShortTermMemory
from app.memory.mid_term import MidTermMemory
from app.memory.long_term import LongTermMemory
from app.llm import LLMClient

logger = logging.getLogger(__name__)
settings = get_settings


class MemoryManager:
    """
    三层记忆管理器：
    1. 短期（窗口）：实时存储本轮问答，构建上下文
    2. 中期（摘要）：窗口溢出时自动压缩历史
    3. 长期（向量）：提取关键实体/结论存入 Chroma，跨会话召回
    """

    def __init__(
        self,
        session_id: str = "default",
        user_id: str = "",
        llm_client: Optional[LLMClient] = None,
    ):
        """为一个用户会话创建并绑定三层记忆组件。

        Args:
            session_id: 当前聊天会话标识，用于内存归属和长期记录元数据。
            user_id: 当前用户标识，作为长期记忆的强制所有权过滤条件。
            llm_client: 中期摘要使用的可选共享 LLM 客户端。
        """
        s = settings()
        self.session_id = session_id
        self.user_id = user_id
        self.short_term = ShortTermMemory(window_size=s.SHORT_TERM_WINDOW_SIZE)
        self.mid_term = MidTermMemory(
            max_summary_tokens=s.MID_TERM_MAX_SUMMARY_TOKENS,
            llm_client=llm_client,
        )
        self.long_term = LongTermMemory(
            top_k=s.LONG_TERM_TOP_K,
            user_id=user_id,
            session_id=session_id,
        )
        self._turn_counter = 0

    # ── 记录一轮问答 ─────────────────────────────────────────────────
    def record_user(self, query: str) -> None:
        """仅把用户问题追加到短期记忆。

        Args:
            query: 已完成预处理的用户问题。

        Notes:
            本方法不增加完整轮次计数，也不触发窗口裁剪，适用于分步记录流程。
        """
        self.short_term.add("user", query)

    async def record_assistant(self, answer: str, sources: Optional[List[dict]] = None) -> None:
        """记录助手回复，并处理由这条回复触发的短期窗口溢出。

        Args:
            answer: 已后处理、最终展示给用户的回答。
            sources: 可选的引用来源，保存在该条消息的元数据中。

        Raises:
            Exception: 长期记忆写入失败。中期摘要自身的 LLM 失败会被其组件降级为
                ``None``，不会由本方法抛出。
        """
        self.short_term.add("assistant", answer, sources=sources or [])
        self._turn_counter += 1

        # 异步执行：溢出压缩 + 长期记忆提取（不阻塞响应）
        await self._handle_overflow()

    async def record_exchange(
        self,
        query: str,
        answer: str,
        sources: Optional[List[dict]] = None,
    ) -> None:
        """连续追加用户问题和助手回答，再处理窗口溢出。

        Args:
            query: 已完成预处理的用户问题。
            answer: 已后处理的最终回答。
            sources: 可选的引用来源，附加到助手消息元数据。

        Raises:
            Exception: 长期记忆写入失败。调用方应在会话级锁内调用，以免并发请求
                交错写入；本类自身不负责并发加锁。
        """
        self.short_term.add("user", query)
        self.short_term.add("assistant", answer, sources=sources or [])
        self._turn_counter += 1
        await self._handle_overflow()

    # ── 溢出处理：短期 -> 中期压缩 ──────────────────────────────────
    async def _handle_overflow(self) -> None:
        """裁剪超出短期窗口的消息，并依次生成摘要和长期记忆。

        裁剪在摘要调用之前发生；摘要失败会被 ``MidTermMemory`` 记录并忽略，随后仍会
        尝试提取长期记忆。没有溢出时立即返回。

        Raises:
            Exception: 长期 Chroma 批量写入失败。
        """
        overflowed = self.short_term.trim_to_window()
        if not overflowed:
            return

        logger.info(f"[记忆管理] 窗口溢出，压缩 {len(overflowed)} 条历史消息")
        start_idx = max(0, self._turn_counter - len(overflowed))
        await self.mid_term.compress_turns(overflowed, start_idx)

        # 从溢出内容中提取关键信息存入长期记忆
        await self._extract_to_long_term(overflowed)

    # ── 长期记忆提取（从历史对话抽取实体/结论）────────────────────────
    async def _extract_to_long_term(self, turns) -> None:
        """用确定性规则从溢出消息提取实体、问题和简短结论并持久化。

        Args:
            turns: 已从短期窗口移除、按时间排序的 ``Turn`` 序列。

        Notes:
            引号内容和大写英文词会成为共享实体；所有用户问题以及少于 300 字符的
            助手回答会形成候选，本批最多写入前五项。

        Raises:
            Exception: 长期记忆批量写入失败。
        """
        conversation = "\n".join(
            f"{'用户' if t.role == 'user' else '助手'}：{t.content}" for t in turns
        )
        # 简单规则提取：识别业务术语（引号内、大写英文、驼峰命名等）
        entities = set()
        for turn in turns:
            # 提取引号内内容
            entities.update(re.findall(r'["「](.+?)["」]', turn.content))
            # 提取常见业务术语（可按需扩展）
            entities.update(re.findall(r'[A-Z][a-zA-Z]{2,}', turn.content))

        # 将用户的核心问题和助手的关键结论存入
        items = []
        for turn in turns:
            if turn.role == "user":
                items.append({
                    "content": f"用户问题：{turn.content}",
                    "category": "topic",
                    "entities": list(entities),
                })
            elif turn.role == "assistant" and len(turn.content) < 300:
                items.append({
                    "content": f"助手结论：{turn.content[:300]}",
                    "category": "conclusion",
                    "entities": list(entities),
                })

        if items:
            self.long_term.store_batch(items[:5], session_id=self.session_id)  # 每批最多5条

    # ── 构建上下文（供 Prompt 使用）──────────────────────────────────
    def build_context(self, current_query: str) -> dict:
        """用当前问题召回长期记忆，并组合三层上下文。

        Args:
            current_query: 用于长期向量召回的当前问题。

        Returns:
            包含 ``short_term_messages``、``mid_term_summary`` 和
            ``long_term_memories`` 的新字典，供 Prompt 构建器使用。

        Raises:
            Exception: Chroma 长期记忆查询失败。
        """
        # 长期记忆：以当前 query 进行语义召回
        long_term_results = self.long_term.recall(current_query)

        return {
            "short_term_messages": self.short_term.get_window_as_messages(),
            "mid_term_summary": self.mid_term.get_summaries_text(),
            "long_term_memories": long_term_results,
        }

    # ── 工具方法 ─────────────────────────────────────────────────────
    def clear_all(self, include_long_term: bool = False) -> None:
        """清空当前会话的短期和中期状态，并按需删除持久记忆。

        Args:
            include_long_term: 为 ``True`` 时，同时删除当前用户、当前会话的长期记录；
                不会删除该用户其他会话或其他用户的数据。
        """
        self.short_term.clear()
        self.mid_term.clear()
        if include_long_term:
            self.long_term.clear_session(self.session_id)
