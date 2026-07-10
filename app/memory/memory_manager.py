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
        self.short_term.add("user", query)

    async def record_assistant(self, answer: str, sources: Optional[List[dict]] = None) -> None:
        """记录助手回复，同时触发溢出压缩和长期记忆提取"""
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
        """成功完成回答后原子地追加一轮完整对话。"""
        self.short_term.add("user", query)
        self.short_term.add("assistant", answer, sources=sources or [])
        self._turn_counter += 1
        await self._handle_overflow()

    # ── 溢出处理：短期 -> 中期压缩 ──────────────────────────────────
    async def _handle_overflow(self) -> None:
        """当短期记忆超出窗口时，将溢出部分交给中期摘要"""
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
        """从溢出的对话中提取关键实体和结论，存入 Chroma"""
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
        """
        融合三层记忆，返回结构化上下文：
        {
            "short_term_messages": [...],   # 最近 N 轮完整对话
            "mid_term_summary": str,         # 滚动摘要
            "long_term_memories": [...],     # 相关历史记忆
        }
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
        self.short_term.clear()
        self.mid_term.clear()
        if include_long_term:
            self.long_term.clear_session(self.session_id)
