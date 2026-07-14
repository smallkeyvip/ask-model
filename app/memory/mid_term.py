"""
中期摘要记忆（滚动摘要）：将溢出的对话压缩成摘要，避免上下文过长
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import get_settings
from app.llm import LLMClient

logger = logging.getLogger(__name__)
settings = get_settings

SUMMARY_SYSTEM_PROMPT = """你是一个专业的对话摘要助手。请将以下对话历史压缩成简洁的摘要，
保留关键信息、用户意图、重要结论和业务术语。
要求：
1. 摘要不超过 {max_tokens} 字
2. 保留所有实体名、产品名、业务术语
3. 保留用户的核心诉求和助手给出的关键结论
4. 使用结构化表述，分点列出
"""


@dataclass
class Summary:
    text: str
    turn_range: tuple   # (start_idx, end_idx) 被摘要覆盖的轮次区间
    created_at: float = field(default_factory=lambda: __import__("time").time())


class MidTermMemory:
    """
    滚动摘要：当短期窗口溢出时，调用 LLM 对溢出部分生成摘要，
    摘要结果保存在内存列表中，构建 Prompt 时拼入上下文。
    """

    def __init__(
        self,
        max_summary_tokens: int = 500,
        llm_client: Optional[LLMClient] = None,
    ):
        """初始化进程内滚动摘要存储。

        Args:
            max_summary_tokens: 单次摘要请求允许生成的最大 token 数，也会写入提示约束。
            llm_client: 可选的共享 LLM 客户端；未提供时创建新客户端。
        """
        self.max_summary_tokens = max_summary_tokens
        self._summaries: List[Summary] = []
        self._llm = llm_client or LLMClient()

    async def compress_turns(self, turns, start_idx: int) -> Optional[Summary]:
        """把一批溢出消息压缩成摘要并追加到中期列表。

        Args:
            turns: 按时间顺序排列、具有 ``role`` 和 ``content`` 属性的消息序列。
            start_idx: 该批消息在逻辑历史中的起始索引，用于记录覆盖范围。

        Returns:
            新建并已保存的 ``Summary``；输入为空或 LLM 调用失败时返回 ``None``。
            失败会记录日志，不向调用方抛出 LLM 异常。
        """
        if not turns:
            return None

        conversation_text = "\n".join(
            f"{'用户' if t.role == 'user' else '助手'}：{t.content}" for t in turns
        )

        prompt = SUMMARY_SYSTEM_PROMPT.format(max_tokens=self.max_summary_tokens)
        user_msg = f"请将以下对话历史压缩成摘要：\n\n{conversation_text}"

        try:
            summary_text = await self._call_llm(prompt, user_msg)
            summary = Summary(
                text=summary_text,
                turn_range=(start_idx, start_idx + len(turns) - 1),
            )
            self._summaries.append(summary)
            logger.info(f"[中期记忆] 生成摘要，覆盖轮次 {summary.turn_range}")
            return summary
        except Exception as e:
            logger.error(f"[中期记忆] 摘要生成失败: {e}")
            return None

    async def _call_llm(self, system_prompt: str, user_msg: str) -> str:
        """使用低温度和固定超时调用底层 LLM 生成摘要。

        Args:
            system_prompt: 包含摘要规则与长度限制的系统提示。
            user_msg: 包含待压缩对话的用户消息。

        Returns:
            模型生成的原始摘要文本。

        Raises:
            Exception: 底层 LLM 调用错误；由 :meth:`compress_turns` 捕获并降级。
        """
        return await self._llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=self.max_summary_tokens,
            timeout=60,
        )

    def get_summaries_text(self) -> str:
        """按生成顺序把所有滚动摘要拼成可注入 Prompt 的文本。

        Returns:
            带一基编号标题的摘要段落；尚无摘要时返回空字符串。
        """
        if not self._summaries:
            return ""
        parts = []
        for i, s in enumerate(self._summaries, 1):
            parts.append(f"[历史摘要{i}]\n{s.text}")
        return "\n\n".join(parts)

    @property
    def summaries(self) -> List[Summary]:
        """返回内部摘要列表的实时引用。

        调用方可观察后续追加，也能够修改内部状态；只读场景应避免原地变更该列表。
        """
        return self._summaries

    def clear(self) -> None:
        """原地删除当前进程中的全部中期摘要。

        清空后 :attr:`summaries` 的既有列表引用仍然有效但内容为空，
        :meth:`get_summaries_text` 会返回空字符串；不会影响短期或长期记忆。
        """
        self._summaries.clear()
